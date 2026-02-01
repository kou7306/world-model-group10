import os
import numpy as np
import socket
from gym import spaces
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel


def get_available_port(start_port=5005, end_port=5100):
    while True:
        port = np.random.randint(start_port, end_port)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port


LIDAR_FRONT_SIZE = 20  # 前方LiDAR: 20個
LIDAR_BACK_SIZE = 20   # 後方LiDAR: 20個
AUDIO_SIZE = 8         # 方向別音量分布 directionCount=8, 各0.0～1.0
TARGET_SIZE = 3        # ターゲット相対位置 (x,y,z)
VECTOR_OBS_SIZE = LIDAR_FRONT_SIZE + LIDAR_BACK_SIZE + AUDIO_SIZE + TARGET_SIZE  # 51

# 観測の並び: 前方LiDAR [0:20] → 後方LiDAR [20:40] → Audio [40:48] → Target [48:51]
OBS_ORDER = "LiDAR_Front → LiDAR_Back → Audio → Target"
LIDAR_FRONT_START, LIDAR_FRONT_END = 0, LIDAR_FRONT_SIZE
LIDAR_BACK_START, LIDAR_BACK_END = LIDAR_FRONT_END, LIDAR_FRONT_END + LIDAR_BACK_SIZE
AUDIO_START, AUDIO_END = LIDAR_BACK_END, LIDAR_BACK_END + AUDIO_SIZE
TARGET_START, TARGET_END = AUDIO_END, AUDIO_END + TARGET_SIZE


class UnityEnv:
    def __init__(self, action_repeat=1, seed=None, retries=3, time_scale=20.0, id=0, debug_obs=None, debug_reward=False, use_audio=True):
        self.retries = retries
        self.id = id
        self.timeout_wait = 180
        self._debug_obs = (os.environ.get("UNITY_DEBUG_OBS", "0").lower() in ("1", "true", "yes")
                          if debug_obs is None else debug_obs)
        self._debug_obs_count = 0
        self._debug_reward = debug_reward
        self._use_audio = use_audio
        self._init_env(action_repeat, seed, time_scale)

    def _init_env(self, action_repeat, seed, time_scale):
        for attempt in range(self.retries):
            try:
                self.channel = EngineConfigurationChannel()
                self.stats_channel = StatsSideChannel()
                print(f"Attempt {attempt + 1}: Initializing Unity Environment...")
                
                self.channel.set_configuration_parameters(width=420, height=280, quality_level=1)
                self.set_time_scale(time_scale)

                base_port = get_available_port()
                print(f"Base port: {base_port}")

                side_channels = [self.channel, self.stats_channel]
                if self.id < 10:
                    self.env = UnityEnvironment(file_name="UnityBuildMulti", base_port=base_port, side_channels=side_channels, timeout_wait=180)
                else:
                    self.env = UnityEnvironment(file_name="UnityBuildMulti", base_port=base_port, side_channels=side_channels, timeout_wait=180)
                self.env.reset()
                print("Environment successfully initialized.")
                print(f"Obs: {OBS_ORDER} | LiDAR_Front={LIDAR_FRONT_SIZE}, LiDAR_Back={LIDAR_BACK_SIZE}, Audio={AUDIO_SIZE}, Target={TARGET_SIZE} (total={VECTOR_OBS_SIZE})")
                if self._debug_obs:
                    print("[UnityObs] UNITY_DEBUG_OBS=1: 観測のダンプを有効にしました")

                self._action_repeat = action_repeat
                self._seed = seed

                behavior_name = list(self.env.behavior_specs)[0]
                self.spec = self.env.behavior_specs[behavior_name]
                self._behavior_name = behavior_name
                
                self._action_dim = self.spec.action_spec.continuous_size
                break
            except Exception as e:
                print(f"Initialization failed on attempt {attempt + 1} with error: {e}")
                if attempt == self.retries - 1:
                    raise

    @property
    def observation_space(self):
        obs_space = {
            "lidar_front": spaces.Box(low=-np.inf, high=np.inf, shape=(LIDAR_FRONT_SIZE,), dtype=np.float32),
            "lidar_back": spaces.Box(low=-np.inf, high=np.inf, shape=(LIDAR_BACK_SIZE,), dtype=np.float32),
            "target": spaces.Box(low=-np.inf, high=np.inf, shape=(TARGET_SIZE,), dtype=np.float32),
        }
        if self._use_audio:
            obs_space["audio"] = spaces.Box(low=0.0, high=1.0, shape=(AUDIO_SIZE,), dtype=np.float32)
        return spaces.Dict(obs_space)

    @property
    def action_space(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(self._action_dim,), dtype=np.float32)

    def get_unity_stats(self):
        stats = {}
        try:
            raw_stats = self.stats_channel.get_and_reset_stats()
            for key, value_list in raw_stats.items():
                if value_list:
                    numeric_values = []
                    for v in value_list:
                        if isinstance(v, (tuple, list)):
                            if len(v) > 0:
                                try:
                                    numeric_values.append(float(v[0]))
                                except (TypeError, ValueError):
                                    pass
                        elif isinstance(v, (int, float)):
                            numeric_values.append(float(v))
                    
                    if numeric_values:
                        stats[key] = sum(numeric_values) / len(numeric_values)
        except Exception as e:
            print(f"Warning: Failed to get Unity stats: {e}")
        return stats

    def step(self, action):
        total_reward = 0.
        is_terminate = False
        last_decision_steps = None
        
        action_tuple = ActionTuple(continuous=np.array([action]))
        self.env.set_actions(self._behavior_name, action_tuple)
        self.env.step()
        decision_steps, terminal_steps = self.env.get_steps(self._behavior_name)
        
        if len(terminal_steps) > 0:
            is_terminate = True
            if len(terminal_steps.reward) > 0:
                reward = terminal_steps.reward[0]
        elif len(decision_steps.reward) > 0:
            reward = decision_steps.reward[0]
        else:
            reward = 0.0
            
        total_reward += reward
        last_decision_steps = decision_steps

        if self._debug_reward:
            print(f"[UnityReward] id={self.id} reward={total_reward} done={is_terminate}")

        unity_stats = self.get_unity_stats()
        obs = self._create_observation(last_decision_steps, terminal_steps, is_terminate)
        
        info = {"unity_stats": unity_stats} if unity_stats else {}
        return obs, total_reward, is_terminate, info

    def _get_vector_obs(self, steps):
        if steps is None or len(steps) == 0 or len(steps.obs) == 0:
            return np.zeros(VECTOR_OBS_SIZE, dtype=np.float32)
        o = np.asarray(steps.obs[0])
        if o.size == 0:
            return np.zeros(VECTOR_OBS_SIZE, dtype=np.float32)
        if o.ndim >= 2 and o.shape[0] > 0:
            row = o[0]
        else:
            row = o
        return np.array(row, dtype=np.float32).flatten()

    def _create_observation(self, decision_steps, terminal_steps, is_terminate):
        if is_terminate and terminal_steps is not None and len(terminal_steps) > 0:
            vector_obs = self._get_vector_obs(terminal_steps)
        else:
            vector_obs = self._get_vector_obs(decision_steps)

        vector_obs = np.array(vector_obs, dtype=np.float32).flatten()
        n = len(vector_obs)

        # 並び: 前方LiDAR [0:20] → 後方LiDAR [20:40] → Audio [40:48] → Target [48:51]
        lidar_front = np.zeros(LIDAR_FRONT_SIZE, dtype=np.float32)
        if n > LIDAR_FRONT_START:
            lidar_front[: min(LIDAR_FRONT_SIZE, n - LIDAR_FRONT_START)] = vector_obs[LIDAR_FRONT_START : min(LIDAR_FRONT_END, n)]

        lidar_back = np.zeros(LIDAR_BACK_SIZE, dtype=np.float32)
        if n > LIDAR_BACK_START:
            lidar_back[: min(LIDAR_BACK_SIZE, n - LIDAR_BACK_START)] = vector_obs[LIDAR_BACK_START : min(LIDAR_BACK_END, n)]

        audio = np.zeros(AUDIO_SIZE, dtype=np.float32)
        if n > AUDIO_START:
            audio[: min(AUDIO_SIZE, n - AUDIO_START)] = vector_obs[AUDIO_START : min(AUDIO_END, n)]
        audio = np.clip(audio, 0.0, 1.0)

        target = np.zeros(TARGET_SIZE, dtype=np.float32)
        if n > TARGET_START:
            target[: min(TARGET_SIZE, n - TARGET_START)] = vector_obs[TARGET_START : min(TARGET_END, n)]

        if self._debug_obs:
            self._debug_obs_count += 1
            if self._use_audio:
                print(f"[Obs] id={self.id} step={self._debug_obs_count} lidar_f={lidar_front.tolist()[:5]}... lidar_b={lidar_back.tolist()[:5]}... audio={audio.tolist()} target={target.tolist()}")
            else:
                print(f"[Obs] id={self.id} step={self._debug_obs_count} lidar_f={lidar_front.tolist()[:5]}... lidar_b={lidar_back.tolist()[:5]}... target={target.tolist()}")

        obs = {
            "lidar_front": lidar_front,
            "lidar_back": lidar_back,
            "target": target,
            "is_first": False,
            "is_terminal": is_terminate,
        }
        if self._use_audio:
            obs["audio"] = audio
        return obs


    def reset(self):
        self.env.reset()
        decision_steps, terminal_steps = self.env.get_steps(self._behavior_name)
        obs = self._create_observation(decision_steps, terminal_steps, is_terminate=False)
        obs["is_first"] = True
        return obs

    def close(self):
        self.env.close()
        print("Unity environment closed.")

    def set_time_scale(self, time_scale: float):
        self.channel.set_configuration_parameters(time_scale=time_scale)
        print(f"Set time scale to {time_scale}")
