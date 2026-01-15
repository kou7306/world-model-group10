import gym
import numpy as np


class AudioToy(gym.Env):
    """
    Simple toy environment emitting an audio vector correlated with latent state.
    Audio is a low-dimensional embedding (e.g., MFCC-like) so the current MLP path can learn it.
    Image is a blank canvas to satisfy existing vision configs.
    """

    metadata = {"render.modes": []}

    def __init__(self, size=(64, 64), audio_dim=16, horizon=200, seed=0):
        super().__init__()
        self.size = size
        self.audio_dim = audio_dim
        self.horizon = horizon
        self._rng = np.random.RandomState(seed)
        self._step = 0
        # Continuous 2D action to steer latent phase and amplitude
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(*size, 3), dtype=np.uint8
                ),
                "audio": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(audio_dim,), dtype=np.float32
                ),
            }
        )
        self._latent_phase = 0.0
        self._latent_amp = 0.5

    def _make_audio(self):
        # Generate a smooth audio-like vector from latent phase/amplitude with noise.
        idx = np.arange(self.audio_dim, dtype=np.float32)
        base = np.sin(self._latent_phase + idx * 0.2) * self._latent_amp
        noise = 0.05 * self._rng.randn(self.audio_dim).astype(np.float32)
        return (base + noise).astype(np.float32)

    def reset(self):
        self._step = 0
        self._latent_phase = self._rng.uniform(0, 2 * np.pi)
        self._latent_amp = self._rng.uniform(0.3, 1.0)
        obs = {
            "image": np.zeros((*self.size, 3), dtype=np.uint8),
            "audio": self._make_audio(),
            "is_first": True,
            "is_terminal": False,
        }
        return obs

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        # Update latent state: phase advances with action[0], amplitude with action[1]
        self._latent_phase += 0.2 + 0.3 * float(action[0])
        self._latent_amp = np.clip(self._latent_amp + 0.05 * float(action[1]), 0.1, 1.2)
        self._step += 1
        obs = {
            "image": np.zeros((*self.size, 3), dtype=np.uint8),
            "audio": self._make_audio(),
            "is_first": False,
            "is_terminal": False,
        }
        # Reward encourages keeping audio energy small (act to minimize amplitude)
        reward = -np.mean(np.square(obs["audio"]))
        done = self._step >= self.horizon
        if done:
            obs["is_terminal"] = True
        info = {"discount": np.array(1.0, dtype=np.float32)}
        return obs, reward, done, info
