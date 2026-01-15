import gym
import numpy as np


class AudioVisionToy(gym.Env):
    """
    Simple multimodal toy env.
    - Audio: low-dim vector derived from latent phase/amplitude with noise.
    - Image: colored square whose hue/brightness correlates with the same latent state.
    This lets you verify cross-modal consistency (audio + image) in the world model.
    """

    metadata = {"render.modes": []}

    def __init__(self, size=(64, 64), audio_dim=16, horizon=200, seed=0):
        super().__init__()
        self.size = size
        self.audio_dim = audio_dim
        self.horizon = horizon
        self._rng = np.random.RandomState(seed)
        self._step = 0
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
        idx = np.arange(self.audio_dim, dtype=np.float32)
        base = np.sin(self._latent_phase + idx * 0.25) * self._latent_amp
        noise = 0.05 * self._rng.randn(self.audio_dim).astype(np.float32)
        return (base + noise).astype(np.float32)

    def _make_image(self):
        h, w = self.size
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # Square size tied to amplitude, color tied to phase.
        square = max(4, int(min(h, w) * (0.2 + 0.5 * self._latent_amp)))
        start = (h // 2 - square // 2, w // 2 - square // 2)
        end = (start[0] + square, start[1] + square)
        hue = (self._latent_phase % (2 * np.pi)) / (2 * np.pi)
        color = np.array(
            [
                127 + 100 * np.sin(2 * np.pi * hue),
                127 + 100 * np.sin(2 * np.pi * (hue + 1 / 3)),
                127 + 100 * np.sin(2 * np.pi * (hue + 2 / 3)),
            ],
            dtype=np.float32,
        )
        color = np.clip(color, 0, 255).astype(np.uint8)
        img[start[0] : end[0], start[1] : end[1], :] = color
        return img

    def reset(self):
        self._step = 0
        self._latent_phase = self._rng.uniform(0, 2 * np.pi)
        self._latent_amp = self._rng.uniform(0.3, 1.0)
        obs = {
            "image": self._make_image(),
            "audio": self._make_audio(),
            "is_first": True,
            "is_terminal": False,
        }
        return obs

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        self._latent_phase += 0.25 + 0.35 * float(action[0])
        self._latent_amp = np.clip(self._latent_amp + 0.05 * float(action[1]), 0.1, 1.2)
        self._step += 1
        obs = {
            "image": self._make_image(),
            "audio": self._make_audio(),
            "is_first": False,
            "is_terminal": False,
        }
        reward = -np.mean(np.square(obs["audio"]))
        done = self._step >= self.horizon
        if done:
            obs["is_terminal"] = True
        info = {"discount": np.array(1.0, dtype=np.float32)}
        return obs, reward, done, info
