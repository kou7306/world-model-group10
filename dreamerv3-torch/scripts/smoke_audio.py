"""
Minimal smoke test to verify audio modality plumbing works end-to-end.
- Builds a WorldModel on CPU with a dummy observation/action space (image + audio).
- Runs a single training step on random data to ensure forward/backward succeeds.
"""

import pathlib
from types import SimpleNamespace

import numpy as np
import ruamel.yaml as yaml
import torch
import gym

import models
import tools


def load_default_config():
    config_path = pathlib.Path(__file__).parents[1] / "configs.yaml"
    configs = yaml.safe_load(config_path.read_text())
    defaults = configs["defaults"]
    cfg = SimpleNamespace(**defaults)
    # Overrides for a quick CPU smoke run.
    cfg.device = "cpu"
    cfg.compile = False
    cfg.video_pred_log = False
    cfg.batch_size = 2
    cfg.batch_length = 4
    cfg.precision = 32
    cfg.envs = 1
    cfg.train_ratio = 1
    cfg.audio_dim = max(cfg.audio_dim, 16)
    cfg.audio_noise_std = cfg.audio_noise_std or 0.0
    cfg.logdir = "./logdir/smoke_audio"
    cfg.traindir = pathlib.Path(cfg.logdir) / "train_eps"
    cfg.evaldir = pathlib.Path(cfg.logdir) / "eval_eps"
    cfg.grad_heads = ["decoder", "reward", "cont"]
    cfg.reward_head["dist"] = "symlog_disc"
    cfg.cont_head["loss_scale"] = 1.0
    return cfg


def make_dummy_spaces(cfg):
    obs_space = gym.spaces.Dict(
        {
            "image": gym.spaces.Box(
                low=0, high=255, shape=tuple(cfg.size) + (3,), dtype=np.uint8
            ),
            "audio": gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(cfg.audio_dim,),
                dtype=np.float32,
            ),
            "is_first": gym.spaces.Box(0, 1, shape=(), dtype=bool),
            "is_terminal": gym.spaces.Box(0, 1, shape=(), dtype=bool),
            "discount": gym.spaces.Box(0, 1, shape=(), dtype=np.float32),
            "reward": gym.spaces.Box(-np.inf, np.inf, shape=(), dtype=np.float32),
        }
    )
    action_dim = 3
    act_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    cfg.num_actions = action_dim
    return obs_space, act_space


def make_fake_batch(cfg, obs_space, act_space):
    B, T = cfg.batch_size, cfg.batch_length
    obs = {}
    obs["image"] = np.random.randint(
        0, 256, size=(B, T) + obs_space["image"].shape, dtype=np.uint8
    )
    obs["audio"] = np.random.randn(B, T, cfg.audio_dim).astype(np.float32)
    obs["action"] = np.random.uniform(
        low=act_space.low, high=act_space.high, size=(B, T, act_space.shape[0])
    ).astype(np.float32)
    obs["is_first"] = np.zeros((B, T), dtype=bool)
    obs["is_terminal"] = np.zeros((B, T), dtype=bool)
    obs["discount"] = np.ones((B, T), dtype=np.float32)
    obs["reward"] = np.random.randn(B, T).astype(np.float32)
    return obs


def main():
    cfg = load_default_config()
    obs_space, act_space = make_dummy_spaces(cfg)
    wm = models.WorldModel(obs_space, act_space, step=0, config=cfg).to(cfg.device)
    batch = make_fake_batch(cfg, obs_space, act_space)
    with torch.no_grad():
        # Ensure preprocess and encoder run.
        _ = wm.preprocess({k: v.copy() for k, v in batch.items()})
    post, context, metrics = wm._train(batch)
    print("Smoke test succeeded. Sample metrics:")
    for key in ["model_loss", "reward_loss", "cont_loss", "kl"]:
        if key in metrics:
            print(f"  {key}: {np.mean(metrics[key]):.4f}")


if __name__ == "__main__":
    main()
