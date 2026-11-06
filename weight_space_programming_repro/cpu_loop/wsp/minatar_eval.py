"""MinAtar scoring for exp5: mean score over N episodes, actions sampled from softmax(logits), sticky actions 0.1,
difficulty ramping on, episodes cut at 10,000 steps.

Uses the official `minatar` package (1.0.15) when it is installed, otherwise the game's source file exactly as it is
quoted in the prompt (written to <runs>/exp5/env_<game>.py) with the same sticky-action wrapper.
"""
import importlib.util, os, time
from multiprocessing import Pool
import numpy as np

# 6 semantic channels <- the game's own channels (from each prompt, "THE INPUT").
CHANNEL_MAP = {
    "breakout": [["paddle"], [], ["brick"], ["ball"], ["trail"], []],
    "seaquest": [["sub_front", "sub_back"], ["enemy_fish", "enemy_sub", "enemy_bullet"],
                 ["enemy_fish", "enemy_sub", "diver"], ["friendly_bullet"], ["trail", "sub_back"],
                 ["oxygen_guage", "diver_guage"]],
    "asterix": [["player"], ["enemy"], ["gold"], [], ["trail"], []],
}
MAX_STEPS, STICKY = 10_000, 0.1


class _QuotedEnv:
    """The quoted game file plus MinAtar's Environment wrapper behaviour (sticky actions, ramping)."""

    def __init__(self, src, seed):
        spec = importlib.util.spec_from_file_location(f"quoted_{os.getpid()}_{seed}", src)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        self.random = np.random.RandomState(seed)
        self.env = m.Env(ramping=True)
        self.env.random = self.random
        self.last_action = 0

    def reset(self):
        self.env.reset()
        self.last_action = 0

    def state(self):
        return self.env.state()

    def act(self, a):
        if self.random.rand() < STICKY:
            a = self.last_action
        self.last_action = a
        return self.env.act(a)


def _make_env(game, src, seed):
    try:
        from minatar import Environment
        env = Environment(game, sticky_action_prob=STICKY, difficulty_ramping=True)
        env.seed(seed)   # MinAtar 1.0.15: one RandomState shared by the sticky actions and the game
        return env, env.env.channels, "minatar"
    except ImportError:
        e = _QuotedEnv(src, seed)
        return e, e.env.channels, "quoted-source"


def _episodes(args):
    game, src, weights, seeds = args
    W1, b1, W2, b2, W3, b3 = (weights[k] for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
    res = []
    for s in seeds:
        env, ch, kind = _make_env(game, src, s)
        idx = [[ch[n] for n in grp] for grp in CHANNEL_MAP[game]]
        rng = np.random.RandomState(1_000_003 + s)
        env.reset()
        total, steps, done = 0.0, 0, False
        while not done and steps < MAX_STEPS:
            st = env.state()
            x = np.zeros((10, 10, 6), np.float32)
            for c, g in enumerate(idx):
                if g:
                    x[:, :, c] = st[:, :, g].any(-1)
            h = np.tanh(x.reshape(-1) @ W1 + b1)
            h = np.tanh(h @ W2 + b2)
            z = (h @ W3 + b3).astype(np.float64)
            p = np.exp(z - z.max())
            a = rng.choice(6, p=p / p.sum())
            r, done = env.act(a)
            total += r
            steps += 1
        res.append((total, steps, kind))
    return res


def evaluate(game, weights, src, episodes=1000, workers=0):
    workers = workers or os.cpu_count() or 1
    t0 = time.time()
    chunks = [(game, src, weights, list(range(i, episodes, workers))) for i in range(workers)]
    with Pool(workers) as pool:
        res = [r for part in pool.map(_episodes, chunks) for r in part]
    sc = np.array([r[0] for r in res])
    ln = np.array([r[1] for r in res])
    return {"mean": float(sc.mean()), "se": float(sc.std(ddof=1) / np.sqrt(len(sc))), "episodes": len(sc),
            "max": float(sc.max()), "mean_length": float(ln.mean()), "env": res[0][2],
            "seconds": round(time.time() - t0, 1)}
