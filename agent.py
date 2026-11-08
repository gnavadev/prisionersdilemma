"""Self-learning Double-DQN agent with its own persistent storage folder."""
import json
import math
import os

import numpy as np

from nn import MLP

C, D = 0, 1


def obs_dim(history_len):
    return 4 * history_len + 3


def encode_obs(history, my_score, opp_score, history_len):
    """history: list of (my_move, opp_move) from this agent's point of view."""
    x = np.zeros(obs_dim(history_len), dtype=np.float32)
    recent = history[-history_len:][::-1]  # most recent first
    for k, (m, o) in enumerate(recent):
        x[4 * k + m] = 1.0          # my move one-hot
        x[4 * k + 2 + o] = 1.0      # opponent move one-hot
    base = 4 * history_len
    x[base] = math.tanh((my_score - opp_score) / 15.0)  # who is ahead, and by how much
    if history:
        n = len(history)
        x[base + 1] = sum(1 for m, _ in history if m == C) / n
        x[base + 2] = sum(1 for _, o in history if o == C) / n
    else:
        x[base + 1] = x[base + 2] = 0.5
    return x


class DQNAgent:
    is_learner = True

    def __init__(self, name, directory, cfg, history_len, hidden):
        self.name = name
        self.dir = directory
        self.cfg = cfg
        self.history_len = history_len
        self.dim = obs_dim(history_len)
        self.rng = np.random.default_rng(cfg["seed"])
        sizes = [self.dim] + list(hidden) + [2]
        self.net = MLP(sizes, self.rng)
        self.target = MLP(sizes, self.rng)
        self.target.copy_from(self.net)

        cap = cfg["replay_capacity"]
        self.buf_s = np.zeros((cap, self.dim), np.float32)
        self.buf_a = np.zeros(cap, np.int8)
        self.buf_r = np.zeros(cap, np.float32)
        self.buf_s2 = np.zeros((cap, self.dim), np.float32)
        self.buf_d = np.zeros(cap, np.float32)
        self.buf_ptr = 0
        self.buf_size = 0

        self.total_moves = 0
        self.total_updates = 0
        self.games_played = 0
        self.created_at = None
        self.gpu = None  # (GPUPopulation, index) when the GPU engine owns this network
        self.reset_game_stats()

    # ----------------------------------------------------------- acting
    @property
    def epsilon(self):
        c = self.cfg
        return c["eps_min"] + (c["eps_start"] - c["eps_min"]) * math.exp(
            -self.total_moves / c["eps_decay_steps"])

    def q_values(self, obs):
        if self.gpu is not None:
            return self.gpu[0].q_values(self.gpu[1], obs)
        return self.net.predict(obs[None, :])[0]

    def act(self, obs, greedy=False, hist=None):
        q = self.q_values(obs)
        if not greedy and self.rng.random() < self.epsilon:
            return int(self.rng.integers(2)), q, True
        return int(np.argmax(q)), q, False

    # --------------------------------------------------------- learning
    def remember(self, s, a, r, s2, done):
        i = self.buf_ptr
        self.buf_s[i], self.buf_a[i], self.buf_r[i] = s, a, r * self.cfg.get("reward_scale", 1.0)
        self.buf_s2[i], self.buf_d[i] = s2, float(done)
        self.buf_ptr = (i + 1) % len(self.buf_a)
        self.buf_size = min(self.buf_size + 1, len(self.buf_a))
        self.total_moves += 1
        if self.buf_size >= self.cfg["learn_start"] and self.total_moves % self.cfg["train_every"] == 0:
            self.train_step()

    def train_step(self):
        c = self.cfg
        idx = self.rng.integers(0, self.buf_size, c["batch_size"])
        s = self.buf_s[idx]
        a = self.buf_a[idx].astype(np.int64)
        r = self.buf_r[idx]
        s2 = self.buf_s2[idx]
        d = self.buf_d[idx]
        n = len(a)

        # one online pass over [s; s2]: Q(s) for the update, Q(s2) for action selection
        q_all, acts_all = self.net.forward(np.concatenate([s, s2]))
        q = q_all[:n]
        acts = [x[:n] for x in acts_all]
        # Double DQN target: online net picks the action, target net scores it
        a2 = np.argmax(q_all[n:], axis=1)
        q2 = self.target.predict(s2)[np.arange(n), a2]
        y = r + c["gamma"] * (1.0 - d) * q2

        rows = np.arange(len(a))
        td = q[rows, a] - y
        # Huber loss gradient
        g = np.clip(td, -1.0, 1.0) / len(a)
        grad_out = np.zeros_like(q)
        grad_out[rows, a] = g
        grads = self.net.backward(acts, grad_out)
        self.net.adam_step(grads, c["lr"])

        self.total_updates += 1
        if self.total_updates % c["target_update_every"] == 0:
            self.target.copy_from(self.net)

        absd = np.abs(td)
        loss = np.where(absd <= 1.0, 0.5 * td ** 2, absd - 0.5).mean()
        self.g_loss_sum += float(loss)
        self.g_q_sum += float(q[rows, a].mean())
        self.g_updates += 1

    def reset_game_stats(self):
        self.g_loss_sum = 0.0
        self.g_q_sum = 0.0
        self.g_updates = 0

    def pop_game_stats(self):
        n = max(self.g_updates, 1)
        out = dict(
            updates=self.g_updates,
            mean_loss=self.g_loss_sum / n if self.g_updates else "",
            mean_q=self.g_q_sum / n if self.g_updates else "",
        )
        self.reset_game_stats()
        return out

    # ------------------------------------------------------ persistence
    @property
    def checkpoint_path(self):
        return os.path.join(self.dir, "checkpoint.npz")

    def save(self):
        os.makedirs(self.dir, exist_ok=True)
        meta = dict(
            name=self.name,
            total_moves=self.total_moves,
            total_updates=self.total_updates,
            games_played=self.games_played,
            buf_ptr=self.buf_ptr,
            buf_size=self.buf_size,
            created_at=self.created_at,
            rng_state=self.rng.bit_generator.state,
        )
        arrays = {}
        arrays.update(self.net.to_arrays("net_"))
        arrays.update(self.target.to_arrays("tgt_", with_optimizer=False))
        n = self.buf_size
        arrays.update(buf_s=self.buf_s[:n], buf_a=self.buf_a[:n], buf_r=self.buf_r[:n],
                      buf_s2=self.buf_s2[:n], buf_d=self.buf_d[:n])
        arrays["meta"] = np.array(json.dumps(meta))
        tmp = self.checkpoint_path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(f, **arrays)
            f.flush()
            os.fsync(f.fileno())
        if os.path.exists(self.checkpoint_path):  # keep one backup
            os.replace(self.checkpoint_path, os.path.join(self.dir, "checkpoint.prev.npz"))
        os.replace(tmp, self.checkpoint_path)

    def load(self):
        """Returns True if a checkpoint was loaded, False if starting fresh."""
        for path in (self.checkpoint_path, os.path.join(self.dir, "checkpoint.prev.npz")):
            if not os.path.exists(path):
                continue
            try:
                with np.load(path, allow_pickle=False) as data:
                    self.net.load_arrays(data, "net_")
                    self.target.load_arrays(data, "tgt_", with_optimizer=False)
                    meta = json.loads(str(data["meta"][()]))
                    n = min(int(meta["buf_size"]), len(self.buf_a))
                    self.buf_s[:n] = data["buf_s"][:n]
                    self.buf_a[:n] = data["buf_a"][:n]
                    self.buf_r[:n] = data["buf_r"][:n]
                    self.buf_s2[:n] = data["buf_s2"][:n]
                    self.buf_d[:n] = data["buf_d"][:n]
            except ValueError as e:
                if "does not match" in str(e):
                    raise
                print(f"[{self.name}] could not read {path}: {e}")
                continue
            except Exception as e:  # corrupted file -> try backup
                print(f"[{self.name}] could not read {path}: {e}")
                continue
            self.buf_size = n
            self.buf_ptr = int(meta["buf_ptr"]) % len(self.buf_a) if n == len(self.buf_a) else n
            self.total_moves = int(meta["total_moves"])
            self.total_updates = int(meta["total_updates"])
            self.games_played = int(meta["games_played"])
            self.created_at = meta.get("created_at")
            try:
                self.rng.bit_generator.state = meta["rng_state"]
            except Exception:
                pass
            if path != self.checkpoint_path:
                print(f"[{self.name}] main checkpoint unreadable, restored backup")
            return True
        return False
