"""Minimal fully-connected neural network in pure NumPy (ReLU MLP + Adam)."""
import numpy as np


class MLP:
    def __init__(self, sizes, rng):
        self.sizes = list(sizes)
        self.params = []
        for i in range(len(sizes) - 1):
            W = rng.normal(0.0, np.sqrt(2.0 / sizes[i]), (sizes[i], sizes[i + 1])).astype(np.float32)
            b = np.zeros(sizes[i + 1], np.float32)
            self.params += [W, b]
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0
        self.n_layers = len(self.params) // 2

    def forward(self, x):
        """Returns output and the list of activations (needed for backward)."""
        acts = [x]
        h = x
        for i in range(self.n_layers):
            W, b = self.params[2 * i], self.params[2 * i + 1]
            h = h @ W + b
            if i < self.n_layers - 1:
                h = np.maximum(h, 0.0)
            acts.append(h)
        return h, acts

    def predict(self, x):
        return self.forward(x)[0]

    def backward(self, acts, grad_out):
        grads = [None] * len(self.params)
        g = grad_out
        for i in reversed(range(self.n_layers)):
            grads[2 * i] = acts[i].T @ g
            grads[2 * i + 1] = g.sum(axis=0)
            if i > 0:
                g = (g @ self.params[2 * i].T) * (acts[i] > 0)
        return grads

    def adam_step(self, grads, lr, b1=0.9, b2=0.999, eps=1e-8, clip=10.0):
        norm = np.sqrt(sum(float((g * g).sum()) for g in grads))
        if norm > clip:
            grads = [g * (clip / norm) for g in grads]
        self.t += 1
        c1 = 1 - b1 ** self.t
        c2 = 1 - b2 ** self.t
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = b1 * self.m[i] + (1 - b1) * g
            self.v[i] = b2 * self.v[i] + (1 - b2) * g * g
            p -= lr * (self.m[i] / c1) / (np.sqrt(self.v[i] / c2) + eps)
        return norm

    def copy_from(self, other):
        self.params = [p.copy() for p in other.params]

    # --- serialization -------------------------------------------------
    def to_arrays(self, prefix, with_optimizer=True):
        out = {f"{prefix}sizes": np.array(self.sizes)}
        for i, p in enumerate(self.params):
            out[f"{prefix}p{i}"] = p
            if with_optimizer:
                out[f"{prefix}m{i}"] = self.m[i]
                out[f"{prefix}v{i}"] = self.v[i]
        if with_optimizer:
            out[f"{prefix}t"] = np.array(self.t)
        return out

    def load_arrays(self, data, prefix, with_optimizer=True):
        sizes = [int(s) for s in data[f"{prefix}sizes"]]
        if sizes != self.sizes:
            raise ValueError(
                f"Saved network shape {sizes} does not match configured shape {self.sizes}. "
                "Revert the config change or delete that agent's data folder to start over."
            )
        n = len(self.params)
        self.params = [data[f"{prefix}p{i}"].astype(np.float32) for i in range(n)]
        if with_optimizer:
            self.m = [data[f"{prefix}m{i}"].astype(np.float32) for i in range(n)]
            self.v = [data[f"{prefix}v{i}"].astype(np.float32) for i in range(n)]
            self.t = int(data[f"{prefix}t"])
