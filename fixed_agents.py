"""Non-learning players with a fixed, hand-written strategy.
They join the population and play the learners (and each other), but never change."""
import json
import os

import numpy as np

C, D = 0, 1


def tit_for_tat(hist, rng, p):
    return hist[-1][1] if hist else C


def generous_tit_for_tat(hist, rng, p):
    if not hist or hist[-1][1] == C:
        return C
    return C if rng.random() < p["generosity"] else D  # sometimes forgives a defection


def grim_trigger(hist, rng, p):
    return D if any(o == D for _, o in hist) else C  # one defection = defect for the rest of the game


STRATEGIES = {
    "Tit for Tat": tit_for_tat,
    "Generous Tit for Tat": generous_tit_for_tat,
    "Grim Trigger": grim_trigger,
}


class FixedAgent:
    """Same interface as DQNAgent so the arena can treat every player alike."""
    is_learner = False
    epsilon = 0.0
    buf_size = 0

    def __init__(self, name, directory, seed, params):
        self.name, self.dir = name, directory
        self.fn = STRATEGIES[name]
        self.params = params
        self.rng = np.random.default_rng(seed)
        self.games_played = 0
        self.total_moves = 0
        self.created_at = None

    def act(self, obs, greedy=False, hist=None):
        return self.fn(hist or [], self.rng, self.params), None, False

    def remember(self, s, a, r, s2, done):
        self.total_moves += 1  # nothing to learn

    def pop_game_stats(self):
        return dict(updates=0, mean_loss="", mean_q="")

    @property
    def _state_path(self):
        return os.path.join(self.dir, "state.json")

    def save(self):
        os.makedirs(self.dir, exist_ok=True)
        tmp = self._state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(dict(name=self.name, games_played=self.games_played, total_moves=self.total_moves,
                           created_at=self.created_at, rng_state=self.rng.bit_generator.state), f)
        os.replace(tmp, self._state_path)

    def load(self):
        if not os.path.exists(self._state_path):
            return False
        with open(self._state_path) as f:
            s = json.load(f)
        self.games_played, self.total_moves = s["games_played"], s["total_moves"]
        self.created_at = s.get("created_at")
        try:
            self.rng.bit_generator.state = s["rng_state"]
        except Exception:
            pass
        return True
