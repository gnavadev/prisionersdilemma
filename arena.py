#!/usr/bin/env python3
"""A population of self-learning neural networks playing the Iterated Prisoner's Dilemma.

Every game pairs two random agents. Each agent starts playing randomly and
develops its own strategy purely from experience. Runs until Ctrl+C;
everything is saved in ./data and the next run continues from there.

    .venv/bin/python arena.py               # run forever
    .venv/bin/python arena.py --games 5000  # stop after 5000 games
"""
import argparse
import csv
import json
import os
import signal
import time
from datetime import datetime

import numpy as np

import config as cfg
from agent import C, D, DQNAgent, encode_obs
from fixed_agents import FixedAgent

MOVE = "CD"          # network's own choice
MOVE_RANDOM = "cd"   # random exploration move


# ------------------------------------------------------------------ helpers
def now():
    return datetime.now().isoformat(timespec="seconds")


def atomic_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def fmt(x, nd=4):
    return round(x, nd) if isinstance(x, float) else x


class CsvLog:
    """Buffers rows and appends them at checkpoints, so CSVs always match the
    saved models. If the columns changed since the file was created, the old
    file is renamed instead of being mixed with the new format."""

    def __init__(self, path, header):
        self.path, self.header, self.rows = path, header, []
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            with open(self.path, newline="") as f:
                old = next(csv.reader(f), [])
            if old != header:
                stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                os.replace(self.path, self.path.replace(".csv", f".old-format-{stamp}.csv"))

    def add(self, row):
        self.rows.append(row)

    def flush(self):
        if not self.rows:
            return
        new = not os.path.exists(self.path) or os.path.getsize(self.path) == 0
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.header, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerows(self.rows)
        self.rows.clear()


def outcome_bonus(my, opp):
    if my > opp:
        return cfg.WIN_BONUS
    if my == opp:
        return -cfg.TIE_PENALTY
    return -cfg.LOSS_PENALTY


def game_length(rng):
    extra = int(rng.geometric(1.0 - cfg.CONTINUE_PROB)) - 1 if cfg.CONTINUE_PROB < 1 else 0
    return min(cfg.MIN_ROUNDS + extra, cfg.MAX_ROUNDS)


def result_of(my, opp):
    return "win" if my > opp else "tie" if my == opp else "loss"


def elo_update(ra, rb, score_a, k):
    ea = 1.0 / (1.0 + 10 ** ((rb - ra) / 400.0))
    return ra + k * (score_a - ea), rb - k * (score_a - ea)


def side_stats(hist):
    """hist: list of (my, opp) moves from one player's point of view."""
    n = len(hist)
    after_c = [hist[i][0] for i in range(1, n) if hist[i - 1][1] == C]
    after_d = [hist[i][0] for i in range(1, n) if hist[i - 1][1] == D]
    p = lambda xs: fmt(sum(1 for x in xs if x == C) / len(xs)) if xs else ""
    cnt = {"CC": 0, "CD": 0, "DC": 0, "DD": 0}
    for m, o in hist:
        cnt[MOVE[m] + MOVE[o]] += 1
    return dict(
        coop_rate=fmt(sum(1 for m, _ in hist if m == C) / n),
        opp_coop_rate=fmt(sum(1 for _, o in hist if o == C) / n),
        first_move=MOVE[hist[0][0]],
        c_after_opp_c=p(after_c), c_after_opp_d=p(after_d),
        longest_CC_streak=longest_streak(hist, (C, C)),
        longest_DD_streak=longest_streak(hist, (D, D)),
        **cnt)


def longest_streak(hist, pair):
    best = cur = 0
    for h in hist:
        cur = cur + 1 if h == pair else 0
        best = max(best, cur)
    return best


def move_string(moves, explored):
    return "".join((MOVE_RANDOM if e else MOVE)[m] for m, e in zip(moves, explored))


# ------------------------------------------------------------------ games
def play_training_game(a, b, rng):
    n = game_length(rng)
    ha, hb = [], []
    xa, xb = [], []
    sa = sb = 0
    ra_tot = rb_tot = 0.0
    oa = encode_obs(ha, sa, sb, cfg.HISTORY_LEN)
    ob = encode_obs(hb, sb, sa, cfg.HISTORY_LEN)
    for t in range(n):
        ma, _, ea = a.act(oa, hist=ha)
        mb, _, eb = b.act(ob, hist=hb)
        xa.append(ea)
        xb.append(eb)
        pa, pb = cfg.PAYOFF[(ma, mb)], cfg.PAYOFF[(mb, ma)]
        sa += pa
        sb += pb
        ha.append((ma, mb))
        hb.append((mb, ma))
        done = t == n - 1
        ra, rb = float(pa), float(pb)
        if done:
            ra += outcome_bonus(sa, sb)
            rb += outcome_bonus(sb, sa)
        ra_tot += ra
        rb_tot += rb
        na = encode_obs(ha, sa, sb, cfg.HISTORY_LEN)
        nb = encode_obs(hb, sb, sa, cfg.HISTORY_LEN)
        a.remember(oa, ma, ra, na, done)
        b.remember(ob, mb, rb, nb, done)
        oa, ob = na, nb
    return dict(ha=ha, hb=hb, xa=xa, xb=xb, sa=sa, sb=sb, ra=ra_tot, rb=rb_tot, rounds=n)


def play_greedy(a, b, rng, n=None):
    """No exploration, no learning: shows what each agent has really learned."""
    n = n or game_length(rng)
    ha, hb, sa, sb = [], [], 0, 0
    for _ in range(n):
        ma = a.act(encode_obs(ha, sa, sb, cfg.HISTORY_LEN), greedy=True, hist=ha)[0]
        mb = b.act(encode_obs(hb, sb, sa, cfg.HISTORY_LEN), greedy=True, hist=hb)[0]
        sa += cfg.PAYOFF[(ma, mb)]
        sb += cfg.PAYOFF[(mb, ma)]
        ha.append((ma, mb))
        hb.append((mb, ma))
    return ha, hb, sa, sb


# ------------------------------------------------------------------ probes
# What does the agent do in a few standard situations? (my move, opp move)
PROBES = {
    "start": [],
    "after_CC": [(C, C)],
    "after_CD": [(C, D)],
    "after_DC": [(D, C)],
    "after_DD": [(D, D)],
    "long_coop": [(C, C)] * cfg.HISTORY_LEN,
    "betrayed_after_coop": [(C, C)] * (cfg.HISTORY_LEN - 1) + [(C, D)],
    "long_defect": [(D, D)] * cfg.HISTORY_LEN,
}
# Descriptive names for common fingerprints (start, after CC, CD, DC, DD).
# Only used to label the data; the agents never see these.
RESEMBLES = {"CCDCD": "tit-for-tat-like", "DCDCD": "suspicious-tit-for-tat-like",
             "CCDDC": "win-stay-lose-shift-like", "CCCCC": "always-cooperate",
             "DDDDD": "always-defect"}


def probe(agent):
    row, sig = {}, ""
    for name, h in PROBES.items():
        s = sum(cfg.PAYOFF[x] for x in h)
        o = sum(cfg.PAYOFF[(y, x)] for x, y in h)
        obs = encode_obs(h, s, o, cfg.HISTORY_LEN)
        if agent.is_learner:
            q = agent.q_values(obs)
            mv = MOVE[int(np.argmax(q))]
            row[f"{name}_qC"] = fmt(float(q[0]))
            row[f"{name}_qD"] = fmt(float(q[1]))
        else:
            mv = MOVE[agent.act(obs, greedy=True, hist=list(h))[0]]
        row[name] = mv
        if name in ("start", "after_CC", "after_CD", "after_DC", "after_DD"):
            sig += mv
    row["signature"] = sig
    row["resembles"] = RESEMBLES.get(sig, "other") if agent.is_learner else agent.name
    return row


# ------------------------------------------------------------------ main
GAME_HEADER = [
    "game_id", "run_id", "timestamp", "agent_1", "agent_2", "rounds", "score_1", "score_2",
    "winner", "coop_rate_1", "coop_rate_2", "CC", "CD", "DC", "DD",
    "explored_1", "explored_2", "epsilon_1", "epsilon_2", "elo_1", "elo_2", "moves_1", "moves_2"]
AGENT_GAME_HEADER = [
    "game_id", "run_id", "timestamp", "agent", "opponent", "rounds", "score", "opp_score",
    "margin", "result", "coop_rate", "opp_coop_rate", "first_move", "c_after_opp_c",
    "c_after_opp_d", "CC", "CD", "DC", "DD", "longest_CC_streak", "longest_DD_streak",
    "explored_moves", "explored_pct", "epsilon", "shaped_reward", "elo_before", "elo_after",
    "opp_elo_before", "agent_games", "agent_total_moves", "train_updates", "mean_loss", "mean_q",
    "moves", "opp_moves"]
TOURNEY_HEADER = [
    "tournament_at_game", "run_id", "timestamp", "agent", "opponent", "match", "rounds", "score",
    "opp_score", "margin", "result", "coop_rate", "opp_coop_rate", "moves", "opp_moves"]
PROBE_HEADER = ["game_id", "agent_games", "total_moves", "epsilon"] + \
    [k for p in PROBES for k in (p, f"{p}_qC", f"{p}_qD")] + ["signature", "resembles"]


def pick_device(requested):
    if requested == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    if requested == "cuda":
        raise SystemExit("CUDA requested but PyTorch/CUDA is not available.")
    return "cpu"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=0, help="stop after N games (default: run forever)")
    ap.add_argument("--data-dir", default=cfg.DATA_DIR)
    ap.add_argument("--device", choices=["auto", "cuda", "cpu"], default=cfg.DEVICE,
                    help="auto = use the GPU when available (default)")
    args = ap.parse_args()
    device = pick_device(args.device)

    root = os.path.abspath(args.data_dir)
    os.makedirs(os.path.join(root, "agents"), exist_ok=True)
    state_path = os.path.join(root, "state.json")
    state = dict(next_game_id=1, runs=0, created_at=now(), rng_state=None, elo={})
    if os.path.exists(state_path):
        with open(state_path) as f:
            state.update(json.load(f))
    run_id = state["runs"] + 1
    state["runs"] = run_id

    rng = np.random.default_rng(12345)
    if state["rng_state"]:
        rng.bit_generator.state = state["rng_state"]

    # ---- agents (each has its own folder, model, memory, logs)
    agents, agent_logs, probe_logs = {}, {}, {}
    for i in range(cfg.NUM_AGENTS):
        name = f"agent_{i:02d}"
        acfg = dict(cfg.AGENT_DEFAULTS, seed=1000 + i, reward_scale=cfg.REWARD_SCALE)
        acfg.update(cfg.AGENT_OVERRIDES.get(name, {}))
        ag = DQNAgent(name, os.path.join(root, "agents", name), acfg, cfg.HISTORY_LEN, cfg.HIDDEN_LAYERS)
        if ag.load():
            print(f"[{name}] resumed: {ag.games_played} games, {ag.total_moves} moves, eps={ag.epsilon:.3f}")
        else:
            ag.created_at = now()
            print(f"[{name}] new agent (starts playing randomly)")
        agents[name] = ag
    for j, name in enumerate(cfg.FIXED_AGENTS):
        ag = FixedAgent(name, os.path.join(root, "agents", name), 5000 + j,
                        dict(generosity=cfg.GENEROUS_TFT_FORGIVENESS))
        if ag.load():
            print(f"[{name}] fixed strategy, {ag.games_played} games so far")
        else:
            ag.created_at = now()
            print(f"[{name}] fixed strategy joined")
        agents[name] = ag
    for name, ag in agents.items():
        state["elo"].setdefault(name, 1000.0)
        agent_logs[name] = CsvLog(os.path.join(ag.dir, "games.csv"), AGENT_GAME_HEADER)
        probe_logs[name] = CsvLog(os.path.join(ag.dir, "probes.csv"), PROBE_HEADER)
    names = list(agents)
    learners = [ag for ag in agents.values() if ag.is_learner]
    elo = state["elo"]

    pop = None
    if device == "cuda":
        import torch
        from gpu_engine import GPUPopulation, VecGames
        pop = GPUPopulation(learners, device) if learners else None
        print(f"\nUsing GPU: {torch.cuda.get_device_name(0)} ({cfg.GPU_PARALLEL_GAMES} games in parallel)")
    else:
        print("\nUsing CPU (one game at a time)")

    games_log = CsvLog(os.path.join(root, "game_history.csv"), GAME_HEADER)
    tourney_log = CsvLog(os.path.join(root, "tournaments.csv"), TOURNEY_HEADER)
    runs_log = CsvLog(os.path.join(root, "runs.csv"), [
        "run_id", "event", "timestamp", "next_game_id", "games_this_run", "seconds", "num_agents"])
    all_logs = [games_log, tourney_log, runs_log, *agent_logs.values(), *probe_logs.values()]

    with open(os.path.join(root, "run_configs.jsonl"), "a") as f:
        f.write(json.dumps(dict(run_id=run_id, started=now(), first_game_id=state["next_game_id"], device=device,
                                config={k: (str(v) if k == "PAYOFF" else v) for k, v in vars(cfg).items()
                                        if k.isupper()})) + "\n")

    t0 = time.time()
    games_this_run = 0

    def run_row(event):
        runs_log.add(dict(run_id=run_id, event=event, timestamp=now(), next_game_id=state["next_game_id"],
                          games_this_run=games_this_run, seconds=round(time.time() - t0, 1),
                          num_agents=len(agents)))

    def checkpoint():
        for lg in all_logs:
            lg.flush()
        if pop is not None:
            pop.sync_to_agents()
        for ag in agents.values():
            ag.save()
        state["rng_state"] = rng.bit_generator.state
        state["last_saved"] = now()
        atomic_json(state_path, state)

    # ---------------------------------------------------------- tournaments
    def log_tournament_game(gid, ts, x, y, k, hx, sx, sy):
        hy = [(o, m) for m, o in hx]
        mx = "".join(MOVE[m] for m, _ in hx)
        my_ = "".join(MOVE[m] for m, _ in hy)
        for me, opp, h, s1, s2, m1, m2 in ((x, y, hx, sx, sy, mx, my_), (y, x, hy, sy, sx, my_, mx)):
            tourney_log.add(dict(
                tournament_at_game=gid, run_id=run_id, timestamp=ts, agent=me, opponent=opp, match=k + 1,
                rounds=len(h), score=s1, opp_score=s2, margin=s1 - s2, result=result_of(s1, s2),
                coop_rate=fmt(sum(m == C for m, _ in h) / len(h)),
                opp_coop_rate=fmt(sum(o == C for _, o in h) / len(h)), moves=m1, opp_moves=m2))

    def tournament(gid, ts):
        trng = np.random.default_rng(gid)  # separate RNG: doesn't disturb training
        pairs = [(x, y, k) for i, x in enumerate(names) for y in names[i + 1:]
                 for k in range(cfg.TOURNAMENT_GAMES_PER_PAIR)]
        lengths = [game_length(trng) for _ in pairs]
        if device == "cuda":
            vg = VecGames(pop, agents, len(pairs), cfg, device)
            vg.start(np.arange(len(pairs)), [names.index(x) for x, _, _ in pairs],
                     [names.index(y) for _, y, _ in pairs], lengths, np.zeros(len(names)))
            while vg.active.any():
                vg.step(greedy=True, learn=False, eps_by_agent=np.zeros(len(names)))
            for (x, y, k), g in zip(pairs, vg.collect(np.arange(len(pairs)))):
                log_tournament_game(gid, ts, x, y, k, g["ha"], g["sa"], g["sb"])
        else:
            for (x, y, k), n in zip(pairs, lengths):
                hx, hy, sx, sy = play_greedy(agents[x], agents[y], trng, n)
                log_tournament_game(gid, ts, x, y, k, hx, sx, sy)

    # -------------------------------------------------------- game logging
    window = dict(n=0, ties=0, coop=0.0, t=time.time())

    def record(g):
        nonlocal games_this_run, window
        gid = state["next_game_id"]
        n1, n2, sa, sb = g["n1"], g["n2"], g["sa"], g["sb"]
        a, b = agents[n1], agents[n2]
        ts = now()
        e1, e2 = elo[n1], elo[n2]
        elo[n1], elo[n2] = elo_update(e1, e2, 1.0 if sa > sb else 0.5 if sa == sb else 0.0, cfg.ELO_K)
        mv = {n1: move_string([m for m, _ in g["ha"]], g["xa"]),
              n2: move_string([m for m, _ in g["hb"]], g["xb"])}
        st = {n1: side_stats(g["ha"]), n2: side_stats(g["hb"])}
        games_log.add(dict(
            game_id=gid, run_id=run_id, timestamp=ts, agent_1=n1, agent_2=n2, rounds=g["rounds"],
            score_1=sa, score_2=sb, winner=n1 if sa > sb else n2 if sb > sa else "tie",
            coop_rate_1=st[n1]["coop_rate"], coop_rate_2=st[n2]["coop_rate"],
            CC=st[n1]["CC"], CD=st[n1]["CD"], DC=st[n1]["DC"], DD=st[n1]["DD"],
            explored_1=sum(g["xa"]), explored_2=sum(g["xb"]),
            epsilon_1=fmt(g["eps_a"]), epsilon_2=fmt(g["eps_b"]), elo_1=fmt(e1, 1), elo_2=fmt(e2, 1),
            moves_1=mv[n1], moves_2=mv[n2]))
        for me, opp, ag, my, os_, rew, expl, eps, eb, oe in (
                (n1, n2, a, sa, sb, g["ra"], g["xa"], g["eps_a"], e1, e2),
                (n2, n1, b, sb, sa, g["rb"], g["xb"], g["eps_b"], e2, e1)):
            ag.games_played += 1
            ts_ = ag.pop_game_stats()
            agent_logs[me].add(dict(
                game_id=gid, run_id=run_id, timestamp=ts, agent=me, opponent=opp, rounds=g["rounds"],
                score=my, opp_score=os_, margin=my - os_, result=result_of(my, os_), **st[me],
                explored_moves=sum(expl), explored_pct=fmt(sum(expl) / g["rounds"]),
                epsilon=fmt(eps), shaped_reward=fmt(rew), elo_before=fmt(eb, 1),
                elo_after=fmt(elo[me], 1), opp_elo_before=fmt(oe, 1),
                agent_games=ag.games_played, agent_total_moves=ag.total_moves,
                train_updates=ts_["updates"], mean_loss=fmt(ts_["mean_loss"], 5),
                mean_q=fmt(ts_["mean_q"]), moves=mv[me], opp_moves=mv[opp]))

        if cfg.PROBE_EVERY and gid % cfg.PROBE_EVERY == 0:
            for name, ag in agents.items():
                probe_logs[name].add(dict(game_id=gid, agent_games=ag.games_played,
                                          total_moves=ag.total_moves, epsilon=fmt(ag.epsilon), **probe(ag)))
        if cfg.TOURNAMENT_EVERY and gid % cfg.TOURNAMENT_EVERY == 0:
            tournament(gid, ts)

        window["n"] += 1
        window["ties"] += sa == sb
        window["coop"] += (st[n1]["coop_rate"] + st[n2]["coop_rate"]) / 2
        state["next_game_id"] = gid + 1
        games_this_run += 1

    def maybe_print_and_save(last):
        nonlocal window
        now_t = time.time()
        if now_t - window["t"] >= cfg.PRINT_SECONDS and window["n"]:
            k = window["n"]
            top = sorted(elo.items(), key=lambda kv: -kv[1])[:3]
            eps = [x.epsilon for x in learners]
            print(f"game {state['next_game_id'] - 1:>9} | {k / (now_t - window['t']):6.1f} g/s | "
                  f"ties {window['ties'] / k:4.0%} | coop {window['coop'] / k:4.0%} | "
                  f"avg eps {np.mean(eps) if eps else 0:.3f} | "
                  f"top Elo: " + ", ".join(f"{n[-2:] if n.startswith('agent_') else n}={r:.0f}" for n, r in top))
            window = dict(n=0, ties=0, coop=0.0, t=now_t)
        if now_t - last["save"] >= cfg.CHECKPOINT_SECONDS:
            checkpoint()
            last["save"] = time.time()

    stop = {"flag": False}

    def on_signal(sig, frame):
        if stop["flag"]:
            raise KeyboardInterrupt
        stop["flag"] = True
        print("\nStopping and saving... (Ctrl+C again to force)")

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    run_row("start")
    print(f"Run #{run_id} starting at game {state['next_game_id']} with {len(agents)} players. Data: {root}")
    print("Press Ctrl+C to stop (progress is saved).\n")

    last = dict(save=time.time())
    more = lambda: not stop["flag"] and (args.games == 0 or games_this_run < args.games)
    try:
        if device == "cuda":
            K = cfg.GPU_PARALLEL_GAMES
            vg = VecGames(pop, agents, K, cfg, device)
            idx = {n: i for i, n in enumerate(names)}

            def eps_by_agent():
                return np.array([agents[n].epsilon for n in names])

            def new_games(slots):
                pairs = [rng.choice(len(names), size=2, replace=False) for _ in slots]
                vg.start(slots, [p[0] for p in pairs], [p[1] for p in pairs],
                         [game_length(rng) for _ in slots], eps_by_agent())

            new_games(np.arange(K))
            while more():
                done = vg.step(greedy=False, learn=True, eps_by_agent=eps_by_agent())
                if len(done):
                    pop.flush_stats()
                    for g in vg.collect(done):
                        if more():
                            record(g)
                    new_games(done)
                    maybe_print_and_save(last)
        else:
            while more():
                n1, n2 = (str(x) for x in rng.choice(names, size=2, replace=False))
                a, b = agents[n1], agents[n2]
                eps_a, eps_b = a.epsilon, b.epsilon
                g = play_training_game(a, b, rng)
                record(dict(g, n1=n1, n2=n2, eps_a=eps_a, eps_b=eps_b))
                maybe_print_and_save(last)
    except KeyboardInterrupt:
        print("\nForced stop.")
    finally:
        run_row("end")
        checkpoint()
        print(f"Saved. {games_this_run} games this run; next game id {state['next_game_id']}.")


if __name__ == "__main__":
    main()
