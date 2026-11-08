#!/usr/bin/env python3
"""Builds the summary charts (PNG) and prints the headline numbers.

    .venv/bin/python analysis/make_figures.py          (from the project folder)

Only reads data/ (safe while training is running). Charts go to analysis/figures/.
Analysis is limited to games already logged when the script starts.
"""
import os
import sys

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- style
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
LEARNER = "#2a78d6"
FIXED = {"Tit for Tat": "#eb6834", "Generous Tit for Tat": "#1baf7a", "Grim Trigger": "#e34948"}
OUTCOME = {"Both cooperate": "#1baf7a", "One betrays the other": "#eda100", "Both defect": "#e34948"}
plt.rcParams.update({
    "font.family": "Noto Sans", "font.size": 13, "text.color": INK,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.labelsize": 13,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "lines.linewidth": 2.5, "legend.frameon": False, "legend.fontsize": 12,
})
FIGSIZE = (12, 7.5)   # 1800 x 1125 px at 150 dpi
DPI = 150


def frame(fig, title, subtitle, note=""):
    fig.text(0.04, 0.955, title, fontsize=21, fontweight="bold", va="top")
    fig.text(0.04, 0.89, subtitle, fontsize=14, color=INK2, va="top")
    fig.text(0.04, 0.015, SRC + ("\n" + note if note else ""), fontsize=10, color=MUTED, va="bottom",
             linespacing=1.45)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print("  wrote", os.path.relpath(path, ROOT))


thousands = FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M" if v >= 1e6 else f"{v / 1e3:.0f}k" if v >= 1e3 else f"{v:.0f}")

# ---------------------------------------------------------------- data
con = duckdb.connect()
print("Loading data…")
con.execute(f"""CREATE TABLE ag AS SELECT * EXCLUDE (moves, opp_moves)
                FROM read_csv('{DATA}/agents/*/games.csv', union_by_name=true, ignore_errors=true)""")
con.execute(f"""CREATE TABLE pr AS SELECT *, regexp_extract(filename, '([^/]+)/probes[.]csv$', 1) AS agent
                FROM read_csv('{DATA}/agents/*/probes.csv', union_by_name=true, filename=true)""")
# Tournament scores are recomputed from the move strings (scores logged by the GPU tournament
# code before its fix were wrong; the move strings were always correct).
con.execute(f"""CREATE TABLE tr AS
    WITH t AS (SELECT tournament_at_game, agent, opponent, upper(moves) m, upper(opp_moves) o
               FROM read_csv('{DATA}/tournaments.csv'))
    SELECT tournament_at_game, agent, opponent, len(m) rounds,
           list_sum(list_transform(range(1, len(m) + 1), lambda i:
               CASE substr(m, i, 1) || substr(o, i, 1) WHEN 'CC' THEN 3 WHEN 'CD' THEN 0 WHEN 'DC' THEN 5 ELSE 1 END)) score,
           list_sum(list_transform(range(1, len(m) + 1), lambda i:
               CASE substr(o, i, 1) || substr(m, i, 1) WHEN 'CC' THEN 3 WHEN 'CD' THEN 0 WHEN 'DC' THEN 5 ELSE 1 END)) opp_score,
           len(regexp_extract_all(m, 'C')) * 1.0 / len(m) coop_rate
    FROM t""")

q = lambda sql: con.sql(sql).df()
tot = q("SELECT count(*) / 2 games, sum(rounds) / 2 rounds, max(game_id) last FROM ag").iloc[0]
GAMES, ROUNDS, LAST = int(tot.games), int(tot.rounds), int(tot["last"])
LATE = max(1, LAST - 500_000)  # "after learning": the most recent 500k games
SRC = (f"Data: {GAMES / 1e6:.2f} million simulated games ({ROUNDS / 1e6:.0f} million rounds) between 10 self-taught "
       f"neural networks and 3 fixed strategies.\nPoints per round: both cooperate 3 each, both defect 1 each, "
       f"one betrays the other 5 for the betrayer and 0 for the betrayed.")

# ================================================================ 1. learning curve
print("Figure 1: learning curve")
d = q("""SELECT (game_id // 2000) * 2000 + 1000 g,
            sum(CC) * 100.0 / sum(rounds) cc, (sum(CD) + sum(DC)) * 100.0 / sum(rounds) bet,
            sum(DD) * 100.0 / sum(rounds) dd, avg(explored_pct) * 100 rnd,
            sum(score) / sum(rounds) ppr
         FROM ag WHERE agent LIKE 'agent_%' AND game_id < 150000 GROUP BY 1 ORDER BY 1""")
fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_axes([0.07, 0.40, 0.70, 0.44])
ax2 = fig.add_axes([0.07, 0.205, 0.70, 0.10], sharex=ax)
ax.stackplot(d.g, d.cc, d.bet, d.dd, colors=list(OUTCOME.values()), edgecolor=SURFACE, linewidth=0.6)
ax.set_ylim(0, 100)
ax.set_xlim(0, d.g.max())
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
ax.set_ylabel("Share of rounds")
plt.setp(ax.get_xticklabels(), visible=False)
ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in reversed(list(OUTCOME.values()))],
          list(reversed(list(OUTCOME))), loc="upper left", bbox_to_anchor=(1.02, 1.0))
ax2.plot(d.g, d.rnd, color=INK2, linewidth=2)
ax2.set_ylim(0, 100)
ax2.yaxis.set_major_formatter(PercentFormatter(decimals=0))
ax2.set_yticks([0, 50, 100])
ax2.set_title("Share of moves that were random (exploration)", loc="left", fontsize=12, color=INK2, pad=4)
ax2.xaxis.set_major_formatter(thousands)
ax2.set_xlabel("Games played by the population")
frame(fig, "Ten networks that started out playing at random learned to cooperate",
      "What happened in each round of the learners' games over the first 150,000 games",
      "Random moves never drop below 3%, so some betrayals and defections always remain.")
save(fig, "1_learning_to_cooperate.png")

# ================================================================ 2. winning vs scoring
print("Figure 2: winning vs scoring")
d = q(f"""SELECT agent, sum(score) / sum(rounds) ppr, avg((result = 'win')::INT) * 100 win,
             avg((result = 'tie')::INT) * 100 tie
          FROM ag WHERE game_id > {LATE} GROUP BY agent ORDER BY ppr""")
label = lambda a: a.replace("agent_", "Learner ") if a.startswith("agent_") else a
colors = [FIXED.get(a, LEARNER) for a in d.agent]
fig = plt.figure(figsize=FIGSIZE)
axl = fig.add_axes([0.20, 0.20, 0.37, 0.57])
axr = fig.add_axes([0.62, 0.20, 0.33, 0.57], sharey=axl)
y = range(len(d))
axl.hlines(y, 2.0, d.ppr, color=GRID, linewidth=2)
axl.scatter(d.ppr, y, s=140, color=colors, zorder=3, edgecolor=SURFACE, linewidth=2)
for yi, v in zip(y, d.ppr):
    axl.text(v - 0.025, yi + 0.33, f"{v:.2f}", va="center", ha="center", fontsize=11, color=INK)
axl.set_xlim(2.0, 3.12)
axl.set_yticks(list(y), [label(a) for a in d.agent])
for t, a in zip(axl.get_yticklabels(), d.agent):
    if a in FIXED:
        t.set_fontweight("bold")
axl.set_title("Points per round", loc="left", fontsize=15, fontweight="bold", pad=10)
axl.grid(axis="y", visible=False)
axr.barh(list(y), d.win, color=colors, height=0.62)
for yi, v in zip(y, d.win):
    axr.text(v + 1, yi, f"{v:.0f}%", va="center", fontsize=11.5, color=INK)
axr.set_xlim(0, 55)
axr.xaxis.set_major_formatter(PercentFormatter(decimals=0))
axr.set_title("Games won (outscored the opponent)", loc="left", fontsize=15, fontweight="bold", pad=10)
axr.grid(axis="y", visible=False)
plt.setp(axr.get_yticklabels(), visible=False)
axr.tick_params(axis="y", length=0)
frame(fig, "Tit for Tat never wins a game, yet it finishes with the most points",
      f"Results over the most recent 500,000 games, after learning had settled. "
      f"Blue = self-taught networks.")
save(fig, "2_winning_is_not_scoring.png")

# ================================================================ 3. matchups
print("Figure 3: matchups")
d = q(f"""SELECT CASE WHEN opponent LIKE 'agent_%' THEN 'Another learner' ELSE opponent END opp,
             sum(score) / sum(rounds) learner, sum(opp_score) / sum(rounds) other, count(*) n
          FROM ag WHERE agent LIKE 'agent_%' AND game_id > {LATE} GROUP BY 1 ORDER BY learner""")
fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_axes([0.22, 0.21, 0.72, 0.59])
y = list(range(len(d)))
h = 0.36
ax.barh([i + h / 2 for i in y], d.learner, height=h, color=LEARNER, label="Learner")
ax.barh([i - h / 2 for i in y], d.other, height=h, color=[FIXED.get(o, "#9cc2ef") for o in d.opp], label="Opponent")
for i, (a, b) in enumerate(zip(d.learner, d.other)):
    ax.text(a + 0.03, i + h / 2, f"{a:.2f}", va="center", fontsize=11.5)
    ax.text(b + 0.03, i - h / 2, f"{b:.2f}", va="center", fontsize=11.5)
ax.set_yticks(y, [f"vs {o}" for o in d.opp], fontsize=13)
ax.set_xlim(0, 3.4)
ax.set_xlabel("Points per round")
ax.grid(axis="y", visible=False)
g = d[d.opp == "Grim Trigger"].iloc[0]
frame(fig, "Grim Trigger turns small mistakes into permanent conflict",
      f"Points per round for the learners (top, blue bar) and their opponent (bottom bar), last 500,000 games",
      "Learners still make 3% random moves. One random defection makes Grim Trigger defect for the rest of the game.")
save(fig, "3_grim_trigger_punishes_mistakes.png")

# ================================================================ 4. forgiveness
print("Figure 4: forgiveness")
d = q("""SELECT (game_id // 20000) * 20000 + 10000 g, avg(c_after_opp_d) * 100 forgive, avg(c_after_opp_c) * 100 recip
         FROM ag WHERE agent LIKE 'agent_%' AND explored_pct < 0.1 GROUP BY 1 ORDER BY 1""")
d["forgive_s"] = d.forgive.rolling(5, min_periods=1, center=True).mean()
fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_axes([0.08, 0.21, 0.68, 0.60])
ax.plot(d.g, d.forgive_s, color=LEARNER, label="Learners")
for val, name in ((100 / 3, "Generous Tit for Tat"), (0, "Tit for Tat")):
    ax.axhline(val, color=FIXED[name], linestyle=(0, (4, 3)), linewidth=1.8, label=f"{name} ({val:.0f}%)")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
ax.set_ylim(-3, 100)
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
ax.xaxis.set_major_formatter(thousands)
ax.set_xlim(0, d.g.max())
ax.set_ylabel("Cooperates right after the opponent defected")
ax.set_xlabel("Games played by the population")
frame(fig, "The networks learned to forgive twice as often as Generous Tit for Tat",
      "How often a learner cooperated in the round right after its opponent defected",
      "Learner games with 10% or more random moves are left out.")
save(fig, "4_learned_forgiveness.png")

# ================================================================ 5. strategy fingerprints
print("Figure 5: strategies")
names = {"CCDCD": "Tit for Tat", "CCDCC": "Tit for Tat that also ends feuds",
         "CCCCD": "Forgives a betrayal, but not a feud", "CCCCC": "Cooperates in every test"}
d = q(f"""SELECT (game_id // 100000) * 100000 + 50000 g, signature, count(*) n
          FROM pr WHERE agent LIKE 'agent_%' GROUP BY ALL""")
d["kind"] = d.signature.map(lambda s: names.get(s, "Other"))
pv = d.pivot_table(index="g", columns="kind", values="n", aggfunc="sum", fill_value=0)
pv = pv.div(pv.sum(axis=1), axis=0) * 100
order = ["Tit for Tat", "Tit for Tat that also ends feuds", "Forgives a betrayal, but not a feud",
         "Cooperates in every test", "Other"]
cols = {"Tit for Tat": "#eb6834", "Tit for Tat that also ends feuds": "#eda100",
        "Forgives a betrayal, but not a feud": "#2a78d6", "Cooperates in every test": "#1baf7a", "Other": "#c9c8c2"}
pv = pv[[o for o in order if o in pv]]
fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_axes([0.08, 0.21, 0.60, 0.60])
ax.stackplot(pv.index, *[pv[c] for c in pv], colors=[cols[c] for c in pv], edgecolor=SURFACE, linewidth=0.8)
ax.set_ylim(0, 100)
ax.set_xlim(pv.index.min(), pv.index.max())
ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))
ax.xaxis.set_major_formatter(thousands)
ax.set_ylabel("Share of learner snapshots")
ax.set_xlabel("Games played by the population")
ax.legend([plt.Rectangle((0, 0), 1, 1, color=cols[c]) for c in reversed(list(pv))], list(reversed(list(pv))),
          loc="upper left", bbox_to_anchor=(1.02, 1.0))
frame(fig, "The networks settled on several cooperative strategies, not just one",
      "Strategy of each learner, read from its choices (without randomness) in five test situations",
      "Tests: first move, and the move after: both cooperated / it was betrayed / it betrayed / both defected.")
save(fig, "5_strategies_that_emerged.png")

# ================================================================ 6. points and cooperation over time
print("Figure 6: points and cooperation, game 0 to latest")
B = 20_000
d = q(f"""SELECT agent, least((game_id // {B}) * {B} + {B // 2}, {LAST}) g,
             sum(score) / sum(rounds) ppr, avg(coop_rate) * 100 coop
          FROM ag GROUP BY ALL HAVING count(*) >= 500 ORDER BY agent, g""")  # skip a nearly empty last block
fig = plt.figure(figsize=FIGSIZE)
ax1 = fig.add_axes([0.09, 0.53, 0.64, 0.29])
ax2 = fig.add_axes([0.09, 0.20, 0.64, 0.29], sharex=ax1)
learner_names = sorted(a for a in d.agent.unique() if a.startswith("agent_"))
for ax, col in ((ax1, "ppr"), (ax2, "coop")):
    for a in learner_names:
        g = d[d.agent == a]
        ax.plot(g.g, g[col], color=LEARNER, linewidth=1.3, alpha=0.5, zorder=2)
    for a, c in FIXED.items():
        g = d[d.agent == a]
        ax.plot(g.g, g[col], color=c, linewidth=2.6, zorder=3)
ax1.set_ylabel("Points per round")
ax1.set_ylim(2.0, 3.05)
plt.setp(ax1.get_xticklabels(), visible=False)
ax2.set_ylabel("Cooperation rate")
ax2.set_ylim(40, 101)
ax2.yaxis.set_major_formatter(PercentFormatter(decimals=0))
ax2.set_xlim(0, LAST)
ax2.xaxis.set_major_formatter(thousands)
ax2.set_xlabel("Game number")
handles = [plt.Line2D([], [], color=LEARNER, linewidth=1.6, alpha=0.7)] + \
          [plt.Line2D([], [], color=c, linewidth=2.6) for c in FIXED.values()]
ax1.legend(handles, [f"Learners ({len(learner_names)}, one line each)", *FIXED],
           loc="upper left", bbox_to_anchor=(1.02, 1.0))
frame(fig, "Points and cooperation of every player, from the first game to the latest",
      f"Average per player, in blocks of {B:,} games",
      "All games included, also the early ones where the learners played mostly at random.")
save(fig, "6_points_and_cooperation_over_time.png")

# ================================================================ headline numbers
print("\nHeadline numbers")
early = q("""SELECT avg(coop_rate) * 100 coop, sum(score) / sum(rounds) ppr FROM ag
             WHERE agent LIKE 'agent_%' AND game_id <= 5000""").iloc[0]
late = q(f"""SELECT avg(coop_rate) * 100 coop, sum(score) / sum(rounds) ppr, avg((result='tie')::INT) * 100 tie,
               avg(c_after_opp_d) * 100 forgive FROM ag WHERE agent LIKE 'agent_%' AND game_id > {LATE}""").iloc[0]
t = q("""SELECT count(DISTINCT tournament_at_game) n, avg(coop_rate) * 100 coop, sum(score) / sum(rounds) ppr
         FROM tr WHERE tournament_at_game > (SELECT max(tournament_at_game) FROM tr) - 500000""").iloc[0]
runs = q(f"SELECT run_id, count(*) / 2 g, epoch(max(timestamp) - min(timestamp)) s FROM ag GROUP BY 1 ORDER BY 1")
print(f"  games {GAMES:,}  rounds {ROUNDS:,}")
print(f"  learners first 5k games: coop {early.coop:.1f}%  points/round {early.ppr:.2f}")
print(f"  learners last 500k:      coop {late.coop:.1f}%  points/round {late.ppr:.3f}  ties {late.tie:.1f}%  "
      f"forgive {late.forgive:.1f}%")
print(f"  tournaments (no random moves), last 500k games: {int(t.n)} tournaments, coop {t.coop:.1f}%, "
      f"points/round {t.ppr:.3f}")
for r in runs.itertuples():
    print(f"  run {r.run_id}: {int(r.g):,} games in {r.s / 60:.0f} min = {r.g / max(r.s, 1):.0f} games/s")
