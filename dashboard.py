"""Interactive dashboard for the Prisoner's Dilemma population.

    .venv/bin/streamlit run dashboard.py
(optionally: PD_DATA_DIR=/path/to/data .venv/bin/streamlit run dashboard.py)
It reads the CSVs (read-only), so it can run while arena.py is training.
"""
import glob
import os
from datetime import datetime

import duckdb
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config as cfg

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.environ.get("PD_DATA_DIR", os.path.join(HERE, "data")))

st.set_page_config(page_title="Prisoner's Dilemma Lab", page_icon="🎲", layout="wide")

AGENT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
                "#008300", "#6250d6", "#e34948", "#0b8fa3", "#8c5a2b"]
RESULT_COLORS = {"win": "#1baf7a", "tie": "#8a8984", "loss": "#e34948"}
MOVE_COLORS = {"C (chosen)": "#1baf7a", "C (random)": "#a6dcc6", "D (chosen)": "#e34948", "D (random)": "#f2b0ae"}


FIXED_COLORS = {"Tit for Tat": "#3d3d3a", "Generous Tit for Tat": "#9a9992", "Grim Trigger": "#b0306a"}


def is_fixed(agent):
    return not agent.startswith("agent_")


SHORT = {"Tit for Tat": "TFT", "Generous Tit for Tat": "GTFT", "Grim Trigger": "Grim"}


def short_name(agent):
    return SHORT.get(agent, agent[-2:] if agent.startswith("agent_") else agent)


def color_map(agents):
    learners = sorted(a for a in agents if not is_fixed(a))
    cmap = {a: AGENT_COLORS[i % len(AGENT_COLORS)] for i, a in enumerate(learners)}
    for a in agents:
        if is_fixed(a):
            cmap[a] = FIXED_COLORS.get(a, "#6b6a65")
    return cmap


def sql_path(*parts):
    return os.path.join(DATA, *parts).replace("'", "''")


# ------------------------------------------------------------------ loading
def data_stamp():
    """Changes every time the trainer saves (state.json is written last at each checkpoint)."""
    try:
        return os.path.getmtime(os.path.join(DATA, "state.json"))
    except OSError:
        return 0.0


@st.cache_resource(show_spinner="Loading new data…", max_entries=1)
def load(stamp):
    con = duckdb.connect()
    tables = {}
    sources = {
        "agent_games": (sql_path("agents", "*", "games.csv"), "* EXCLUDE (moves, opp_moves)"),
        "probes": (sql_path("agents", "*", "probes.csv"), "*, regexp_extract(filename, '([^/]+)/probes[.]csv$', 1) AS agent"),
        "tournaments": (sql_path("tournaments.csv"), "* EXCLUDE (moves, opp_moves)"),
    }
    for name, (pattern, sel) in sources.items():
        if glob.glob(pattern.replace("''", "'")):
            fn = ", filename=true" if name == "probes" else ""
            con.execute(f"CREATE TABLE {name} AS SELECT {sel} FROM read_csv('{pattern}', union_by_name=true, "
                        f"ignore_errors=true, header=true{fn})")
            if name == "probes":
                con.execute("ALTER TABLE probes DROP filename")
            tables[name] = True
    hist = sql_path("game_history.csv")
    if os.path.exists(hist.replace("''", "'")):
        # full history incl. move strings: scanned on demand, not held in memory
        con.execute(f"CREATE VIEW game_history AS SELECT * FROM read_csv('{hist}', ignore_errors=true, header=true)")
        tables["game_history"] = True
    return con, tables


STAMP = data_stamp()
con, TABLES = load(STAMP)


def q(sql, params=None):
    return con.cursor().execute(sql, params or []).df()


st.title("🎲 Prisoner's Dilemma Lab")

# ------------------------------------------------------------------ auto-update
with st.sidebar:
    auto = st.toggle("Auto-update", value=True, key="auto",
                     help="Reload the dashboard by itself whenever the trainer saves new games")
    every = st.select_slider("Check for new data every", options=[10, 30, 60, 300], value=30, key="every",
                             format_func=lambda s: f"{s} s" if s < 60 else f"{s // 60} min", disabled=not auto)
    if st.button("🔄 Reload now", width="stretch"):
        st.cache_resource.clear()
        st.rerun()
    if STAMP:
        st.caption(f"Data last saved at {datetime.fromtimestamp(STAMP):%H:%M:%S}")

if auto:
    @st.fragment(run_every=every)
    def watch_for_new_data():
        if data_stamp() != STAMP:
            st.rerun(scope="app")   # filters, selections and the open tab are kept

    watch_for_new_data()

if "agent_games" not in TABLES:
    st.info(f"No data yet in `{DATA}`. Start training with `./train.sh`; "
            f"this page will update once the first games are saved (every {cfg.CHECKPOINT_SECONDS} s).")
    st.stop()

# ------------------------------------------------------------------ sidebar filters
info = q("SELECT min(game_id) g0, max(game_id) g1, count(DISTINCT game_id) n FROM agent_games").iloc[0]
ALL_AGENTS = q("SELECT DISTINCT agent FROM agent_games ORDER BY agent")["agent"].tolist()
CMAP = color_map(ALL_AGENTS)

with st.sidebar:
    st.caption(f"{int(info.n):,} games loaded (game {int(info.g0):,} → {int(info.g1):,}). "
               f"The trainer saves every {cfg.CHECKPOINT_SECONDS} s.")
    st.header("Filters")
    g0, g1 = int(info.g0), int(info.g1)
    period = st.radio("Games to include", ["All", "Last 10,000", "Last 50,000", "Custom range"], index=0)
    if period == "All":
        lo, hi = g0, g1
    elif period == "Custom range":
        lo, hi = st.slider("Game range", g0, max(g1, g0 + 1), (g0, g1))
    else:
        n = 10_000 if "10" in period else 50_000
        lo, hi = max(g0, g1 - n + 1), g1
    agents_sel = st.multiselect("Agents", ALL_AGENTS, default=ALL_AGENTS)
    max_explore = st.slider("Max % random moves in a game", 0, 100, 100,
                            help="Agents start out playing randomly. Lower this to look only at games "
                                 "where they mostly played their learned strategy.")
    st.caption("Filters apply to every tab.")

if not agents_sel:
    st.warning("Select at least one agent.")
    st.stop()

ph = ",".join("?" * len(agents_sel))
WHERE = f"game_id BETWEEN ? AND ? AND agent IN ({ph}) AND explored_pct * 100 <= ?"
PARAMS = [lo, hi, *agents_sel, max_explore]


def bucket_size():
    return max(1, (hi - lo + 1) // 150)


PCT = lambda cond: f"avg(CASE WHEN {cond} THEN 100.0 ELSE 0 END)"

tabs = st.tabs(["🏆 Leaderboard", "📈 Over time", "🤝 Cooperation vs winning", "⚔️ Matchups",
                "🧠 Strategies", "🏟️ Tournaments", "🔍 Game viewer", "🧪 Custom SQL"])

# ================================================================== leaderboard
with tabs[0]:
    k = q(f"""SELECT count(*) / 2.0 games, sum(rounds) / 2 rounds, {PCT("result='tie'")} tie,
                     avg(coop_rate) * 100 coop, avg(explored_pct) * 100 expl, sum(score)/sum(rounds) ppr
              FROM agent_games WHERE {WHERE}""", PARAMS).iloc[0]
    c = st.columns(5)
    c[0].metric("Games", f"{int(k.games or 0):,}")
    c[1].metric("Rounds played", f"{int(k.rounds or 0):,}")
    c[2].metric("Games ending in a tie", f"{(k.tie or 0):.1f}%")
    c[3].metric("Moves that cooperate", f"{(k.coop or 0):.1f}%")
    c[4].metric("Moves that were random", f"{(k.expl or 0):.1f}%")

    lb = q(f"""SELECT agent, count(*) games,
                 {PCT("result='win'")} win_pct, {PCT("result='tie'")} tie_pct, {PCT("result='loss'")} loss_pct,
                 avg(score) avg_points, sum(score) / sum(rounds) points_per_round, avg(margin) avg_margin,
                 avg(coop_rate) * 100 coop_pct,
                 avg(c_after_opp_c) * 100 coop_after_opp_coop_pct,
                 avg(c_after_opp_d) * 100 coop_after_opp_defect_pct
               FROM agent_games WHERE {WHERE} GROUP BY agent""", PARAMS)
    cur = q("""SELECT agent, arg_max(elo_after, game_id) elo, arg_max(epsilon, game_id) epsilon,
                      max(agent_games) total_games FROM agent_games GROUP BY agent""")
    lb = lb.merge(cur, on="agent", how="left")
    if "probes" in TABLES:
        fp = q("SELECT agent, arg_max(signature, game_id) fingerprint, arg_max(resembles, game_id) resembles "
               "FROM probes GROUP BY agent")
        lb = lb.merge(fp, on="agent", how="left")
    lb.insert(1, "type", lb["agent"].map(lambda a: "fixed strategy" if is_fixed(a) else "learner"))
    lb = lb.sort_values("points_per_round", ascending=False)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Win / tie / loss per agent")
        m = lb.melt(id_vars="agent", value_vars=["win_pct", "tie_pct", "loss_pct"], var_name="result", value_name="pct")
        m["result"] = m["result"].str.replace("_pct", "")
        fig = px.bar(m, y="agent", x="pct", color="result", orientation="h", color_discrete_map=RESULT_COLORS,
                     category_orders={"agent": lb["agent"].tolist(), "result": ["win", "tie", "loss"]},
                     labels={"pct": "% of games", "agent": ""}, hover_data={"pct": ":.1f"})
        fig.update_layout(barmode="stack", height=420, legend_title="", margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Points vs cooperation")
        fig = px.scatter(lb, x="coop_pct", y="points_per_round", color="agent", color_discrete_map=CMAP,
                         size="games", text=lb["agent"].map(short_name), hover_data={"win_pct": ":.1f", "elo": ":.0f"},
                         labels={"coop_pct": "Cooperation rate (%)", "points_per_round": "Points per round"})
        fig.update_traces(textposition="top center")
        fig.update_layout(height=420, showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, width="stretch")

    st.subheader("Full table")
    st.dataframe(
        lb, hide_index=True, width="stretch",
        column_config={
            "win_pct": st.column_config.ProgressColumn("Win %", format="%.1f", min_value=0, max_value=100),
            "tie_pct": st.column_config.NumberColumn("Tie %", format="%.1f"),
            "loss_pct": st.column_config.NumberColumn("Loss %", format="%.1f"),
            "avg_points": st.column_config.NumberColumn("Avg points/game", format="%.1f"),
            "points_per_round": st.column_config.NumberColumn("Points/round", format="%.2f"),
            "avg_margin": st.column_config.NumberColumn("Avg margin", format="%.1f"),
            "coop_pct": st.column_config.NumberColumn("Cooperates %", format="%.1f"),
            "coop_after_opp_coop_pct": st.column_config.NumberColumn("Coop after opp coop %", format="%.1f",
                                                                     help="Reciprocity"),
            "coop_after_opp_defect_pct": st.column_config.NumberColumn("Coop after opp defect %", format="%.1f",
                                                                       help="Forgiveness (low = retaliates)"),
            "elo": st.column_config.NumberColumn("Elo now", format="%.0f"),
            "epsilon": st.column_config.NumberColumn("Random-move chance now", format="%.3f"),
            "total_games": st.column_config.NumberColumn("Lifetime games"),
            "fingerprint": st.column_config.TextColumn("Fingerprint", help="Move at start / after CC / CD / DC / DD"),
        })

# ================================================================== over time
METRICS = {
    "Win rate (%)": PCT("result='win'"),
    "Tie rate (%)": PCT("result='tie'"),
    "Loss rate (%)": PCT("result='loss'"),
    "Cooperation rate (%)": "avg(coop_rate) * 100",
    "Points per round": "sum(score) / sum(rounds)",
    "Average score margin": "avg(margin)",
    "Elo rating": "avg(elo_after)",
    "Random moves (%)": "avg(explored_pct) * 100",
    "Cooperates after opponent defects (%)": "avg(c_after_opp_d) * 100",
    "Training loss": "avg(mean_loss)",
    "Predicted value (mean Q)": "avg(mean_q)",
}
with tabs[1]:
    c1, c2, c3 = st.columns([2, 1.2, 1.2])
    metric = c1.selectbox("Metric", list(METRICS))
    mode = c2.radio("Show", ["Each agent", "Population average"], horizontal=True)
    smooth = c3.select_slider("Smoothing", options=["Off", "Light", "Strong"], value="Light",
                              help="Rolling average over neighbouring points, to see the trend instead of noise")
    b = max(bucket_size(), 50)
    by = "agent," if mode == "Each agent" else ""
    df = q(f"""SELECT {by} (game_id // {b}) * {b} AS game, {METRICS[metric]} AS value, count(*) n
               FROM agent_games WHERE {WHERE} GROUP BY ALL ORDER BY game""", PARAMS)
    win_n = {"Off": 1, "Light": 5, "Strong": 15}[smooth]
    if win_n > 1 and len(df):
        if by:
            df["value"] = df.groupby("agent")["value"].transform(lambda v: v.rolling(win_n, min_periods=1).mean())
        else:
            df["value"] = df["value"].rolling(win_n, min_periods=1).mean()
    if mode == "Each agent":
        fig = px.line(df, x="game", y="value", color="agent", color_discrete_map=CMAP,
                      category_orders={"agent": ALL_AGENTS}, labels={"value": metric, "game": "Game number"})
    else:
        fig = px.line(df, x="game", y="value", labels={"value": metric, "game": "Game number"})
    fig.update_layout(height=480, hovermode="x unified", legend_title="")
    st.plotly_chart(fig, width="stretch")
    per_agent = b * 2 / max(len(ALL_AGENTS), 1) if by else b
    st.caption(f"Each point covers {b:,} games (≈{per_agent * win_n:,.0f} games per line after smoothing). "
               "Click legend items to hide agents; double-click one to isolate it. Drag to zoom, double-click to reset.")

    st.subheader("What happens in each round, across the population")
    df = q(f"""SELECT (game_id // {b}) * {b} AS game,
                 sum(CC) * 100.0 / sum(rounds) "Both cooperate",
                 (sum(CD) + sum(DC)) * 100.0 / sum(rounds) "One betrays the other",
                 sum(DD) * 100.0 / sum(rounds) "Both defect"
               FROM agent_games WHERE {WHERE} GROUP BY ALL ORDER BY game""", PARAMS)
    m = df.melt(id_vars="game", var_name="outcome", value_name="pct")
    fig = px.area(m, x="game", y="pct", color="outcome",
                  color_discrete_map={"Both cooperate": "#1baf7a", "One betrays the other": "#eda100",
                                      "Both defect": "#e34948"},
                  labels={"pct": "% of rounds", "game": "Game number"})
    fig.update_layout(height=360, hovermode="x unified", legend_title="", yaxis_range=[0, 100])
    st.plotly_chart(fig, width="stretch")

# ================================================================== cooperation vs winning
with tabs[2]:
    st.markdown("How does **how much an agent cooperates** relate to **whether it wins**?")
    c1, c2 = st.columns(2)
    width = c1.select_slider("Bucket size", options=[5, 10, 20, 25], value=10, format_func=lambda v: f"{v}%")
    split = c2.radio("Group by", ["All selected agents together", "Each agent"], horizontal=True)
    by = "agent," if split == "Each agent" else ""
    df = q(f"""SELECT {by} least(floor(coop_rate * 100 / {width}) * {width}, {100 - width})::INT AS bucket,
                 {PCT("result='win'")} win, {PCT("result='tie'")} tie, {PCT("result='loss'")} loss,
                 count(*) games, avg(score) avg_points
               FROM agent_games WHERE {WHERE} GROUP BY ALL ORDER BY bucket""", PARAMS)
    df["Cooperation rate"] = df["bucket"].map(lambda v: f"{v}–{v + width}%")
    m = df.melt(id_vars=[c for c in ("agent", "Cooperation rate", "games", "avg_points") if c in df],
                value_vars=["win", "tie", "loss"], var_name="result", value_name="pct")
    fig = px.bar(m, x="Cooperation rate", y="pct", color="result", color_discrete_map=RESULT_COLORS,
                 facet_col="agent" if by else None, facet_col_wrap=5,
                 hover_data={"games": ":,", "avg_points": ":.1f", "pct": ":.1f"},
                 labels={"pct": "% of games"}, category_orders={"result": ["win", "tie", "loss"]})
    fig.update_layout(barmode="stack", height=420 if not by else 650, legend_title="")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Build your own condition")
    c1, c2, c3 = st.columns(3)
    my_rng = c1.slider("My cooperation rate (%)", 0, 100, (0, 30))
    opp_rng = c2.slider("Opponent's cooperation rate (%)", 0, 100, (0, 100))
    rmax = int(q("SELECT max(rounds) FROM agent_games").iloc[0, 0])
    r_rng = c3.slider("Game length (rounds)", 1, max(rmax, 2), (1, max(rmax, 2)))
    cond = (f"{WHERE} AND coop_rate * 100 BETWEEN ? AND ? AND opp_coop_rate * 100 BETWEEN ? AND ? "
            f"AND rounds BETWEEN ? AND ?")
    cp = PARAMS + [*my_rng, *opp_rng, *r_rng]
    tot = q(f"""SELECT count(*) games, {PCT("result='win'")} win, {PCT("result='tie'")} tie,
                {PCT("result='loss'")} loss, avg(score) pts, avg(margin) margin FROM agent_games WHERE {cond}""",
            cp).iloc[0]
    c = st.columns(6)
    c[0].metric("Matching games", f"{int(tot.games):,}")
    c[1].metric("Win %", f"{(tot.win or 0):.1f}")
    c[2].metric("Tie %", f"{(tot.tie or 0):.1f}")
    c[3].metric("Loss %", f"{(tot.loss or 0):.1f}")
    c[4].metric("Avg points", f"{(tot.pts or 0):.1f}")
    c[5].metric("Avg margin", f"{(tot.margin or 0):+.1f}")
    per = q(f"""SELECT agent, count(*) games, {PCT("result='win'")} win_pct, {PCT("result='tie'")} tie_pct,
                {PCT("result='loss'")} loss_pct, avg(score) avg_points, avg(margin) avg_margin
                FROM agent_games WHERE {cond} GROUP BY agent ORDER BY win_pct DESC""", cp)
    st.dataframe(per.round(2), hide_index=True, width="stretch")

    st.subheader("Win rate by my cooperation × opponent's cooperation")
    df = q(f"""SELECT least(floor(coop_rate * 10) * 10, 90)::INT me, least(floor(opp_coop_rate * 10) * 10, 90)::INT opp,
                 {PCT("result='win'")} win, count(*) n
               FROM agent_games WHERE {WHERE} GROUP BY ALL""", PARAMS)
    if len(df):
        pv = df.pivot(index="opp", columns="me", values="win").sort_index(ascending=False)
        nv = df.pivot(index="opp", columns="me", values="n").reindex_like(pv)
        fig = go.Figure(go.Heatmap(
            z=pv.values, x=[f"{c}–{c + 10}%" for c in pv.columns], y=[f"{r}–{r + 10}%" for r in pv.index],
            customdata=nv.values, colorscale=[[0, "#e34948"], [0.5, "#f3f2ef"], [1, "#1baf7a"]], zmin=0, zmax=100,
            hovertemplate="me %{x}, opponent %{y}<br>win rate %{z:.1f}%<br>%{customdata:,} games<extra></extra>",
            colorbar=dict(title="Win %")))
        fig.update_layout(height=480, xaxis_title="My cooperation rate", yaxis_title="Opponent's cooperation rate")
        st.plotly_chart(fig, width="stretch")

# ================================================================== matchups
with tabs[3]:
    mm = {"Win rate (%)": PCT("result='win'"), "Average margin": "avg(margin)",
          "Cooperation rate (%)": "avg(coop_rate) * 100", "Tie rate (%)": PCT("result='tie'"),
          "Games played": "count(*)"}
    metric = st.selectbox("Metric (row agent vs column opponent)", list(mm))
    df = q(f"SELECT agent, opponent, {mm[metric]} v, count(*) n FROM agent_games WHERE {WHERE} GROUP BY ALL", PARAMS)
    if len(df):
        pv = df.pivot(index="agent", columns="opponent", values="v")
        diverging = metric in ("Win rate (%)", "Average margin")
        fig = px.imshow(pv, text_auto=".0f", aspect="auto",
                        color_continuous_scale=[[0, "#e34948"], [0.5, "#f3f2ef"], [1, "#1baf7a"]] if diverging
                        else [[0, "#f3f2ef"], [1, "#2a78d6"]],
                        color_continuous_midpoint=(50 if "rate" in metric else 0) if diverging else None,
                        labels=dict(x="Opponent", y="Agent", color=metric))
        fig.update_layout(height=560)
        st.plotly_chart(fig, width="stretch")
        st.caption("Read across a row: how that agent does against each opponent.")

# ================================================================== strategies
with tabs[4]:
    if "probes" not in TABLES:
        st.info(f"No strategy fingerprints yet (taken every {cfg.PROBE_EVERY:,} games).")
    else:
        st.markdown(
            f"Every {cfg.PROBE_EVERY:,} games, each agent is shown a few standard situations and we record which move it "
            "would choose. Nothing here is fed back to the agents; it only describes what they learned on "
            "their own. **C** = cooperate, **D** = defect. In `after_CD`, the first letter is the agent's "
            "own last move and the second is the opponent's.")
        situations = ["start", "after_CC", "after_CD", "after_DC", "after_DD", "long_coop",
                      "betrayed_after_coop", "long_defect"]
        cols = ", ".join(f"arg_max({s}, game_id) {s}" for s in situations)
        latest = q(f"""SELECT agent, {cols}, arg_max(resembles, game_id) resembles,
                       max(game_id) as_of_game FROM probes GROUP BY agent ORDER BY agent""")
        st.subheader("Current strategy of each agent")
        st.dataframe(latest, hide_index=True, width="stretch")

        st.subheader("How the population's strategies evolved")
        df = q("SELECT game_id, resembles, count(*) n FROM probes GROUP BY ALL ORDER BY game_id")
        fig = px.area(df, x="game_id", y="n", color="resembles", labels={"n": "Agents", "game_id": "Game number"})
        fig.update_layout(height=340, legend_title="", hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

        st.subheader("One agent's strategy over time")
        who = st.selectbox("Agent", ALL_AGENTS)
        df = q(f"SELECT game_id, {', '.join(situations)} FROM probes WHERE agent = ? ORDER BY game_id", [who])
        if len(df):
            z = (df[situations] == "C").astype(int).T.values
            fig = go.Figure(go.Heatmap(
                z=z, x=df["game_id"], y=situations, zmin=0, zmax=1, xgap=1, ygap=2, showscale=False,
                colorscale=[[0, "#e34948"], [0.5, "#e34948"], [0.5, "#1baf7a"], [1, "#1baf7a"]],
                customdata=df[situations].T.values,
                hovertemplate="game %{x}<br>%{y}: %{customdata}<extra></extra>"))
            fig.update_layout(height=340, xaxis_title="Game number", yaxis_autorange="reversed")
            st.plotly_chart(fig, width="stretch")
            st.caption("Green = would cooperate, red = would defect.")

# ================================================================== tournaments
with tabs[5]:
    if "tournaments" not in TABLES:
        st.info(f"No tournaments yet (one every {cfg.TOURNAMENT_EVERY:,} games).")
    else:
        st.markdown(f"Every {cfg.TOURNAMENT_EVERY:,} games, all agents play each other with **no random moves and no learning**. "
                    "This shows each agent's real skill at that moment.")
        df = q("""SELECT tournament_at_game, agent, sum(score) points, avg(CASE WHEN result='win' THEN 100.0 ELSE 0 END) win_pct,
                         avg(coop_rate) * 100 coop_pct FROM tournaments GROUP BY ALL ORDER BY tournament_at_game""")
        df["rank"] = df.groupby("tournament_at_game")["points"].rank(ascending=False, method="min")
        last = df[df.tournament_at_game == df.tournament_at_game.max()].sort_values("points", ascending=False)
        c1, c2 = st.columns(2)
        with c1:
            st.subheader(f"Latest tournament (game {int(last.tournament_at_game.iloc[0]):,})")
            fig = px.bar(last, x="points", y="agent", orientation="h", color="agent", color_discrete_map=CMAP,
                         hover_data={"win_pct": ":.1f", "coop_pct": ":.1f"}, labels={"points": "Total points", "agent": ""})
            fig.update_layout(height=420, showlegend=False, yaxis_categoryorder="total ascending")
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.subheader("Rank over time")
            fig = px.line(df, x="tournament_at_game", y="rank", color="agent", color_discrete_map=CMAP, markers=True,
                          labels={"tournament_at_game": "Game number", "rank": "Rank (1 = best)"})
            fig.update_layout(height=420, yaxis_autorange="reversed", legend_title="", hovermode="x unified")
            st.plotly_chart(fig, width="stretch")

# ================================================================== game viewer
with tabs[6]:
    c1, c2, c3 = st.columns(3)
    va = c1.selectbox("Agent", ["any"] + ALL_AGENTS)
    vo = c2.selectbox("Opponent", ["any"] + ALL_AGENTS)
    vr = c3.selectbox("Result (for the agent)", ["any", "win", "tie", "loss"])
    cond, prm = [WHERE], list(PARAMS)
    if va != "any":
        cond.append("agent = ?"); prm.append(va)
    if vo != "any":
        cond.append("opponent = ?"); prm.append(vo)
    if vr != "any":
        cond.append("result = ?"); prm.append(vr)
    recent = q(f"""SELECT game_id, agent, opponent, result, score, opp_score, rounds,
                          round(coop_rate * 100, 1) coop_pct, round(explored_pct * 100, 1) random_pct
                   FROM agent_games WHERE {' AND '.join(cond)} ORDER BY game_id DESC LIMIT 200""", prm)
    st.caption("Latest 200 matching games. Pick a game id below to replay it.")
    st.dataframe(recent, hide_index=True, width="stretch", height=240)
    default = int(recent.game_id.iloc[0]) if len(recent) else int(info.g1)
    gid = st.number_input("Game id", min_value=1, value=default, step=1)

    @st.cache_data(max_entries=50)
    def get_game(gid, stamp):
        return q("SELECT * FROM game_history WHERE game_id = ?", [gid])

    if "game_history" in TABLES:
        g = get_game(int(gid), STAMP)
        if not len(g):
            st.warning(f"Game not found (it may not be written yet: the trainer saves every {cfg.CHECKPOINT_SECONDS} s).")
        else:
            g = g.iloc[0]
            st.markdown(f"**Game {gid}**: **{g.agent_1}** {g.score_1} – {g.score_2} **{g.agent_2}** · "
                        f"{g.rounds} rounds · winner: **{g.winner}**")
            rows = [(g.agent_1, g.moves_1), (g.agent_2, g.moves_2)]
            fig = go.Figure()
            for label, color in MOVE_COLORS.items():
                xs, ys = [], []
                for name, s in rows:
                    for i, ch in enumerate(s):
                        kind = f"{ch.upper()} ({'random' if ch.islower() else 'chosen'})"
                        if kind == label:
                            xs.append(i + 1)
                            ys.append(name)
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=label,
                                         marker=dict(symbol="square", size=14, color=color),
                                         hovertemplate="round %{x}<br>%{y}: " + label + "<extra></extra>"))
            fig.update_layout(height=200, xaxis_title="Round", legend=dict(orientation="h", y=1.25),
                              yaxis=dict(categoryorder="array", categoryarray=[g.agent_2, g.agent_1]),
                              margin=dict(t=40, b=40))
            st.plotly_chart(fig, width="stretch")

            pay = {("C", "C"): 3, ("C", "D"): 0, ("D", "C"): 5, ("D", "D"): 1}
            s1 = s2 = 0
            cum = []
            for i, (x, y) in enumerate(zip(g.moves_1.upper(), g.moves_2.upper())):
                s1 += pay[(x, y)]
                s2 += pay[(y, x)]
                cum += [(i + 1, g.agent_1, s1), (i + 1, g.agent_2, s2)]
            cdf = pd.DataFrame(cum, columns=["round", "agent", "score"])
            fig = px.line(cdf, x="round", y="score", color="agent", color_discrete_map=CMAP,
                          labels={"score": "Running score", "round": "Round"})
            fig.update_layout(height=300, hovermode="x unified", legend_title="")
            st.plotly_chart(fig, width="stretch")
            with st.expander("Raw move strings (uppercase = network's choice, lowercase = random move)"):
                st.code(f"{g.agent_1}: {g.moves_1}\n{g.agent_2}: {g.moves_2}")

# ================================================================== custom SQL
with tabs[7]:
    st.markdown("Query anything with SQL ([DuckDB dialect](https://duckdb.org/docs/sql/introduction)). Tables:\n"
                "- `agent_games`: one row per agent per game (from that agent's point of view)\n"
                "- `game_history`: one row per game, including the full move strings `moves_1` and `moves_2`\n"
                "- `probes`: strategy fingerprints · `tournaments`: tournament games\n\n"
                "The sidebar filters do **not** apply here.")
    examples = {
        "Win rate when cooperating less than 20%": """SELECT agent, count(*) games,
  round(avg(CASE WHEN result='win' THEN 100.0 ELSE 0 END), 1) win_pct
FROM agent_games WHERE coop_rate < 0.20
GROUP BY agent ORDER BY win_pct DESC""",
        "Who exploits whom (avg margin)": """SELECT agent, opponent, round(avg(margin), 1) avg_margin, count(*) games
FROM agent_games WHERE explored_pct < 0.1
GROUP BY ALL ORDER BY avg_margin DESC LIMIT 20""",
        "Most common opening (first 5 moves)": """SELECT upper(moves_1[1:5]) opening, count(*) n
FROM game_history GROUP BY opening ORDER BY n DESC LIMIT 15""",
        "Win rate by first move": """SELECT agent, first_move, count(*) games,
  round(avg(CASE WHEN result='win' THEN 100.0 ELSE 0 END), 1) win_pct
FROM agent_games GROUP BY ALL ORDER BY agent, first_move""",
    }
    ex = st.selectbox("Start from an example", list(examples))
    sql = st.text_area("SQL", examples[ex], height=160)
    try:
        res = q(sql)
        st.dataframe(res, hide_index=True, width="stretch")
        if len(res) and len(res.columns) >= 2:
            with st.expander("Chart this result"):
                c1, c2, c3, c4 = st.columns(4)
                kind = c1.selectbox("Chart", ["bar", "line", "scatter"])
                x = c2.selectbox("X", res.columns, index=0)
                y = c3.selectbox("Y", res.columns, index=len(res.columns) - 1)
                col = c4.selectbox("Color", ["(none)"] + list(res.columns))
                fn = {"bar": px.bar, "line": px.line, "scatter": px.scatter}[kind]
                kw = dict(color=col) if col != "(none)" else {}
                if col == "agent":
                    kw["color_discrete_map"] = CMAP
                st.plotly_chart(fn(res, x=x, y=y, **kw), width="stretch")
        st.download_button("Download result as CSV", res.to_csv(index=False), "query_result.csv")
    except Exception as e:
        st.error(str(e))
