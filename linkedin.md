# LinkedIn post

**Images to attach, in this order** (in `analysis/figures/`):
1. `2_winning_is_not_scoring.png`: the hook
2. `1_learning_to_cooperate.png`
3. `4_learned_forgiveness.png`
4. `3_grim_trigger_punishes_mistakes.png`
5. `5_strategies_that_emerged.png`
6. `6_points_and_cooperation_over_time.png`: points and cooperation of every player, game 0 to latest

Copy everything between the lines below. It's plain text because LinkedIn doesn't render markdown.

---

I had 10 neural networks play 2.5 million games (149M rounds) of the Prisoner's Dilemma. They started out choosing at random. Here's what they worked out on their own.

The setup
• Each round, two players cooperate or defect. Both cooperate: 3 points each. Both defect: 1 each. One defects on a cooperator: 5 for the defector, 0 for the other.
• Games end at a random point (~60 rounds on average), so there's no known last round.
• 10 small networks (27→64→64→2) trained with Double Q-learning. Each sees the last 6 rounds, the current score gap and both players' cooperation rates. They don't know who they're playing.
• Their only goal: score as many points as possible in each game.
• Three fixed strategies join the random pairings: Tit for Tat, Generous Tit for Tat (forgives 1 in 3 defections) and Grim Trigger (defects forever after the first defection).

What came out
1. Cooperation emerged without being programmed. Mutual cooperation went from 20% of rounds under random play to 91%. Learners went from 2.10 to 2.90 points per round, where 3.0 is the ceiling. In tournaments with no random moves, they cooperate 99.9% of the time.

2. Winning is not the same as scoring. Tit for Tat never won a game (it can't outscore anyone). It still had the best average: 2.99 points per round. Grim Trigger won 41% of its games and finished last with 2.40.

3. Unforgiving rules are expensive when mistakes happen. Learners still make 3% random moves. Against Grim Trigger, one accidental defection ends cooperation for the rest of the game, and both sides drop to 2.1–2.3 points per round instead of about 2.97.

4. They learned to forgive. Right after an opponent defected, learners still cooperated 68% of the time, twice as often as Generous Tit for Tat. In their world, most defections are noise, not intent.

5. There was no single answer. The final population mixes Tit for Tat (15%), Tit for Tat that also breaks out of mutual defection (15%), "forgive a betrayal but not a feud" (29%) and cooperating in every test situation (36%).

A lesson from my first version: I rewarded beating the opponent instead of scoring points. Defecting is the only way to outscore someone in this game, so cooperation fell to about 11% and scores to 1.3 points per round. What you measure is what you get.

Engineering
• The networks are tiny, so the GPU only helps with batching: 512 games run in parallel, and all 10 networks train in one batched step captured as a CUDA graph. That took it from 95 to about 1,070 games/s on a laptop RTX 4070.
• Every move is logged to CSV and explored in a Streamlit + Plotly + DuckDB dashboard.

Caveats: 10 players, one payoff table and fixed 3% noise. The noise level likely shapes how forgiving they become; that's my next test.

Axelrod's tournaments in the 1980s favoured strategies that are nice, retaliatory and forgiving. Starting from nothing, these networks got most of the way there.

#GameTheory #ReinforcementLearning #Python #PyTorch #DataVisualization
