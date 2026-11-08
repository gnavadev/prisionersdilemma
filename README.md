# Prisoner's Dilemma: a population of self-learning neural networks

```
./train.sh                # train forever; Ctrl+C stops and saves (twice = force)
./train.sh --games 20000  # stop after 20,000 games
./dashboard.sh            # interactive dashboard at http://localhost:8501 (works while training runs)
```

Training always continues from `data/`. Delete `data/` to start from scratch, or delete
`data/agents/agent_XX/` to reset just that agent. The earlier 2-agent run is kept in `data_old_2agents/`.

## GPU
Training runs on the NVIDIA GPU automatically (`DEVICE` in `config.py`, or `./train.sh --device cpu`).
On the GPU, 512 games run at the same time (`GPU_PARALLEL_GAMES`), all 10 networks train together
in one batched step captured as a CUDA graph, and replay memories live in GPU memory. That's about
1,100 games/s, versus about 100 on the CPU. Both modes use the same checkpoint files, so you can switch
between them at any time. Models and CSVs are saved every `CHECKPOINT_SECONDS` (60 s) and on Ctrl+C.

## How it works
- **10 learning agents** (`NUM_AGENTS` in `config.py`) plus **3 fixed-strategy players**
  (`FIXED_AGENTS`). Each game pairs two of them at random.
  - **Tit for Tat:** cooperates first, then copies the opponent's last move.
  - **Generous Tit for Tat:** like Tit for Tat, but forgives a defection 1/3 of the time.
  - **Grim Trigger:** cooperates until the opponent defects once, then always defects for the rest of that game.
- **The learners have no built-in strategy.** Every learning agent is a fresh neural network (Double-DQN) that starts out playing
  100% randomly and learns only from its own games. Random play gradually drops to 3%.
  Each agent has its own network, memory, random seed, and hyper-parameters (`AGENT_OVERRIDES`).
- **What an agent sees:** the last 6 rounds, who is ahead in the current game, and both players'
  cooperation rates in that game. It does not know who its opponent is.
- **Objective:** maximise its own points in each game. Beating the opponent gives nothing extra
  (`WIN_BONUS`, `TIE_PENALTY`, and `LOSS_PENALTY` are all 0).
- **Game length is random** (10+ rounds, about 60 on average), so nobody knows which round is the last.

## Data (`data/`)
| File | One row per… |
|---|---|
| `game_history.csv` | game: both agents, scores, winner, and full move strings (`C`/`D` = the network's choice, `c`/`d` = a random exploration move) |
| `agents/agent_XX/games.csv` | game *this agent* played, from its point of view: opponent, score, result, cooperation, retaliation and forgiveness, exploration, Elo, training stats, and moves |
| `agents/agent_XX/probes.csv` | every 500 games: the move the agent would choose in standard situations (its strategy "fingerprint") |
| `agents/agent_XX/checkpoint.npz` | the agent's brain: network, optimizer, and replay memory (plus a `.prev` backup) |
| `tournaments.csv` | every 2,000 games: round-robin games between all agents with no randomness and no learning |
| `state.json`, `runs.csv`, `run_configs.jsonl` | game counter, Elo ratings, run sessions, and the config used for each run |

## Dashboard tabs
Leaderboard · Over time · Cooperation vs winning (including a condition builder, e.g. "win % when my
cooperation is < 30%") · Matchups heat-map · Strategies · Tournaments · Game viewer (replay any game
move by move) · Custom SQL (query and chart anything, then download it as CSV).
