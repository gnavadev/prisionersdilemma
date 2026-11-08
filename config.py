"""All tunable settings. Safe to edit between runs, EXCEPT the network shape
(HISTORY_LEN, HIDDEN_LAYERS): changing those makes saved models incompatible.
Raising NUM_AGENTS later is fine: new agents are created fresh and join in."""

DATA_DIR = "data"

# ---------------------------------------------------------------- hardware
DEVICE = "auto"               # "auto" = NVIDIA GPU if available, else CPU. Or "cuda" / "cpu".
GPU_PARALLEL_GAMES = 512      # games played simultaneously on the GPU (more = faster, ~1000+ games/s)

# ---------------------------------------------------------------- population
NUM_AGENTS = 10               # learning agents; every game pairs two random distinct players

# Non-learning players with a fixed strategy that join the random pairings.
# Remove a name from this list to take that player out.
FIXED_AGENTS = ["Tit for Tat", "Generous Tit for Tat", "Grim Trigger"]
GENEROUS_TFT_FORGIVENESS = 1 / 3   # chance Generous TFT cooperates after being defected on

# ---------------------------------------------------------------- game rules
PAYOFF = {  # (my_move, opp_move) -> my points ; 0 = Cooperate, 1 = Defect
    (0, 0): 3,  # both cooperate
    (0, 1): 0,  # I cooperate, they defect (sucker)
    (1, 0): 5,  # I defect, they cooperate (temptation)
    (1, 1): 1,  # both defect
}

# Random game length so nobody can exploit a known last round.
MIN_ROUNDS = 10
CONTINUE_PROB = 0.98          # expected length ~ MIN_ROUNDS + 50
MAX_ROUNDS = 400

# ------------------------------------------------------------ reward
# Each agent's objective is simply to maximise ITS OWN points in every game
# (reward per round = points it scored). Winning, tying or losing against the
# opponent gives nothing extra. Set these above 0 to make agents care about
# beating the opponent; note that this pushes them towards always defecting,
# because defecting is the only way to outscore the other player.
WIN_BONUS = 0.0
TIE_PENALTY = 0.0
LOSS_PENALTY = 0.0
REWARD_SCALE = 0.1            # rewards are multiplied by this before training

# --------------------------------------------------------------- agent setup
HISTORY_LEN = 6               # how many past rounds each agent sees
HIDDEN_LAYERS = [64, 64]

# Every agent starts by playing randomly (eps_start=1.0) and gradually relies
# on what it has learned. Override per agent with AGENT_OVERRIDES.
AGENT_DEFAULTS = dict(
    lr=5e-4,
    gamma=0.99,                # how much future rounds matter (cooperation needs foresight)
    batch_size=256,
    replay_capacity=60_000,
    learn_start=1_000,
    train_every=8,             # one update per 8 moves, 256 samples each (32 samples per move)
    target_update_every=1_000,
    eps_start=1.0,
    eps_min=0.03,
    eps_decay_steps=150_000,  # in the agent's own moves
)
AGENT_OVERRIDES = {
    # "agent_03": dict(lr=1e-3, gamma=0.99),
}

# ------------------------------------------------------------------ logging
CHECKPOINT_SECONDS = 60       # save models + write CSVs this often (also on Ctrl+C)
PRINT_SECONDS = 10            # console summary this often
PROBE_EVERY = 2_000           # games between strategy "fingerprints" of every agent
TOURNAMENT_EVERY = 20_000     # games between round-robin tournaments (no exploration, no learning)
TOURNAMENT_GAMES_PER_PAIR = 2
ELO_K = 16
