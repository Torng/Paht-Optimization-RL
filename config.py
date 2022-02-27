ORDER_PATH = "data/orders_v2.csv"
DISTANCE_PATH = "data/distance_matrix_v2.csv"
TIME_PATH = "data/time_matrix_v2.csv"

MORTOS = ["A", "B", "C"]



# RL module parameter
# MAX STEP -> agent only go less than max step
MAX_STEP = 23
REPLAY_BUFFER_CAPACITY = 20000
# EPSILON_GREEDY
EPS_END = 0.05
EPS_START = 0.9
EPS_DECAY = 200

REWARD_SCALE_FACTOR = 1.0  # default
# GRADIENT_CLIPPING = 1.0
# TARGET_UPDATE_TAU = 0.1
TARGET_UPDATE_PERIOD = 5
DISCOUNT = 0.9

# DL parameter
LEARNING_RATE = 0.001
BATCH_SIZE = 64
TRAINING_EPISODE = 20000
LOG_INTERVAL = 200
NUM_EVAL_EPISODES = 10
EVAL_INTERVAL = 1000
MODEL_FILE = './policy_set/policy'
# Early stopping
BEST_RETURN = -1e10
TOLERANCE_STEP = 10

LOAD_MODEL = False
