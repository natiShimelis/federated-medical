import torch

NUM_CLIENTS = 5        # number of simulated hospitals (clients) per round
NUM_ROUNDS = 20        # total communication rounds between clients and server
BATCH_SIZE = 32
LEARNING_RATE = 0.001
LOCAL_EPOCHS = 1       # each client does one full pass before sending weights back

# alpha controls how skewed the per-client class distribution is.
# 0.5 gives moderate heterogeneity — realistic for hospitals with different patient demographics.
# Lower values (e.g. 0.1) would make partitions almost single-class; higher values approach IID.
DIRICHLET_ALPHA = 0.5

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# proximal penalty weight for FedProx; 0.1 is the value used in the original FedProx paper
FEDPROX_MU = 0.1

# fixed seeds so experiments are reproducible across runs
RANDOM_SEED = 42

DATA_DIR = "./data/raw"
OUTPUT_DIR = "./outputs"
