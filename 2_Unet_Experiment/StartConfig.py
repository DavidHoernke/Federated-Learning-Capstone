# Max or explicitly defined number of epochs per round
NUM_EPOCHS = 25

# Early stopping patience for clients. 0= no early stopping, -1 = early stopping relative to the round number.
EPOCH_EARLY_STOPPING_PATIENCE = 2

# Batch size on each of the clients
BATCH_SIZE = 8

# How many rounds of federated learning can pass without aggregated DICE not improving before early stopping.
NUM_ROUNDS_NO_IMPROVEMENT_EARLY_STOP = 4

# Client dropout
CLIENT_PARTICIPATION_FRACTION = 1 # Represents what fraction of clients will be trained and used for eval each round

# max rounds
NUM_ROUNDS= 20
