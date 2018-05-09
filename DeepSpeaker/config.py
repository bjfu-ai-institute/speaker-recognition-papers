"""
The config of model, train, enrollment and test.
"""


CONV_WEIGHT_DECAY = 0.0010

FC_WEIGHT_DECAY = 0.0015

BN_EPSILON = 0.001

MAX_STEP = 75

N_GPU = 4

LEARNING_RATE = 0.001

BATCH_SIZE = 100

N_SPEAKER = 1000

N_RES_BLOCKS = 4

OUT_CHANNEL = [64, 128, 256, 512]

# OUT_CHANNEL is the array represents number of out channel of each residual block. 
# So the length of OUT_CHANNEL must equal to the N_RES_BLOCKS 
