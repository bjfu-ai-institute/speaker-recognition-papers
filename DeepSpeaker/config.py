"""
The config of model, train, enrollment and test.
"""


CONV_WEIGHT_DECAY = 0.0010

FC_WEIGHT_DECAY = 0.0015

BN_EPSILON = 0.001

N_GPU = 4

N_SPEAKER = 1000

N_RES_BLOCKS = 4

OUT_CHANNEL = [64, 128, 256, 512]

assert len(OUT_CHANNEL) == N_RES_BLOCKS
