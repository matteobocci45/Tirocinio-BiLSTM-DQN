import numpy as np

from replay_buffer import ReplayBuffer



buffer = ReplayBuffer(
    capacity=100
)



state = np.array(
    [
        0.9,
        300,
        400,
        2,
        1,
        1,
        1,
        0,
        0.8
    ]
)



next_state = np.array(
    [
        0.1,
        0,
        10,
        3,
        0,
        0,
        0,
        1,
        0.2
    ]
)



buffer.push(
    state,
    4,          # BLOCK_IP
    100,
    next_state,
    True
)



print(
    "Buffer size:",
    len(buffer)
)



sample = buffer.sample(1)



print("\nSample:")

for item in sample:
    print(item)