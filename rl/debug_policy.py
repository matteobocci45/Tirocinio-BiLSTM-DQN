import numpy as np

from cyber_env import CyberDefenseEnv
from dqn_agent import DQNAgent


MODEL_PATH = "../models/dqn.keras"


def test_state(name, state, env, agent):

    print("\n")
    print("="*50)
    print(name)
    print("="*50)


    q = agent.model.predict(
        np.expand_dims(state, axis=0),
        verbose=0
    )[0]


    for i, value in enumerate(q):

        print(
            f"{env.ACTIONS[i]:<20}: {value:.3f}"
        )


    print(
        "\nBEST:",
        env.ACTIONS[np.argmax(q)]
    )



env = CyberDefenseEnv()


agent = DQNAgent(
    env.observation_space.shape[0],
    env.action_space.n
)


agent.load(MODEL_PATH)


agent.epsilon = 0



normal = np.array([
    0.05,
    5,
    20,
    14,
    0,
    0,
    0,
    1,
    0.05
], dtype=np.float32)



brute_force = np.array([
    0.95,
    400,
    500,
    2,
    1,
    1,
    1,
    0,
    0.95
], dtype=np.float32)



privileged = np.array([
    0.90,
    120,
    200,
    23,
    1,
    1,
    1,
    0,
    0.90
], dtype=np.float32)



test_state(
    "NORMAL",
    normal,
    env,
    agent
)


test_state(
    "BRUTE FORCE",
    brute_force,
    env,
    agent
)


test_state(
    "PRIVILEGED ATTACK",
    privileged,
    env,
    agent
)