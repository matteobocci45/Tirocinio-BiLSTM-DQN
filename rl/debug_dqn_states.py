import os

import numpy as np

from rl.dqn_agent import DQNAgent


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DQN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "dqn.keras"
)


ACTIONS = {
    0: "ALLOW",
    1: "MONITOR",
    2: "MFA",
    3: "BLOCK_SESSION",
    4: "BLOCK_IP",
    5: "QUARANTINE",
}


REQUESTS_SCALE = 4486.0


def normalize_state(state):

    state = np.asarray(
        state,
        dtype=np.float32
    ).copy()

    # 0 attack_probability
    # già 0..1

    # 1 failure_ratio
    # già 0..1 -> NON dividere

    # 2 requests_5m
    state[2] /= REQUESTS_SCALE

    # 3 hour
    state[3] /= 23.0

    return np.clip(
        state,
        0.0,
        1.0
    )


def test_state(
        agent,
        name,
        raw_state
):

    state = normalize_state(
        raw_state
    )

    q_values = agent.model.predict(
        np.expand_dims(
            state,
            axis=0
        ),
        verbose=0
    )[0]

    action = int(
        np.argmax(q_values)
    )

    print(
        "\n"
        + "=" * 70
    )

    print(name)

    print(
        "=" * 70
    )

    print(
        "RAW STATE:"
    )

    print(
        raw_state
    )

    print(
        "\nNORMALIZED:"
    )

    print(
        state
    )

    print(
        "\nQ-VALUES:"
    )

    for action_id, q_value in enumerate(
        q_values
    ):

        print(
            f"{ACTIONS[action_id]:<20}: "
            f"{float(q_value):.4f}"
        )

    print(
        "\nDECISION:",
        ACTIONS[action]
    )


def main():

    agent = DQNAgent(
        state_size=9,
        action_size=6
    )

    agent.load(
        DQN_MODEL_PATH
    )

    agent.epsilon = 0.0


    # =================================================
    # 1. COMPLETAMENTE NORMALE
    # =================================================

    test_state(
        agent,
        "1 - NORMAL LOW RISK",
        [
            0.01,   # attack_probability
            0.00,   # failure_ratio
            800,    # requests_5m
            12,     # hour
            0,      # is_night
            0,      # privileged
            0,      # critical
            1,      # certificate
            0.02,   # historical_risk
        ]
    )


    # =================================================
    # 2. NORMALE MA MOLTO TRAFFICO
    # =================================================

    test_state(
        agent,
        "2 - HIGH TRAFFIC",
        [
            0.02,
            0.01,
            950,
            14,
            0,
            1,
            0,
            0,
            0.05,
        ]
    )


    # =================================================
    # 3. SOSPETTO
    # =================================================

    test_state(
        agent,
        "3 - SUSPICIOUS",
        [
            0.40,
            0.25,
            1500,
            23,
            1,
            0,
            0,
            0,
            0.40,
        ]
    )


    # =================================================
    # 4. ATTACCO PROBABILE
    # =================================================

    test_state(
        agent,
        "4 - LIKELY ATTACK",
        [
            0.85,
            0.80,
            2500,
            2,
            1,
            0,
            0,
            0,
            0.85,
        ]
    )


    # =================================================
    # 5. ATTACCO PRIVILEGIATO
    # =================================================

    test_state(
        agent,
        "5 - PRIVILEGED ATTACK",
        [
            0.98,
            0.90,
            4000,
            2,
            1,
            1,
            1,
            0,
            0.95,
        ]
    )


    # =================================================
    # 6. BASSA PROBABILITA MA RISORSA CRITICA
    # =================================================

    test_state(
        agent,
        "6 - NORMAL + CRITICAL RESOURCE",
        [
            0.01,
            0.00,
            850,
            15,
            0,
            0,
            1,
            1,
            0.03,
        ]
    )


if __name__ == "__main__":

    main()