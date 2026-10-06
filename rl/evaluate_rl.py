import numpy as np
from collections import Counter, defaultdict

from cyber_env import CyberDefenseEnv
from dqn_agent import DQNAgent


MODEL_PATH = "../models/dqn.keras"

EPISODES = 200


def run_episode(env, agent):
    """
    Esegue un episodio di valutazione.

    Returns
    -------
    dict
        Contiene le statistiche dell'episodio.
    """

    state, _ = env.reset()

    done = False

    total_reward = 0

    steps = 0

    actions = []

    while not done:

        action = agent.act(state)

        next_state, reward, done, truncated, info = env.step(action)

        actions.append(info["action"])

        total_reward += reward

        steps += 1

        state = next_state

    success = (
        info["is_attack"]
        and env.state[0] == 0
    )

    return {

        "scenario": info["scenario"],

        "reward": total_reward,

        "steps": steps,

        "success": success,

        "actions": actions
    }


def evaluate():

    env = CyberDefenseEnv()

    state_size = env.observation_space.shape[0]

    action_size = env.action_space.n

    agent = DQNAgent(
        state_size,
        action_size
    )

    agent.load(MODEL_PATH)

    # Evaluation -> niente esplorazione
    agent.epsilon = 0.0

    rewards = []

    steps_list = []

    successes = []

    action_counter = Counter()

    scenario_rewards = defaultdict(list)

    scenario_counter = Counter()

    for _ in range(EPISODES):

        result = run_episode(env, agent)

        rewards.append(result["reward"])

        steps_list.append(result["steps"])

        scenario_rewards[result["scenario"]].append(
            result["reward"]
        )

        scenario_counter[result["scenario"]] += 1

        action_counter.update(result["actions"])

        if result["scenario"] != "NORMAL":

            successes.append(result["success"])

    print("\n" + "=" * 50)
    print("      RL EVALUATION REPORT")
    print("=" * 50)

    print(f"Episodes: {EPISODES}")

    print(f"Average Reward : {np.mean(rewards):.2f}")

    print(f"Average Steps  : {np.mean(steps_list):.2f}")

    if len(successes):

        print(
            f"Success Rate   : {100*np.mean(successes):.2f}%"
        )

    print("\nAction Distribution")

    print("-" * 50)

    for action_name in env.ACTIONS.values():

        print(
            f"{action_name:<20}"
            f"{action_counter[action_name]}"
        )

    print("\nReward by Scenario")

    print("-" * 50)

    for scenario in sorted(scenario_rewards.keys()):

        if len(scenario_rewards[scenario]):

            print(

                f"{scenario:<20}"

                f"{np.mean(scenario_rewards[scenario]):8.2f}"

                f" ({scenario_counter[scenario]} episodes)"

            )

    print("=" * 50)


if __name__ == "__main__":

    evaluate()