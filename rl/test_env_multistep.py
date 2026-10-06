from cyber_env import CyberDefenseEnv


env = CyberDefenseEnv()


episodes = 3


for episode in range(episodes):

    print("\n==============================")
    print(f"EPISODE {episode + 1}")
    print("==============================")


    state, info = env.reset()


    print(
        "Scenario:",
        env.scenario
    )


    print(
        "Initial state:"
    )

    print(state)



    done = False

    total_reward = 0



    while not done:


        # per ora azione casuale
        # il DQN arriverà dopo

        action = env.action_space.sample()



        next_state, reward, done, truncated, info = env.step(action)



        total_reward += reward



        print("\nStep:", info["step"])

        print(
            "Action:",
            info["action"]
        )

        print(
            "Reward:",
            reward
        )

        print(
            "Attack probability:",
            round(
                float(next_state[0]),
                3
            )
        )



    print("\nEpisode finished")

    print(
        "Total reward:",
        total_reward
    )