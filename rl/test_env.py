from cyber_env import CyberDefenseEnv


env = CyberDefenseEnv()


state, info = env.reset()


print("SCENARIO:", info["scenario"])

print("START STATE:")
print(state)



done = False


while not done:


    action = env.action_space.sample()


    state, reward, done, _, info = env.step(action)


    print(
        "STEP:",
        info["step"],
        "ACTION:",
        info["action"],
        "REWARD:",
        reward
    )