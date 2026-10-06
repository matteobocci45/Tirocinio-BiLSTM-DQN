import os
import numpy as np

from cyber_env import CyberDefenseEnv
from replay_buffer import ReplayBuffer
from dqn_agent import DQNAgent



# -------------------------
# Configurazione training
# -------------------------

EPISODES = 1000

BATCH_SIZE = 64

TARGET_UPDATE = 20

MODEL_PATH = "../models/dqn.keras"



def train():


    env = CyberDefenseEnv()



    state_size = env.observation_space.shape[0]

    action_size = env.action_space.n



    agent = DQNAgent(

        state_size,

        action_size

    )



    memory = ReplayBuffer(

        capacity=100000

    )



    reward_history = []



    success_history = []



    for episode in range(EPISODES):


        state, info = env.reset()


        done = False


        total_reward = 0


        steps = 0



        while not done:



            # --------------------
            # scelta azione
            # --------------------

            action = agent.act(
                state
            )



            # --------------------
            # ambiente
            # --------------------

            next_state, reward, done, truncated, info = env.step(
                action
            )



            # --------------------
            # memoria esperienza
            # --------------------

            memory.push(

                state,

                action,

                reward,

                next_state,

                done

            )



            state = next_state



            total_reward += reward


            steps += 1



            # --------------------
            # training rete
            # --------------------

            agent.train_step(

                memory,

                BATCH_SIZE

            )



        reward_history.append(
            total_reward
        )



        # --------------------
        # aggiornamento target
        # --------------------

        if episode % TARGET_UPDATE == 0:

            agent.update_target_network()



        # --------------------
        # statistiche
        # --------------------

        if env.scenario != "NORMAL":
                success_history.append(1)

        elif env.state[0] <= 0.10:
            success_history.append(1)
        else:
            success_history.append(0)



        if episode % 20 == 0:


            avg_reward = np.mean(

                reward_history[-20:]

            )


            success_rate = np.mean(

                success_history[-20:]

            )


            print(

                f"Episode {episode:4d}"

                f" | Scenario={env.scenario}"

                f" | Reward={avg_reward:.2f}"

                f" | Success={success_rate:.2f}"

                f" | epsilon={agent.epsilon:.3f}"

            )



    # -------------------------
    # salvataggio modello
    # -------------------------

    os.makedirs(

        "../models",

        exist_ok=True

    )


    agent.save(

        MODEL_PATH

    )


    print("\nTraining completato")

    print(
        "Modello salvato:",
        MODEL_PATH
    )




if __name__ == "__main__":

    train()