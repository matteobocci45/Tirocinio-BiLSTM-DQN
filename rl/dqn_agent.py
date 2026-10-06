import random
import numpy as np

from keras import Model
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.losses import Huber
from keras.models import load_model



# -----------------------------
# Q Network
# -----------------------------

def build_q_network(
        state_size,
        action_size
):

    inputs = Input(
        shape=(state_size,)
    )


    x = Dense(
        128,
        activation="relu"
    )(inputs)


    x = Dense(
        64,
        activation="relu"
    )(x)


    x = Dense(
        32,
        activation="relu"
    )(x)

    action_size = int(action_size)
    outputs = Dense(
        action_size,
        activation="linear"
    )(x)



    model = Model(
        inputs,
        outputs
    )


    model.compile(

        optimizer=Adam(

            learning_rate=0.0003,

            clipnorm=1.0

        ),

        loss=Huber()

    )


    return model





# -----------------------------
# DQN Agent
# -----------------------------

class DQNAgent:


    def __init__(
            self,
            state_size,
            action_size
    ):


        self.state_size = state_size

        self.action_size = action_size
        self.train_counter = 0


        # discount factor

        self.gamma = 0.95



        # exploration

        self.epsilon = 1.0

        self.epsilon_min = 0.05

        self.epsilon_decay = 0.998



        # reti

        self.model = build_q_network(

            state_size,

            action_size

        )


        self.target_model = build_q_network(

            state_size,

            action_size

        )


        self.update_target_network()





    # -------------------------
    # Target Network
    # -------------------------

    def update_target_network(self):


        self.target_model.set_weights(

            self.model.get_weights()

        )





    # -------------------------
    # Action selection
    # -------------------------

    def act(
            self,
            state
    ):


        if np.random.rand() <= self.epsilon:

            return random.randrange(

                self.action_size

            )



        q_values = self.model.predict(

            np.expand_dims(
                state,
                axis=0
            ),

            verbose=0

        )


        return np.argmax(

            q_values[0]

        )





    # -------------------------
    # Training
    # -------------------------

    def train_step(
            self,
            replay_buffer,
            batch_size
    ):


        if len(replay_buffer) < batch_size:

            return



        (
            states,
            actions,
            rewards,
            next_states,
            dones

        ) = replay_buffer.sample(

            batch_size

        )



        current_q = self.model.predict(

            states,

            verbose=0

        )



        next_q = self.target_model.predict(

            next_states,

            verbose=0

        )



        for i in range(batch_size):


            if dones[i]:


                target = rewards[i]


            else:


                target = (

                    rewards[i]

                    +

                    self.gamma *

                    np.max(
                        next_q[i]
                    )

                )



            current_q[i][actions[i]] = target





        self.model.fit(

            states,

            current_q,

            epochs=1,

            verbose=0

        )



        self.train_counter += 1


        if self.train_counter % 100 == 0:

            self.update_target_network()



        if self.epsilon > self.epsilon_min:

            self.epsilon *= self.epsilon_decay





    # -------------------------
    # Save
    # -------------------------

    def save(
            self,
            path
    ):


        self.model.save(

            path

        )




    # -------------------------
    # Load
    # -------------------------

    def load(
            self,
            path
    ):


        self.model = load_model(

            path

        )


        self.update_target_network()