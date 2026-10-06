from collections import deque
import random
import numpy as np



class ReplayBuffer:
    """
    Experience Replay Memory per DQN.

    Memorizza:

    state
    action
    reward
    next_state
    done

    """



    def __init__(
            self,
            capacity=100000
    ):

        self.memory = deque(
            maxlen=capacity
        )



    def push(
            self,
            state,
            action,
            reward,
            next_state,
            done
    ):
        """
        Inserisce una nuova esperienza.
        """


        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )



    def sample(
            self,
            batch_size
    ):
        """
        Estrae un batch casuale
        per il training del DQN.
        """


        batch = random.sample(
            self.memory,
            batch_size
        )


        states, actions, rewards, next_states, dones = zip(*batch)


        return (

            np.array(states),

            np.array(actions),

            np.array(rewards, dtype=np.float32),

            np.array(next_states),

            np.array(dones, dtype=np.float32)

        )



    def __len__(self):

        return len(self.memory)