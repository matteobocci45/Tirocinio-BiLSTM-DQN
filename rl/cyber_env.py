import random

import gymnasium as gym
import numpy as np

from gymnasium import spaces

from reward import RewardCalculator



class CyberDefenseEnv(gym.Env):

    """
    Adaptive Cyber Defense Environment.

    Simula una sessione di attacco
    composta da più step.


    Stato:

    0 attack_probability

    1 failed_requests_ratio

    2 requests_5m

    3 hour

    4 is_night

    5 privileged

    6 critical_resource

    7 has_certificate

    8 historical_risk

    """



    ACTIONS = {

        0:"ALLOW",

        1:"MONITOR",

        2:"MFA",

        3:"BLOCK_SESSION",

        4:"BLOCK_IP",

        5:"QUARANTINE"

    }



    SCENARIOS = [

        "NORMAL",
        "NORMAL",
        "NORMAL",
        "BRUTE_FORCE",
        "BOT",
        "PRIVILEGED_ATTACK",
        "RESOURCE_SCAN"

    ]



    MAX_STEPS = 5



    def __init__(self):

        super().__init__()



        self.action_space = spaces.Discrete(

            len(self.ACTIONS)

        )


        self.observation_space = spaces.Box(
            low=np.zeros(
                9,
                dtype=np.float32
            ),
            high=np.ones(
                9,
                dtype=np.float32
            ),
            dtype=np.float32
        )

        


        self.state = None

        self.scenario = None

        self.current_step = 0




    def reset(
            self,
            seed=None,
            options=None
    ):


        super().reset(seed=seed)


        self.current_step = 0


        self.scenario = random.choice(

            self.SCENARIOS

        )


        self.state = self._generate_initial_state()



        return self._get_normalized_state(), {
            "scenario": self.scenario
        }




    def _generate_initial_state(self):

        hour = random.randint(
            0,
            23
        )

        is_night = int(
            hour < 6
            or
            hour >= 22
        )


        # ====================================================
        # NORMAL
        # ====================================================

        if self.scenario == "NORMAL":

            # La maggior parte dei NORMAL ha
            # attack_probability molto bassa.
            if random.random() < 0.95:

                attack_probability = random.uniform(
                    0.0,
                    0.02
                )

            else:

                # Alcuni casi normali possono comunque
                # essere più ambigui / sospetti.
                attack_probability = random.uniform(
                    0.02,
                    0.30
                )


            # Failure ratio normalmente molto bassa.
            if random.random() < 0.97:

                failure_ratio = random.uniform(
                    0.0,
                    0.03
                )

            else:

                # Rari NORMAL con più fallimenti.
                failure_ratio = random.uniform(
                    0.05,
                    0.30
                )


            # Dai dati:
            # NORMAL median ~ 850
            # P95 ~ 890
            # P99 TRAIN NORMAL = 948
            requests_5m = random.uniform(
                650,
                950
            )


            # Dai dati reali:
            # circa 75% dei NORMAL sono privileged
            privileged = int(
                random.random() < 0.75
            )


            # Nei NORMAL critical_resource è rarissimo
            critical_resource = int(
                random.random() < 0.003
            )


            # Circa 28% dei NORMAL ha certificato
            has_certificate = int(
                random.random() < 0.28
            )


            # Historical risk:
            # nella maggior parte dei NORMAL è molto basso,
            # ma esiste una piccola coda ad alto rischio.
            if random.random() < 0.94:

                historical_risk = random.uniform(
                    0.0,
                    0.10
                )

            else:

                historical_risk = random.uniform(
                    0.50,
                    0.90
                )


        # ====================================================
        # BRUTE FORCE
        # ====================================================

        elif self.scenario == "BRUTE_FORCE":

            attack_probability = random.uniform(
                0.80,
                1.00
            )

            # Molte richieste fallite
            failure_ratio = random.uniform(
                0.70,
                1.00
            )

            requests_5m = random.uniform(
                1500,
                3500
            )

            privileged = int(
                random.random() < 0.20
            )

            critical_resource = int(
                random.random() < 0.15
            )

            has_certificate = 0

            historical_risk = random.uniform(
                0.80,
                0.95
            )



        # ====================================================
        # BOT
        # ====================================================

        elif self.scenario == "BOT":

            attack_probability = random.uniform(
                0.70,
                0.98
            )

            # BOT:
            # grande volume di traffico,
            # ma non necessariamente tutti fallimenti.
            failure_ratio = random.uniform(
                0.05,
                0.40
            )

            requests_5m = random.uniform(
                2500,
                5000
            )

            privileged = int(
                random.random() < 0.10
            )

            critical_resource = int(
                random.random() < 0.10
            )

            has_certificate = 0

            historical_risk = random.uniform(
                0.60,
                0.95
            )


        # ====================================================
        # PRIVILEGED ATTACK
        # ====================================================

        elif self.scenario == "PRIVILEGED_ATTACK":

            attack_probability = random.uniform(
                0.85,
                1.00
            )

            failure_ratio = random.uniform(
                0.40,
                1.00
            )

            requests_5m = random.uniform(
                1000,
                3500
            )

            # Per definizione dello scenario
            privileged = 1

            critical_resource = int(
                random.random() < 0.70
            )

            has_certificate = 0

            historical_risk = random.uniform(
                0.85,
                0.95
            )


        # ====================================================
        # RESOURCE SCAN
        # ====================================================

        elif self.scenario == "RESOURCE_SCAN":

            attack_probability = random.uniform(
                0.60,
                0.95
            )

            failure_ratio = random.uniform(
                0.15,
                0.65
            )

            requests_5m = random.uniform(
                1200,
                3000
            )

            privileged = int(
                random.random() < 0.20
            )

            # Caratteristica principale dello scenario
            critical_resource = 1

            has_certificate = 0

            historical_risk = random.uniform(
                0.65,
                0.95
            )


        else:

            raise ValueError(
                f"Scenario non riconosciuto: "
                f"{self.scenario}"
            )


        # ====================================================
        # RAW STATE
        # ====================================================

        state = np.array(
            [
                attack_probability,   # 0
                failure_ratio,        # 1
                requests_5m,          # 2 - RAW
                hour,                 # 3
                is_night,             # 4
                privileged,           # 5
                critical_resource,    # 6
                has_certificate,      # 7
                historical_risk       # 8
            ],
            dtype=np.float32
        )

        return state




    def _get_normalized_state(self):
        """
        Restituisce lo stato normalizzato per il DQN.
        """

        s = self.state.copy()

        s[2] /= 4486.0     # requests_5m
        s[3] /= 23.0        # hour

        return np.clip(s,0.0,1.0).astype(np.float32)
    





    def _evolve_state(self, action):

        if self.scenario == "NORMAL":

            self._evolve_normal(action)

        elif self.scenario == "BRUTE_FORCE":

            self._evolve_brute_force(action)

        elif self.scenario == "BOT":

            self._evolve_bot(action)

        elif self.scenario == "RESOURCE_SCAN":

            self._evolve_resource_scan(action)

        elif self.scenario == "PRIVILEGED_ATTACK":

            self._evolve_privileged_attack(action)

        else:

            self._evolve_normal(action)

        self._update_historical_risk()
        self._clip_state()


    def _clip_state(self):
        # attack probability
        self.state[0] = np.clip(self.state[0], 0, 1)
        # failure ratio
        self.state[1] = np.clip(self.state[1], 0, 1)
        # requests_5m raw
        self.state[2] = np.clip(self.state[2], 0.0, 4486.0 * 2)


        self.state[4] = int(
            self.state[3] < 6 or self.state[3] >= 22
        )
        # historical risk
        self.state[8] = np.clip(self.state[8], 0.0, 1.0)
        
    


    def _evolve_normal(self, action):

        s = self.state

        if action == 0:          # ALLOW

            s[0] -= 0.01

        elif action == 1:        # MONITOR

            s[0] -= 0.005

        elif action == 2:        # MFA

            s[0] -= 0.01

        elif action in [3, 4, 5]:

            pass
            

        





    def _evolve_brute_force(self, action):

        s = self.state

        if action == 0:          # ALLOW

            s[0] += 0.10
            s[1] += 0.10
            s[2] += 400

        elif action == 1:        # MONITOR

            s[0] += 0.03
            s[1] += 0.03
            s[2] += 150

        elif action == 2:        # MFA

            s[0] -= 0.20
            s[1] -= 0.15
            s[2] -= 300

        elif action == 3:        # BLOCK_SESSION

            s[0] = 0.15
            s[1] *= 0.25
            s[2] *= 0.30

        elif action == 4:        # BLOCK_IP

            s[0] = 0.05
            s[1] *= 0.10
            s[2] *= 0.10

        elif action == 5:        # QUARANTINE

            s[0] = 0.02
            s[1] *= 0.05
            s[2] *= 0.05

        
    


    def _update_historical_risk(self):

        s = self.state

        risk = (
            s[8]
            *
            0.95
        )

        failure_ratio = s[1]

        if failure_ratio >= 0.75:

            risk += 0.15

        elif failure_ratio >= 0.25:

            risk += 0.08

        elif failure_ratio >= 0.05:

            risk += 0.03


        s[8] = np.clip(
            risk,
            0.0,
            1.0
        )



    def _evolve_bot(self, action):

        s = self.state

        if action == 0:          # ALLOW

            s[0] += 0.10
            s[1] += 0.02
            s[2] += 500

        elif action == 1:        # MONITOR

            s[0] += 0.03
            s[1] += 0.01
            s[2] += 200

        elif action == 2:        # MFA

            s[0] -= 0.10
            s[1] -= 0.03
            s[2] -= 300

        elif action == 3:        # BLOCK_SESSION

            s[0] = 0.15
            s[1] *= 0.50
            s[2] *= 0.30

        elif action == 4:        # BLOCK_IP

            s[0] = 0.05
            s[1] *= 0.20
            s[2] *= 0.10

        elif action == 5:        # QUARANTINE

            s[0] = 0.02
            s[1] *= 0.10
            s[2] *= 0.05

        






    def _evolve_resource_scan(self, action):

        s = self.state

        if action == 0:          # ALLOW

            s[0] += 0.10
            s[1] += 0.05
            s[2] += 350

        elif action == 1:        # MONITOR

            s[0] += 0.03
            s[1] += 0.02
            s[2] += 120

        elif action == 2:        # MFA

            s[0] -= 0.15
            s[1] -= 0.08
            s[2] -= 200

        elif action == 3:        # BLOCK_SESSION

            s[0] = 0.12
            s[1] *= 0.40
            s[2] *= 0.35

        elif action == 4:        # BLOCK_IP

            s[0] = 0.05
            s[1] *= 0.15
            s[2] *= 0.10

        elif action == 5:        # QUARANTINE

            s[0] = 0.02
            s[1] *= 0.10
            s[2] *= 0.05

        






    def _evolve_privileged_attack(self, action):

        s = self.state

        if action == 0:          # ALLOW

            s[0] += 0.10
            s[1] += 0.10
            s[2] += 400

        elif action == 1:        # MONITOR

            s[0] += 0.04
            s[1] += 0.03
            s[2] += 150

        elif action == 2:        # MFA

            s[0] -= 0.20
            s[1] -= 0.10
            s[2] -= 250

        elif action == 3:        # BLOCK_SESSION

            s[0] = 0.15
            s[1] *= 0.30
            s[2] *= 0.30

        elif action == 4:        # BLOCK_IP

            s[0] = 0.05
            s[1] *= 0.10
            s[2] *= 0.10

        elif action == 5:        # QUARANTINE

            s[0] = 0.02
            s[1] *= 0.05
            s[2] *= 0.05

        









    def step(
            self,
            action
    ):


        old_state = self.state.copy()



        self._evolve_state(action)



        reward = RewardCalculator.compute(
            old_state,
            action,
            self.state,
            self.scenario
        )



        self.current_step += 1



        done = False

        if self.current_step >= self.MAX_STEPS:

            done = True

        elif self.scenario != "NORMAL" and self.state[0] <= 0.05:

            # Attacco neutralizzato
            done = True

        elif self.scenario == "NORMAL" and self.current_step >= 3:

            # Episodi normali più brevi
            done = True



        info = {
            "scenario": self.scenario,
            "action": self.ACTIONS[action],
            "step": self.current_step,
            "is_attack": self.scenario != "NORMAL"
        }



        return (
            self._get_normalized_state(),
            reward,
            done,
            False,
            info
        )