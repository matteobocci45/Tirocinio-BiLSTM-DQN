import numpy as np


class RewardCalculator:
    """
    Obiettivo:

    - premiare la neutralizzazione reale degli attacchi
    - penalizzare azioni invasive non necessarie
    - evitare falsi positivi
    - considerare la severità
    - considerare la riduzione del rischio

    La reward considera:
    - tipo di scenario
    - severità attacco
    - costo operativo dell'azione
    """

    REQUESTS_SCALE = 4486.0

    ACTION_COST = {
        0: 0,      # ALLOW
        1: 1,      # MONITOR
        2: 4,      # MFA
        3: 12,     # BLOCK_SESSION
        4: 20,     # BLOCK_IP
        5: 30      # QUARANTINE
    }

    NORMAL_REWARD = {
        0: 20,       # ALLOW
        1: 8,        # MONITOR
        2: -10,      # MFA
        3: -60,      # BLOCK_SESSION
        4: -90,      # BLOCK_IP
        5: -120      # QUARANTINE
    }


    # ============================================================
    # THREAT SCORE
    # ============================================================

    @staticmethod
    def _threat_score(state):

        """
        Calcola una misura sintetica della minaccia
        utilizzando lo stato RAW dell'ambiente.

        Stato:

        0 attack_probability
        1 failure_ratio
        2 requests_5m
        3 hour
        4 is_night
        5 privileged
        6 critical_resource
        7 has_certificate
        8 historical_risk
        """

        attack_probability = np.clip(state[0],0.0,1.0)

        failure_ratio = np.clip(state[1],0.0,1.0)

        request_intensity = np.clip(state[2]/RewardCalculator.REQUESTS_SCALE,0.0,1.0)

        critical_resource = np.clip(state[6],0.0,1.0)

        historical_risk = np.clip(state[8],0.0,1.0)


        threat = (
            0.35 * attack_probability
            +
            0.25 * failure_ratio
            +
            0.15 * request_intensity
            +
            0.15 * critical_resource
            +
            0.10 * historical_risk
        )


        return float(np.clip(threat,0.0,1.0))


    # ============================================================
    # REWARD
    # ============================================================

    @staticmethod
    def compute(
            old_state,
            action,
            new_state,
            scenario="NORMAL"
    ):

        # ========================================================
        # TRUE NORMAL
        # ========================================================

        if scenario == "NORMAL":

            # Qui conosciamo il ground truth perché siamo
            # nell'ambiente simulato di training.
            #
            # Non premiamo la "riduzione della minaccia",
            # perché non esiste una vera minaccia da eliminare.

            reward = (
                RewardCalculator.NORMAL_REWARD[
                    action
                ]
            )

            return float(
                reward
            )


        # ========================================================
        # ATTACK SCENARIO
        # ========================================================

        old_threat = (
            RewardCalculator._threat_score(
                old_state
            )
        )

        new_threat = (
            RewardCalculator._threat_score(
                new_state
            )
        )


        # Quanto l'azione ha ridotto la minaccia.
        threat_reduction = (
            old_threat
            -
            new_threat
        )


        # ========================================================
        # 1. EFFECTIVENESS
        # ========================================================

        # Una riduzione di 0.50 produce +60 reward.
        #
        # Una crescita della minaccia produce invece
        # automaticamente reward negativo.

        reward = (
            threat_reduction
            *
            120.0
        )


        # ========================================================
        # 2. REMAINING RISK
        # ========================================================

        # Anche dopo l'azione vogliamo che la minaccia
        # residua sia bassa.

        reward -= (
            new_threat
            *
            20.0
        )


        # ========================================================
        # 3. ACTION COST
        # ========================================================

        reward -= (
            RewardCalculator.ACTION_COST[
                action
            ]
        )


        # ========================================================
        # 4. ALLOW DURING REAL ATTACK
        # ========================================================

        if action == 0:

            reward -= 25.0


        # ========================================================
        # 5. ATTACK NEUTRALIZATION BONUS
        # ========================================================

        # Quando la probabilità di attacco viene
        # portata quasi a zero, premiamo la
        # neutralizzazione.

        if new_state[0] <= 0.05:

            reward += 20.0


        # ========================================================
        # 6. EXTRA PENALTY IF THREAT INCREASES
        # ========================================================

        if threat_reduction < 0:

            reward += (threat_reduction*40.0)


        return float(reward)