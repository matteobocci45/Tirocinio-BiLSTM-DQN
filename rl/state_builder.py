import numpy as np


class StateBuilder:
    """
    Converte l'output della Detection
    nello stato utilizzato dal DQN.
    """

    FEATURE_ORDER = [
        "attack_probability",
        "failed_requests",
        "requests_5m",
        "hour",
        "is_night",
        "privileged",
        "critical_resource",
        "has_certificate",
        "user_risk"
    ]

    @staticmethod
    def build(detection_output):
        state = []

        for feature in StateBuilder.FEATURE_ORDER:
            state.append(
                float(detection_output.get(feature, 0.0))
            )

        return np.array(
            state,
            dtype=np.float32
        )
    

    @staticmethod
    def validate(
            state
    ):
        """
        Controllo sicurezza.
        Verifica che lo stato sia compatibile
        con il DQN.
        """

        if not isinstance(
            state,
            np.ndarray
        ):

            return False

        if state.shape != (9,):

            return False
        return True