import pickle
import numpy as np


from tensorflow.keras.models import load_model

from detection.sequence_builder import (
    pad_sequences_lstm
)



# ==========================================================
# PATH
# ==========================================================


# MODEL_PATH = "models/lstm.keras"

# SCALER_PATH = "models/scaler.pkl"
import os


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "lstm.keras"
)


SCALER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scaler.pkl"
)



# ==========================================================
# LOAD MODEL
# ==========================================================


def load_lstm():

    model = load_model(
        MODEL_PATH
    )


    with open(
        SCALER_PATH,
        "rb"
    ) as f:

        scaler = pickle.load(f)


    print("Tipo scaler caricato:", type(scaler))

    if isinstance(scaler, dict):
        print("Chiavi scaler:", scaler.keys())


    return (
        model,
        scaler
    )



# ==========================================================
# SCALE PADDED SEQUENCES
# ==========================================================

def scale_padded_sequences(
        X,
        scaler_data
):
    """
    Applica lo stesso scaling utilizzato durante
    il training della BiLSTM.

    X:
        shape (samples, timesteps, features)

    IMPORTANTE:
        il padding è già stato effettuato.
        Lo scaling viene applicato DOPO il padding,
        esattamente come nel training.
    """

    X = np.asarray(
        X,
        dtype=np.float32
    ).copy()

    if X.ndim != 3:
        raise ValueError(
            "X deve avere forma "
            "(samples, timesteps, features). "
            f"Shape ricevuta: {X.shape}"
        )

    # ------------------------------------------------------
    # Lo scaler salvato durante il training è un dizionario
    # ------------------------------------------------------

    if not isinstance(scaler_data, dict):
        raise TypeError(
            "Formato scaler non supportato: "
            f"{type(scaler_data)}"
        )

    scaler = scaler_data.get("scaler")

    if scaler is None:
        raise KeyError(
            "Il dizionario scaler.pkl non contiene "
            "la chiave 'scaler'."
        )

    numeric_indices = scaler_data.get(
        "numeric_indices"
    )

    if numeric_indices is None:
        raise KeyError(
            "Il dizionario scaler.pkl non contiene "
            "'numeric_indices'."
        )

    numeric_indices = np.asarray(
        numeric_indices,
        dtype=np.int64
    )

    samples, timesteps, n_features = X.shape

    if np.any(
        numeric_indices < 0
    ) or np.any(
        numeric_indices >= n_features
    ):
        raise ValueError(
            "Gli indici numerici non sono compatibili "
            f"con {n_features} feature."
        )

    # ------------------------------------------------------
    # IDENTICO AL TRAINING
    #
    # Nel training:
    #
    # train_flat = X_train.reshape(-1, n_features)
    # test_flat  = X_test.reshape(-1, n_features)
    #
    # test_scaled[:, numeric_indices] =
    #       scaler.transform(
    #           test_flat[:, numeric_indices]
    #       )
    # ------------------------------------------------------

    X_flat = X.reshape(
        -1,
        n_features
    )

    X_flat[:, numeric_indices] = scaler.transform(
        X_flat[:, numeric_indices]
    )

    X_scaled = X_flat.reshape(
        samples,
        timesteps,
        n_features
    )

    return X_scaled.astype(
        np.float32
    )


# ==========================================================
# SCALE SINGLE SEQUENCE
# ==========================================================


def scale_sequence(
        sequence,
        scaler_data
):
    """
    Normalizza una singola sequenza usando lo stesso
    scaler e gli stessi indici numerici del training.

    Le feature categoriche non vengono scalate.
    """

    X = np.asarray(
        sequence,
        dtype=np.float32
    )

    if X.ndim != 2:
        raise ValueError(
            "La sequenza deve avere forma "
            "(numero_eventi, numero_feature). "
            f"Shape ricevuta: {X.shape}"
        )

    X_scaled = X.copy()

    # Caso 1:
    # scaler.pkl contiene direttamente uno StandardScaler
    if hasattr(scaler_data, "transform"):

        expected_features = getattr(
            scaler_data,
            "n_features_in_",
            None
        )

        if (
            expected_features is not None
            and expected_features != X.shape[1]
        ):
            raise ValueError(
                "Lo scaler è stato addestrato su "
                f"{expected_features} feature, ma la sequenza "
                f"ne contiene {X.shape[1]}."
            )

        return scaler_data.transform(
            X_scaled
        ).astype(np.float32)

    # Caso 2:
    # scaler.pkl contiene un dizionario
    if not isinstance(scaler_data, dict):
        raise TypeError(
            "Formato scaler non supportato: "
            f"{type(scaler_data)}"
        )

    scaler = scaler_data.get("scaler")

    if scaler is None:
        raise KeyError(
            "Il dizionario scaler.pkl non contiene "
            "la chiave 'scaler'. "
            f"Chiavi disponibili: {list(scaler_data.keys())}"
        )

    numeric_indices = (
        scaler_data.get("numeric_indices")
        or scaler_data.get("numerical_indices")
        or scaler_data.get("numeric_feature_indices")
    )

    if numeric_indices is None:
        raise KeyError(
            "Non sono stati trovati gli indici delle feature "
            "numeriche nel dizionario scaler.pkl. "
            f"Chiavi disponibili: {list(scaler_data.keys())}"
        )

    numeric_indices = np.asarray(
        numeric_indices,
        dtype=np.int64
    )

    if np.any(numeric_indices < 0) or np.any(
        numeric_indices >= X.shape[1]
    ):
        raise ValueError(
            "Gli indici numerici salvati non sono compatibili "
            f"con una sequenza di {X.shape[1]} feature. "
            f"Indici ricevuti: {numeric_indices.tolist()}"
        )

    X_scaled[:, numeric_indices] = scaler.transform(
        X[:, numeric_indices]
    )

    return X_scaled.astype(np.float32)





# ==========================================================
# PREDICTION
# ==========================================================


def predict_attack(
        sequence,
        model=None,
        scaler=None
):
    """
    Restituisce la probabilità di attacco
    per una singola sequenza.

    Pipeline:
        sequence
        -> padding
        -> scaling
        -> BiLSTM
        -> probability
    """

    if model is None or scaler is None:
        model, scaler = load_lstm()

    # Importante:
    # il padding viene effettuato PRIMA dello scaling,
    # esattamente come durante il training della BiLSTM.
    #
    # In questo modo la distribuzione dell'input
    # fornita alla BiLSTM è identica a quella utilizzata
    # durante l'addestramento/evaluation.

    # --------------------------------------
    # 1. PADDING
    # --------------------------------------

    X = pad_sequences_lstm(
        [sequence]
    )

    
    # --------------------------------------
    # 2. SCALING
    # --------------------------------------

    X = scale_padded_sequences(
        X,
        scaler
    )

    # X shape:
    #
    # (1, 50, 15)

    # --------------------------------------
    # 3. Prediction
    # --------------------------------------

    probability = model.predict(
        X,
        verbose=0
    )[0][0]

    return float(
        probability
    )




# ==========================================================
# RL STATE GENERATION
# ==========================================================


def build_detection_output(
        probability,
        context
):


    """
    Crea il dato passato
    al Reinforcement Learning.

    """



    state={

        # output LSTM

        "attack_probability":
            probability,


        # contesto utente

        "user":
            context.get(
                "user",
                "unknown"
            ),


        "ip":
            context.get(
                "ip",
                None
            ),



        # sicurezza

        "privileged":
            context.get(
                "privileged",
                0
            ),


        "critical_resource":
            context.get(
                "critical_resource",
                0
            ),


        "failed_requests":
            context.get(
                "failed_requests",
                0
            ),


        "requests_5m":
            context.get(
                "requests_5m",
                0
            ),


        "hour": context.get(
            "hour",
            0
        ),

        "has_certificate": context.get(
            "has_certificate",
            0
        ),

        "user_risk": context.get(
            "user_risk",
            0
        ),


        "is_night":
            context.get(
                "is_night",
                0
            )

    }


    return state




# ==========================================================
# DETECTION -> RL STATE
# ==========================================================


def generate_rl_state(
        sequence,
        context,
        model=None,
        scaler=None
):
    """
    Pipeline completa:

    Sequence
        |
        v
    BiLSTM
        |
        v
    Detection output
        |
        v
    StateBuilder
        |
        v
    RL state

    """



    from rl.state_builder import StateBuilder


    probability = predict_attack(
        sequence,
        model,
        scaler
    )



    detection_output = build_detection_output(
        probability,
        context
    )


    rl_state = StateBuilder.build(
        detection_output
    )


    return (
        rl_state,
        detection_output
    )






# ==========================================================
# TEST
# ==========================================================


if __name__=="__main__":
    model, scaler = load_lstm()
    print(
        "LSTM loaded correctly"
    )