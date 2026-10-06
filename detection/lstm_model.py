import tensorflow as tf
import os
import random
import numpy as np

SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.config.experimental.enable_op_determinism()

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout,
    Bidirectional,
    LayerNormalization,
    Input
)


from tensorflow.keras.optimizers import Adam



# ============================================================
# BUILD MODEL
# ============================================================


def build_lstm_classifier(
        sequence_length,
        feature_number
):


    model = Sequential()



    # --------------------------------------------------------
    # INPUT
    # --------------------------------------------------------


    model.add(
        Input(
            shape=(
                sequence_length,
                feature_number
            )
        )
    )



    # --------------------------------------------------------
    # FIRST BiLSTM
    # --------------------------------------------------------


    model.add(

        Bidirectional(

            LSTM(

                128,

                return_sequences=True,

                dropout=0.2,

                recurrent_dropout=0.1

            )

        )

    )



    model.add(
        LayerNormalization()
    )



    model.add(
        Dropout(0.3)
    )



    # --------------------------------------------------------
    # SECOND BiLSTM
    # --------------------------------------------------------


    model.add(

        Bidirectional(

            LSTM(

                64,

                return_sequences=False,

                dropout=0.2,

                recurrent_dropout=0.1

            )

        )

    )



    model.add(
        LayerNormalization()
    )



    model.add(
        Dropout(0.3)
    )



    # --------------------------------------------------------
    # CLASSIFIER
    # --------------------------------------------------------


    model.add(

        Dense(

            64,

            activation="relu"

        )

    )



    model.add(

        Dropout(0.25)

    )



    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    model.add(

        Dense(

            1,

            activation="sigmoid",

            name="attack_probability"

        )

    )



    # --------------------------------------------------------
    # COMPILE
    # --------------------------------------------------------


    model.compile(

        optimizer=Adam(

            learning_rate=0.0005

        ),

        loss="binary_crossentropy",

        metrics=[

            "accuracy",

            tf.keras.metrics.Precision(
                name="precision"
            ),

            tf.keras.metrics.Recall(
                name="recall"
            ),

            tf.keras.metrics.AUC(
                name="auc"
            )

        ]

    )



    return model




# ============================================================
# MODEL SUMMARY
# ============================================================


def print_model_info(model):


    print("\n========== BiLSTM MODEL ==========")

    model.summary()

    print(
        "=================================\n"
    )