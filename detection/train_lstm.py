import os
import pickle

import numpy as np


from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from preprocessing.create_logs import create_logs
from preprocessing.encoders import fit_encoders
from detection.feature_engineering import feature_engineering

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)


from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)



from detection.splitter import temporal_train_val_test_split

from detection.sequence_builder import (
    build_sequences,
    pad_sequences_lstm
)


from detection.lstm_model import (
    build_lstm_classifier,
    print_model_info
)



# ==========================================================
# PATH
# ==========================================================


MODEL_DIR = "models"


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)



# ==========================================================
# NORMALIZATION
# ==========================================================


def normalize_train_val_test(
    X_train,
    X_val,
    X_test,
):
    """
    Adatta lo scaler soltanto sulle feature numeriche del training.

    Le prime quattro feature sono categoriche:
    0 event_id
    1 source_id
    2 role_id
    3 resource_id
    """

    samples_train, timesteps_train, n_features = X_train.shape
    samples_val, timesteps_val, val_features = X_val.shape
    samples_test, timesteps_test, test_features = X_test.shape

    if n_features != val_features or n_features != test_features:
        raise ValueError(
            "Train, validation e test hanno un numero diverso di feature."
        )

    train_flat = X_train.reshape(-1, n_features)
    val_flat = X_val.reshape(-1, n_features)
    test_flat = X_test.reshape(-1, n_features)

    categorical_indices = [0, 1, 2, 3]

    numeric_indices = [
        index
        for index in range(n_features)
        if index not in categorical_indices
    ]

    scaler = StandardScaler()

    train_scaled = train_flat.copy()
    val_scaled = val_flat.copy()
    test_scaled = test_flat.copy()

    # Fit esclusivamente sul training e sulle colonne numeriche.
    train_scaled[:, numeric_indices] = scaler.fit_transform(
        train_flat[:, numeric_indices]
    )

    # TRANSFORM SU VALIDATION
    val_scaled[:, numeric_indices] = scaler.transform(
        val_flat[:, numeric_indices]
    )


    test_scaled[:, numeric_indices] = scaler.transform(
        test_flat[:, numeric_indices]
    )

    X_train_scaled = train_scaled.reshape(
        samples_train,
        timesteps_train,
        n_features,
    )

    X_val_scaled = val_scaled.reshape(
        samples_val,
        timesteps_val,
        n_features
    )


    X_test_scaled = test_scaled.reshape(
        samples_test,
        timesteps_test,
        n_features,
    )

    with open(
        os.path.join(MODEL_DIR, "scaler.pkl"),
        "wb",
    ) as file:
        pickle.dump(
            {
                "scaler": scaler,
                "numeric_indices": numeric_indices,
                "categorical_indices": categorical_indices,
            },
            file,
        )

    return X_train_scaled, X_val_scaled, X_test_scaled




# ==========================================================
# TRAIN FUNCTION
# ==========================================================


def train_lstm(
        enriched_events,
        epochs=50,
        batch_size=32
):


    print("\n==============================")
    print(" TEMPORAL SPLIT ")
    print("==============================")



    # ------------------------------------------------------
    # 1) SPLIT TEMPORALE
    # ------------------------------------------------------


    train_events, val_events, test_events = temporal_train_val_test_split(
        enriched_events
    )


    print(
        "Train events:",
        len(train_events)
    )


    print(
        "Validation events:",
        len(val_events)
    )


    print(
        "Test events:",
        len(test_events)
    )

    # ------------ INCOLLA QUESTO BLOCCO QUI ------------
    print("\n=== FINESTRE TEMPORALI DEGLI SPLIT ===")
    print(f"TRAIN      : {train_events[0]['time']} ---> {train_events[-1]['time']}")
    print(f"VALIDATION : {val_events[0]['time']} ---> {val_events[-1]['time']}")
    print(f"TEST       : {test_events[0]['time']} ---> {test_events[-1]['time']}")
    print("=======================================\n")
    # ---------------------------------------------------



    # ------------------------------------------------------
    # 2) BUILD SEQUENCES
    # ------------------------------------------------------


    print("\nBuilding training sequences...")


    X_train_seq, y_train, _ = build_sequences(
        train_events
    )


    print(
        "Building validation sequences..."
    )

    X_val_seq, y_val, _ = build_sequences(
        val_events
    )


    print(
        "Building testing sequences..."
    )


    X_test_seq, y_test, _ = build_sequences(
        test_events
    )



    print(
        "\nTrain sequences:",
        len(X_train_seq)
    )


    print(
        "Validation sequences:",
        len(X_val_seq)
    )


    print(
        "Test sequences:",
        len(X_test_seq)
    )



    if len(X_train_seq)==0:

        raise Exception(
            "Nessuna sequenza TRAIN generata"
        )


    if len(X_val_seq) == 0:
        raise Exception(
            "Nessuna sequenza VALIDATION generata"
        )

    
    if len(X_test_seq)==0:

        raise Exception(
            "Nessuna sequenza TEST generata"
        )



    # ------------------------------------------------------
    # 3) PADDING
    # ------------------------------------------------------


    X_train = pad_sequences_lstm(
        X_train_seq
    )


    X_val = pad_sequences_lstm(
        X_val_seq
    )


    X_test = pad_sequences_lstm(
        X_test_seq
    )



    print("\nShapes prima scaling:")

    print(
        "X_train:",
        X_train.shape
    )

    print(
        "X_val:",
        X_val.shape
    )

    print(
        "X_test:",
        X_test.shape
    )



    # ------------------------------------------------------
    # 4) NORMALIZATION
    # ------------------------------------------------------


    X_train, X_val, X_test = normalize_train_val_test(
        X_train, X_val,
        X_test
    )



    print("\nScaling completato")



    # ------------------------------------------------------
    # 5) CLASS WEIGHTS
    # ------------------------------------------------------


    weights = class_weight.compute_class_weight(

        class_weight="balanced",

        classes=np.unique(y_train),

        y=y_train

    )



    class_weights = {

        i:w

        for i,w in enumerate(weights)

    }



    print(
        "\nClass weights:"
    )

    print(
        class_weights
    )



    # ------------------------------------------------------
    # 6) MODEL
    # ------------------------------------------------------


    model = build_lstm_classifier(

        sequence_length=X_train.shape[1],

        feature_number=X_train.shape[2]

    )



    print_model_info(
        model
    )



    # ------------------------------------------------------
    # CALLBACKS
    # ------------------------------------------------------


    callbacks=[


        EarlyStopping(

            monitor="val_auc",

            patience=8,

            mode="max",

            restore_best_weights=True

        ),



        ReduceLROnPlateau(

            monitor="val_loss",

            factor=0.5,

            patience=4,

            verbose=1

        ),



        ModelCheckpoint(

            filepath=
            f"{MODEL_DIR}/lstm.keras",

            monitor="val_auc",

            mode="max",

            save_best_only=True,

            verbose=1

        )

    ]



    # ------------------------------------------------------
    # 7) TRAINING
    # ------------------------------------------------------


    print("\n==============================")
    print(" TRAINING BiLSTM ")
    print("==============================")



    history = model.fit(

        X_train,

        y_train,

        validation_data=(

            X_val,

            y_val

        ),

        epochs=epochs,

        batch_size=batch_size,

        class_weight=class_weights,

        callbacks=callbacks,

        verbose=1

    )



    # ------------------------------------------------------
    # 8) EVALUATION
    # ------------------------------------------------------


    print("\n==============================")
    print(" EVALUATION ")
    print("==============================")



    results = model.evaluate(

        X_test,

        y_test,

        verbose=0

    )



    print(
        dict(
            zip(
                model.metrics_names,
                results
            )
        )
    )



    # probabilità

    y_probability = model.predict(
        X_test
    )



    y_probability = y_probability.flatten()



    y_prediction = (

        y_probability >= 0.5

    ).astype(int)



    print(
        "\nClassification report:"
    )


    print(

        classification_report(

            y_test,

            y_prediction

        )

    )



    print(
        "\nConfusion matrix:"
    )


    print(

        confusion_matrix(

            y_test,

            y_prediction

        )

    )



    auc = roc_auc_score(

        y_test,

        y_probability

    )


    print(
        "\nROC-AUC:",
        auc
    )



    return (

        model,

        history

    )

