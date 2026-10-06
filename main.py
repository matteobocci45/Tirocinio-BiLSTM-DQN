# from preprocessing.correlation import correlate_logs
# from preprocessing.create_logs import create_logs
# from preprocessing.encoders import fit_encoders
# from detection.feature_engineering import feature_engineering

# from detection.train_lstm import train_lstm

# import json

# with open("logsprova.json", "r", encoding="utf-8") as f:
#     raw_logs = json.load(f)

# print("Raw events:", len(raw_logs))

# grouped = correlate_logs(raw_logs)

# print("Sessions:", len(grouped))

# flat_logs = create_logs(grouped)

# print("Flat logs:", len(flat_logs))

# encoders = fit_encoders(flat_logs)

# features = feature_engineering(
#     flat_logs,
#     encoders
# )

# print("Feature events:", len(features))

# model = train_lstm(features)



import json
import os
import pickle
from preprocessing.correlation import correlate_logs
from preprocessing.create_logs import create_logs
from preprocessing.encoders import fit_encoders

from detection.feature_engineering import feature_engineering
from detection.train_lstm import train_lstm


def load_logs(file_path):
    """Carica gli eventi raw dal file JSON."""

    with open(
        file_path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main():
    file_path = "logsprova.json"

    # ======================================================
    # 1. CARICAMENTO
    # ======================================================

    print("\nCaricamento log...")

    raw_events = load_logs(file_path)

    print("Eventi raw:", len(raw_events))

    if not raw_events:
        raise RuntimeError(
            "Il file non contiene eventi."
        )

    # ======================================================
    # 2. CORRELAZIONE
    # ======================================================

    print("\nCorrelazione log...")

    grouped_logs = correlate_logs(raw_events)

    print("Sessioni correlate:", len(grouped_logs))

    if not grouped_logs:
        raise RuntimeError(
            "La correlazione non ha prodotto sessioni."
        )

    # ======================================================
    # 3. CREAZIONE LOG FLAT
    # ======================================================

    print("\nCreazione log flat...")

    flat_logs = create_logs(grouped_logs)

    print("Eventi flat:", len(flat_logs))

    if not flat_logs:
        raise RuntimeError(
            "create_logs() non ha prodotto eventi."
        )

    # ======================================================
    # 4. ENCODER
    # ======================================================

    print("\nFit degli encoder...")

    encoders = fit_encoders(flat_logs)
    os.makedirs("models", exist_ok=True)

    with open(
        "models/encoders.pkl",
        "wb",
    ) as file:
        pickle.dump(encoders, file)

    print("Encoder salvati in models/encoders.pkl")

    # ======================================================
    # 5. FEATURE ENGINEERING
    # ======================================================

    print("\nFeature engineering...")

    enriched_events = feature_engineering(
        flat_logs,
        encoders,
    )

    print(
        "Eventi con feature:",
        len(enriched_events),
    )

    if not enriched_events:
        raise RuntimeError(
            "Il feature engineering non ha prodotto eventi."
        )

    print("\nPrimo evento arricchito:")
    print(enriched_events[0])

    # ======================================================
    # 6. TRAINING
    # ======================================================

    print("\nAvvio training BiLSTM...")

    model, history = train_lstm(
        enriched_events,
        epochs=50,
        batch_size=32,
    )

    print("\nTraining completato.")
    print("Modello disponibile in models/lstm.keras")
    print("Scaler disponibile in models/scaler.pkl")


if __name__ == "__main__":
    main()