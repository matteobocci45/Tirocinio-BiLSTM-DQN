from collections import defaultdict, deque
from datetime import datetime, timedelta
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


DEFAULT_WINDOW_SECONDS = 300
DEFAULT_MIN_EVENTS = 5
DEFAULT_MAX_EVENTS = 50
DEFAULT_STRIDE = 5


def build_sequences(
    enriched_events,
    window_seconds=DEFAULT_WINDOW_SECONDS,
    min_events=DEFAULT_MIN_EVENTS,
    max_events=DEFAULT_MAX_EVENTS,
    stride=DEFAULT_STRIDE,
):
    """
    Costruisce sequenze temporali per coppia utente/IP.

    La sequenza include anche l'evento corrente. La label corrisponde
    all'ultimo evento della sequenza.

    Questo implementa una classificazione della sequenza corrente,
    non una previsione del prossimo evento.
    """

    sequences = []
    labels = []
    contexts = []

    entity_events = defaultdict(list)

    for event in enriched_events:
        context = event.get("context", {})

        user = event.get("user") or "ANONYMOUS"
        ip = context.get("ip") or "UNKNOWN_IP"

        entity_key = (user, ip)
        entity_events[entity_key].append(event)

    for entity_key, events in entity_events.items():
        events = sorted(events, key=lambda item: item["time"])

        window = deque()

        for index, event in enumerate(events):
            current_time = event["time"]

            while (
                window
                and (current_time - window[0]["time"]).total_seconds()
                > window_seconds
            ):
                window.popleft()

            window.append(event)

            if len(window) < min_events:
                continue

            if index % stride != 0:
                continue

            sequence_events = list(window)[-max_events:]

            # La LSTM vede anche l'evento corrente.
            sequence = [
                sequence_event["features"]
                for sequence_event in sequence_events
            ]

            target_event = sequence_events[-1]
            label = int(target_event["label"])

            context = dict(target_event.get("context", {}))
            context["sequence_start"] = sequence_events[0]["time"]
            context["sequence_end"] = target_event["time"]
            context["sequence_length"] = len(sequence_events)

            sequences.append(sequence)
            labels.append(label)
            contexts.append(context)

    return (
        sequences,
        np.asarray(labels, dtype=np.int32),
        contexts,
    )


def pad_sequences_lstm(
    sequences,
    max_length=DEFAULT_MAX_EVENTS,
):
    """
    Applica padding posteriore e tronca gli eventi più vecchi quando
    una sequenza supera max_length.
    """

    if not sequences:
        raise ValueError("Non sono presenti sequenze da sottoporre a padding.")

    return pad_sequences(
        sequences,
        maxlen=max_length,
        padding="post",
        truncating="pre",
        value=0.0,
        dtype="float32",
    )


def sequence_statistics(sequences, labels):
    if len(sequences) == 0:
        print("\nNessuna sequenza generata.")
        return

    labels = np.asarray(labels, dtype=np.int32)
    lengths = np.asarray([len(sequence) for sequence in sequences])

    class_counts = np.bincount(labels, minlength=2)

    print("\n===== DATASET STATISTICS =====")
    print("Sequenze:", len(sequences))
    print("Sequenze normali:", int(class_counts[0]))
    print("Sequenze di attacco:", int(class_counts[1]))
    print("Percentuale attacchi:", float(labels.mean()))
    print("Lunghezza media:", float(lengths.mean()))
    print("Lunghezza minima:", int(lengths.min()))
    print("Lunghezza massima:", int(lengths.max()))