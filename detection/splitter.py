


def temporal_train_val_test_split(
        events,
        train_ratio=0.8,
        val_ratio=0.1
):
    """
    Suddivide gli eventi in tre insiemi mantenendo l'ordine temporale:

    TRAIN      -> 80%
    VALIDATION -> 10%
    TEST       -> 10%

    La validation viene utilizzata durante l'addestramento
    per monitorare le prestazioni e selezionare il modello.

    Il test viene mantenuto separato e utilizzato soltanto
    nella valutazione finale.
    """

    events = sorted(
        events,
        key=lambda x: x["time"]
    )

    n_events = len(events)

    train_end = int(n_events * train_ratio)
    val_end = int(n_events * (train_ratio + val_ratio))

    train_events = events[:train_end]

    val_events = events[
        train_end:val_end
    ]

    test_events = events[
        val_end:
    ]

    return (
        train_events,
        val_events,
        test_events
    )