import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from detection.sequence_builder import (
    build_sequences
)

from detection.splitter import (
    temporal_train_val_test_split
)

from detection.inference import (
    generate_rl_state,
    load_lstm,
)

from preprocessing.create_logs import create_logs
from preprocessing.encoders import load_encoders
from preprocessing.correlation import correlate_logs
from detection.feature_engineering import feature_engineering

from rl.dqn_agent import DQNAgent

MAX_SEQUENCES_TO_PRINT = 20
DETECTION_THRESHOLD = 0.50

# =====================================================
# EVALUATION MODE
# =====================================================

# True:
# valuta SOLO la parte temporale TEST,
# cioè gli eventi che la BiLSTM non ha usato
# per aggiornare i pesi durante il training.
#
# False:
# valuta tutto il dataset.
EVALUATE_TEST_ONLY = True


# None = processa tutte le sequenze.
#
# Durante il debug puoi mettere per esempio 500.
MAX_SEQUENCES_TO_PROCESS = None

# =====================================================
# PATH
# =====================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


LOG_PATH = os.path.join(
    BASE_DIR,
    "logsprova.json",
)

DQN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "dqn.keras",
)

ENCODERS_PATH = os.path.join(
    BASE_DIR,
    "models",
    "encoders.pkl",
)

FIGURES_DIR = os.path.join(
    BASE_DIR,
    "results",
    "figures",
)
# =====================================================
# ACTION MAP
# =====================================================

ACTIONS = {
    0: "ALLOW",
    1: "MONITOR",
    2: "MFA",
    3: "BLOCK_SESSION",
    4: "BLOCK_IP",
    5: "QUARANTINE",
}



# =====================================================
# LOAD LOGS
# =====================================================

def load_raw_logs(path):
    """
    Legge logsprova.json.

    Supporta:
    - lista JSON
    - JSON Lines
    - oggetto con chiave "logs"
    - oggetto con chiave "events"
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File non trovato: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        content = file.read().strip()

    if not content:
        raise ValueError(
            "logsprova.json è vuoto."
        )

    # -------------------------------------------------
    # JSON normale
    # -------------------------------------------------

    try:

        data = json.loads(content)

        if isinstance(data, list):
            return data

        if isinstance(data, dict):

            if "logs" in data:
                return data["logs"]

            if "events" in data:
                return data["events"]

            return [data]

    except json.JSONDecodeError:
        pass

    # -------------------------------------------------
    # JSON Lines
    # -------------------------------------------------

    logs = []

    for line_number, line in enumerate(
        content.splitlines(),
        start=1,
    ):

        line = line.strip()

        if not line:
            continue

        try:

            logs.append(
                json.loads(line)
            )

        except json.JSONDecodeError as exc:

            raise ValueError(
                f"JSON non valido alla riga "
                f"{line_number}"
            ) from exc

    return logs







# =====================================================
# CONTEXT FROM SEQUENCE
# =====================================================

def build_context_from_sequence(
        sequence,
        original_context
):
    """
    Costruisce il contesto RL usando la stessa
    sequenza fornita alla BiLSTM.
    """

    sequence = np.asarray(
        sequence,
        dtype=np.float32
    )

    if sequence.ndim != 2:
        raise ValueError(
            f"Sequenza non valida: {sequence.shape}"
        )

    if sequence.shape[1] != 15:
        raise ValueError(
            "La sequenza deve contenere 15 feature. "
            f"Ricevute: {sequence.shape[1]}"
        )

    last_event = sequence[-1]

    # quanti eventi falliti sono presenti
    # nella finestra corrente, però failure ratio, non numero assoluto
    failed_count = int(
        np.sum(
            sequence[:, 9] > 0
        )
    )
    failed_ratio = (
        failed_count / len(sequence)
    )

    context = {

        # metadata già disponibili
        "user":
            original_context.get(
                "user",
                "ANONYMOUS"
            ),

        "ip":
            original_context.get(
                "ip",
                "UNKNOWN_IP"
            ),

        # comportamento della finestra
        "failed_requests":
            failed_ratio,

        "requests_5m":
            float(last_event[10]),

        # tempo
        "hour":
            float(last_event[6]),

        "is_night":
            int(last_event[8]),

        # security context
        "has_certificate":
            int(last_event[11]),

        "privileged":
            int(last_event[12]),

        # "critical_resource":
        #     int(
        #         np.max(
        #             sequence[:, 13]
        #         )
        #     ),

        "critical_resource":
            int(last_event[13]),

        "user_risk":
            float(last_event[14]),
    }

    # metadata sequence_builder
    context["sequence_start"] = (
        original_context.get(
            "sequence_start"
        )
    )

    context["sequence_end"] = (
        original_context.get(
            "sequence_end"
        )
    )

    context["sequence_length"] = (
        original_context.get(
            "sequence_length",
            len(sequence)
        )
    )

    return context






# =====================================================
# DQN STATE NORMALIZATION
# =====================================================

def normalize_rl_state(state):
    """
    Applica la stessa normalizzazione utilizzata
    durante il training del DQN.

    Indici:
        0 attack_probability   0-1
        1 failure_ratio        0-1
        2 requests_5m          
        3 hour                 0-23
        4 is_night             0-1
        5 privileged           0-1
        6 critical_resource    0-1
        7 has_certificate      0-1
        8 historical_risk      0-1
    """

    normalized_state = np.asarray(
        state,
        dtype=np.float32,
    ).copy()

    if normalized_state.shape != (9,):
        raise ValueError(
            "Lo stato RL deve contenere esattamente 9 feature. "
            f"Shape ricevuta: {normalized_state.shape}"
        )

    normalized_state[0] = np.clip(
        normalized_state[0],
        0.0,
        1.0,
    )

    normalized_state[1] = np.clip(
        normalized_state[1],
        0.0,
        1.0,
    )

    normalized_state[2] = np.clip(
        normalized_state[2] / 4486.0,
        0.0,
        1.0,
    )

    normalized_state[3] = np.clip(
        normalized_state[3] / 23.0,
        0.0,
        1.0,
    )

    normalized_state[4:] = np.clip(
        normalized_state[4:],
        0.0,
        1.0,
    )

    return normalized_state












# =====================================================
# PRINT RESULT
# =====================================================

def print_result(
        index,
        sequence,
        true_label,
        context,
        detection_output,
        raw_state,
        normalized_state,
        q_values,
        action_name,
):
    """
    Stampa il risultato di una singola sequenza.
    """

    probability = float(
        detection_output[
            "attack_probability"
        ]
    )

    predicted_label = int(
        probability >= DETECTION_THRESHOLD
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"SEQUENCE {index}"
    )

    print(
        "=" * 70
    )

    print(
        "User:",
        context.get(
            "user",
            "ANONYMOUS",
        )
    )

    print(
        "IP:",
        context.get(
            "ip",
            "UNKNOWN_IP",
        )
    )

    print(
        "Sequence length:",
        context.get(
            "sequence_length",
            len(sequence),
        )
    )

    if context.get(
        "sequence_start"
    ) is not None:

        print(
            "Start:",
            context[
                "sequence_start"
            ]
        )

    if context.get(
        "sequence_end"
    ) is not None:

        print(
            "End:",
            context[
                "sequence_end"
            ]
        )

    # -------------------------------------------------
    # CONTEXT
    # -------------------------------------------------

    print(
        "\n--- CONTEXT ---"
    )

    print(
        "Failed requests   :",
        context[
            "failed_requests"
        ]
    )

    print(
        "Requests 5m       :",
        context[
            "requests_5m"
        ]
    )

    print(
        "Hour              :",
        context[
            "hour"
        ]
    )

    print(
        "Is night          :",
        context[
            "is_night"
        ]
    )

    print(
        "Privileged        :",
        context[
            "privileged"
        ]
    )

    print(
        "Critical resource :",
        context[
            "critical_resource"
        ]
    )

    print(
        "Has certificate   :",
        context[
            "has_certificate"
        ]
    )

    print(
        "User risk         :",
        context[
            "user_risk"
        ]
    )

    # -------------------------------------------------
    # BiLSTM
    # -------------------------------------------------

    print(
        "\n--- BiLSTM ---"
    )

    print(
        "Attack probability:",
        f"{probability:.6f}",
    )

    print(
        "Prediction:",
        "ATTACK"
        if predicted_label == 1
        else "NORMAL",
    )

    print(
        "Dataset label:",
        "ATTACK"
        if int(true_label) == 1
        else "NORMAL",
    )

    # -------------------------------------------------
    # RL
    # -------------------------------------------------

    print(
        "\n--- RL STATE ---"
    )

    print(
        "Raw:"
    )

    print(
        raw_state
    )

    print(
        "Normalized:"
    )

    print(
        normalized_state
    )

    # -------------------------------------------------
    # DQN
    # -------------------------------------------------

    print(
        "\n--- DQN Q-VALUES ---"
    )

    for action_id, q_value in enumerate(
        q_values
    ):

        print(
            f"{ACTIONS[action_id]:<20}: "
            f"{float(q_value):.4f}"
        )

    print(
        "\n>>> DECISIONE DQN:",
        action_name
    )



















def print_state_distribution(
        title,
        states
):
    """
    Analizza la distribuzione degli stati RAW
    realmente forniti al DQN.

    Stato:
        0 attack_probability
        1 failed_requests
        2 requests_5m
        3 hour
        4 is_night
        5 privileged
        6 critical_resource
        7 has_certificate
        8 user_risk
    """

    if not states:

        print(
            f"\n{title}: nessuno stato."
        )

        return


    X = np.asarray(
        states,
        dtype=np.float32
    )


    print(
        "\n"
        + "=" * 70
    )

    print(title)

    print(
        "=" * 70
    )

    print(
        "Numero stati:",
        len(X)
    )


    # =================================================
    # ATTACK PROBABILITY
    # =================================================

    attack_probability = X[:, 0]

    print(
        "\n--- ATTACK PROBABILITY ---"
    )

    print(
        f"Mean   : "
        f"{np.mean(attack_probability):.4f}"
    )

    print(
        f"Median : "
        f"{np.median(attack_probability):.4f}"
    )

    print(
        f"P90    : "
        f"{np.percentile(attack_probability, 90):.4f}"
    )

    print(
        f"P95    : "
        f"{np.percentile(attack_probability, 95):.4f}"
    )

    print(
        f"Max    : "
        f"{np.max(attack_probability):.4f}"
    )


    # =================================================
    # FAILED REQUESTS
    # =================================================

    failed = X[:, 1]

    print(
        "\n--- FAILED REQUESTS ---"
    )

    print(
        f"Mean   : "
        f"{np.mean(failed):.2f}"
    )

    print(
        f"Median : "
        f"{np.median(failed):.2f}"
    )

    print(
        f"P95    : "
        f"{np.percentile(failed, 95):.2f}"
    )

    print(
        f"Max    : "
        f"{np.max(failed):.2f}"
    )


    # =================================================
    # REQUESTS 5 MINUTES
    # =================================================

    requests = X[:, 2]

    print(
        "\n--- REQUESTS 5M ---"
    )

    print(
        f"Mean   : "
        f"{np.mean(requests):.2f}"
    )

    print(
        f"Median : "
        f"{np.median(requests):.2f}"
    )

    print(
        f"P90    : "
        f"{np.percentile(requests, 90):.2f}"
    )

    print(
        f"P95    : "
        f"{np.percentile(requests, 95):.2f}"
    )

    print(
        f"Max    : "
        f"{np.max(requests):.2f}"
    )


    # =================================================
    # BINARY FEATURES
    # =================================================

    is_night = X[:, 4]
    privileged = X[:, 5]
    critical = X[:, 6]
    certificate = X[:, 7]

    print(
        "\n--- BINARY FEATURES ---"
    )

    print(
        f"Is night = 1          : "
        f"{np.mean(is_night) * 100:.2f}%"
    )

    print(
        f"Privileged = 1        : "
        f"{np.mean(privileged) * 100:.2f}%"
    )

    print(
        f"Critical resource = 1 : "
        f"{np.mean(critical) * 100:.2f}%"
    )

    print(
        f"Has certificate = 1   : "
        f"{np.mean(certificate) * 100:.2f}%"
    )


    # =================================================
    # USER RISK
    # =================================================

    risk = X[:, 8]

    print(
        "\n--- USER RISK ---"
    )

    print(
        f"Mean              : "
        f"{np.mean(risk):.4f}"
    )

    print(
        f"Median            : "
        f"{np.median(risk):.4f}"
    )

    print(
        f"P90               : "
        f"{np.percentile(risk, 90):.4f}"
    )

    print(
        f"P95               : "
        f"{np.percentile(risk, 95):.4f}"
    )

    print(
        f"Risk >= 0.50      : "
        f"{np.mean(risk >= 0.50) * 100:.2f}%"
    )

    print(
        f"Risk >= 0.80      : "
        f"{np.mean(risk >= 0.80) * 100:.2f}%"
    )

    print(
        f"Risk >= 0.99      : "
        f"{np.mean(risk >= 0.99) * 100:.2f}%"
    )











def print_action_distribution(
        title,
        action_counts
):

    total = sum(
        action_counts.values()
    )

    print(
        "\n"
        + "=" * 70
    )

    print(title)

    print(
        "=" * 70
    )

    print(
        f"Totale sequenze: {total}"
    )

    print()

    for action_name, count in (
        action_counts.items()
    ):

        percentage = (
            count / total * 100
            if total > 0
            else 0.0
        )

        print(
            f"{action_name:<20}: "
            f"{count:<6} "
            f"({percentage:.2f}%)"
        )









# =====================================================
# FIGURE GENERATION
# =====================================================

def _prepare_figures_dir():
    """
    Crea la cartella in cui salvare le figure della tesi.
    """
    os.makedirs(
        FIGURES_DIR,
        exist_ok=True
    )


def save_confusion_matrix_figure(
        tn,
        fp,
        fn,
        tp
):
    """
    Salva la matrice di confusione della BiLSTM.
    """

    _prepare_figures_dir()

    matrix = np.array([
        [tn, fp],
        [fn, tp],
    ], dtype=np.int64)

    fig, ax = plt.subplots(
        figsize=(7, 6)
    )

    image = ax.imshow(
        matrix
    )

    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        "Numero di sequenze"
    )

    ax.set_xticks(
        [0, 1]
    )

    ax.set_xticklabels([
        "Predetto NORMAL",
        "Predetto ATTACK",
    ])

    ax.set_yticks(
        [0, 1]
    )

    ax.set_yticklabels([
        "Vero NORMAL",
        "Vero ATTACK",
    ])

    ax.set_xlabel(
        "Predizione BiLSTM",
        fontsize=12
    )

    ax.set_ylabel(
        "Etichetta reale",
        fontsize=12
    )

    ax.set_title(
        "Matrice di confusione della BiLSTM",
        fontsize=15,
        pad=12
    )

    # -------------------------------------------------
    # Testo ad alto contrasto rispetto alla cella
    # -------------------------------------------------

    for row in range(matrix.shape[0]):

        for column in range(matrix.shape[1]):

            value = int(
                matrix[row, column]
            )

            # Recupera il colore reale della cella
            rgba = image.cmap(
                image.norm(value)
            )

            red, green, blue = rgba[:3]

            # Luminosità percepita del colore di sfondo
            luminance = (
                0.299 * red
                + 0.587 * green
                + 0.114 * blue
            )

            # Sfondo chiaro -> testo nero
            # Sfondo scuro  -> testo bianco
            text_color = (
                "black"
                if luminance > 0.55
                else "white"
            )

            ax.text(
                column,
                row,
                f"{value}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=17,
                fontweight="bold",
            )

    fig.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "bilstm_confusion_matrix.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Figura salvata:",
        output_path
    )

def save_state_boxplots(
        normal_states,
        attack_states
):
    """
    Salva quattro boxplot separati delle principali
    feature dello stato RL confrontando TRUE NORMAL
    e TRUE ATTACK.

    Gli outlier vengono esclusi soltanto dalla
    visualizzazione (showfliers=False), non dai dati
    né dalle statistiche della pipeline.
    """

    if not normal_states or not attack_states:

        print(
            "Boxplot non generati: "
            "mancano stati NORMAL o ATTACK."
        )

        return

    _prepare_figures_dir()

    normal = np.asarray(
        normal_states,
        dtype=np.float32
    )

    attack = np.asarray(
        attack_states,
        dtype=np.float32
    )

    features = [
        (
            0,
            "Probabilità di attacco",
            "Probabilità di attacco: NORMAL vs ATTACK",
            "attack_probability_boxplot.png",
        ),
        (
            1,
            "Rapporto di fallimenti",
            "Rapporto di fallimenti: NORMAL vs ATTACK",
            "failure_ratio_boxplot.png",
        ),
        (
            2,
            "Richieste negli ultimi 5 minuti",
            "Richieste negli ultimi 5 minuti: NORMAL vs ATTACK",
            "requests_5m_boxplot.png",
        ),
        (
            8,
            "Rischio storico",
            "Rischio storico: NORMAL vs ATTACK",
            "historical_risk_boxplot.png",
        ),
    ]

    for (
        feature_index,
        y_label,
        title,
        filename
    ) in features:

        fig, ax = plt.subplots(
            figsize=(7, 6)
        )

        ax.boxplot(
            [
                normal[:, feature_index],
                attack[:, feature_index],
            ],
            showfliers=False,
        )

        ax.set_xticks(
            [1, 2]
        )

        ax.set_xticklabels([
            "TRUE NORMAL",
            "TRUE ATTACK",
        ])

        ax.set_ylabel(
            y_label,
            fontsize=12
        )

        ax.set_title(
            title,
            fontsize=15,
            pad=12
        )

        ax.grid(
            axis="y",
            alpha=0.3
        )

        fig.tight_layout()

        output_path = os.path.join(
            FIGURES_DIR,
            filename
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight"
        )

        plt.close(fig)

        print(
            "Figura salvata:",
            output_path
        )

def save_dqn_true_label_actions_figure(
        normal_action_counts,
        attack_action_counts
):
    """
    Grafico a barre delle azioni DQN condizionate
    dalla vera classe NORMAL/ATTACK.

    Sopra ogni barra non nulla viene mostrata
    la relativa percentuale.
    """

    _prepare_figures_dir()

    action_names = list(
        ACTIONS.values()
    )

    normal_total = sum(
        normal_action_counts.values()
    )

    attack_total = sum(
        attack_action_counts.values()
    )

    normal_percentages = [
        (
            normal_action_counts[action]
            / normal_total
            * 100
            if normal_total > 0
            else 0.0
        )
        for action in action_names
    ]

    attack_percentages = [
        (
            attack_action_counts[action]
            / attack_total
            * 100
            if attack_total > 0
            else 0.0
        )
        for action in action_names
    ]

    x = np.arange(
        len(action_names)
    )

    width = 0.38

    fig, ax = plt.subplots(
        figsize=(11, 6)
    )

    normal_bars = ax.bar(
        x - width / 2,
        normal_percentages,
        width,
        label="TRUE NORMAL",
    )

    attack_bars = ax.bar(
        x + width / 2,
        attack_percentages,
        width,
        label="TRUE ATTACK",
    )

    ax.set_xticks(
        x
    )

    ax.set_xticklabels(
        action_names,
        rotation=20,
        ha="right",
    )

    ax.set_ylabel(
        "Percentuale di sequenze (%)",
        fontsize=12
    )

    ax.set_title(
        "Distribuzione delle azioni DQN per classe reale",
        fontsize=15,
        pad=12
    )

    ax.set_ylim(
        0,
        106
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    ax.legend()

    # -------------------------------------------------
    # Percentuali sopra le barre
    # -------------------------------------------------

    for bars in (
        normal_bars,
        attack_bars
    ):

        for bar in bars:

            height = float(
                bar.get_height()
            )

            if height <= 0:
                continue

            # Per valori molto piccoli mostriamo 2 decimali,
            # altrimenti 1 decimale.
            label = (
                f"{height:.2f}%"
                if height < 1.0
                else f"{height:.1f}%"
            )

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,
                height + 0.7,
                label,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    fig.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        "dqn_actions_true_normal_vs_attack.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "Figura salvata:",
        output_path
    )

def save_all_thesis_figures(
        normal_states,
        attack_states,
        normal_action_counts,
        attack_action_counts,
        tn,
        fp,
        fn,
        tp
):
    """
    Genera tutte le figure principali
    dell'analisi sperimentale.
    """

    print(
        "\n"
        + "=" * 70
    )

    print(
        "GENERAZIONE FIGURE TESI"
    )

    print(
        "=" * 70
    )

    save_confusion_matrix_figure(
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )

    save_state_boxplots(
        normal_states=normal_states,
        attack_states=attack_states,
    )

    save_dqn_true_label_actions_figure(
        normal_action_counts=normal_action_counts,
        attack_action_counts=attack_action_counts,
    )


# =====================================================
# PIPELINE
# =====================================================

def run_pipeline():

    print(
        "=" * 70
    )

    print(
        "REAL CYBER DEFENSE PIPELINE"
    )

    print(
        "=" * 70
    )

    # =================================================
    # 1. READ LOGSPROVA.JSON
    # =================================================

    print(
        "\n[1] Lettura logsprova.json..."
    )

    raw_logs = load_raw_logs(
        LOG_PATH
    )

    print(
        "Log caricati:",
        len(raw_logs)
    )

    if not raw_logs:

        raise RuntimeError(
            "Nessun evento trovato "
            "in logsprova.json."
        )

    # =================================================
    # 2. CORRELATION
    # =================================================

    print(
        "\n[2] Correlazione log..."
    )

    correlated_events = correlate_logs(
        raw_logs
    )

    print(
        "Correlazione completata."
    )

# =================================================
# 2.5. CREATE LOGS
# =================================================

    print("\n[3] Creazione log flat...")

    flat_logs = create_logs(
        correlated_events
    )

    print(
        "Eventi flat:",
        len(flat_logs)
    )

    if not flat_logs:
        raise RuntimeError(
            "create_logs() non ha prodotto eventi."
        )


# =================================================
# 3. LOAD ENCODERS
# =================================================
    print(
        "\n[3] Caricamento encoder..."
    )

    encoders = load_encoders(
        ENCODERS_PATH
    )

    print(
        "Encoder caricati correttamente."
    )

    # =================================================
    # 3. FEATURE ENGINEERING
    # =================================================

    print(
        "\n[3] Feature engineering..."
    )

    enriched_events = feature_engineering(
        flat_logs,
        encoders
    )

    print(
        "Eventi arricchiti:",
        len(enriched_events)
    )

    # =================================================
    # TEMPORAL HOLDOUT
    # =================================================

    if EVALUATE_TEST_ONLY:

        print(
            "\n[6] Temporal train/test split..."
        )

        train_events, val_events, test_events = (
            temporal_train_val_test_split(
                enriched_events
            )
        )

        print(
            "Eventi TRAIN:",
            len(train_events)
        )

        print(
            "Eventi Validation:",
            len(val_events)
        )
    
        print(
            "Eventi TEST :",
            len(test_events)
        )

        # IMPORTANTISSIMO:
        #
        # Da questo momento la pipeline utilizza
        # SOLO gli eventi TEST.
        evaluation_events = test_events

    else:

        print(
            "\nATTENZIONE: evaluation "
            "su tutto il dataset."
        )

        evaluation_events = enriched_events
    
    if not enriched_events:

        raise RuntimeError(
            "Il feature engineering "
            "non ha prodotto eventi."
        )

    # =================================================
    # 4. BUILD REAL SEQUENCES
    # =================================================

    print(
        "\n[4] Costruzione sequenze..."
    )

    (
        sequences,
        labels,
        contexts
    ) = build_sequences(
        #enriched_events
        evaluation_events
    )

    print(
        "Sequenze generate:",
        len(sequences)
    )

    if len(sequences) == 0:

        raise RuntimeError(
            "build_sequences non ha "
            "generato nessuna sequenza."
        )

    if not (
        len(sequences)
        == len(labels)
        == len(contexts)
    ):

        raise RuntimeError(
            "Errore: sequences, labels "
            "e contexts hanno lunghezze "
            "diverse."
        )

    # =================================================
    # 5. LOAD BiLSTM
    # =================================================

    print(
        "\n[5] Caricamento BiLSTM..."
    )

    lstm_model, scaler = load_lstm()

    print(
        "BiLSTM e scaler caricati "
        "correttamente."
    )

    # =================================================
    # 6. LOAD DQN
    # =================================================

    print(
        "\n[6] Caricamento DQN..."
    )

    agent = DQNAgent(
        state_size=9,
        action_size=6,
    )

    agent.load(
        DQN_MODEL_PATH
    )

    # Inferenza:
    # niente epsilon-greedy/random.
    agent.epsilon = 0.0

    print(
        "DQN caricato correttamente."
    )

    # =================================================
    # STATISTICHE
    # =================================================

    action_counts = {
        action_name: 0
        for action_name
        in ACTIONS.values()
    }

    probabilities = []

    normal_states = []
    attack_states = []

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # DQN classification metrics
    dqn_y_true = []
    dqn_y_pred = []

    normal_action_counts = {
        action_name: 0
        for action_name in ACTIONS.values()
    }

    attack_action_counts = {
        action_name: 0
        for action_name in ACTIONS.values()
    }

    false_negative_action_counts = {
        action_name: 0
        for action_name in ACTIONS.values()
    }


    def new_action_counter():
        return {
            action_name: 0
            for action_name in ACTIONS.values()
        }


    tn_action_counts = new_action_counter()
    fp_action_counts = new_action_counter()
    fn_action_counts = new_action_counter()
    tp_action_counts = new_action_counter()

    # =================================================
    # 7. REAL PIPELINE
    # =================================================

    processed_sequences = 0

    print(
        "\n[7] Inferenza sulle "
        "sequenze del dataset..."
    )

    for index, (
        sequence,
        true_label,
        original_context
    ) in enumerate(
        zip(
            sequences,
            labels,
            contexts,
        ),
        start=1,
    ):

        if (
            MAX_SEQUENCES_TO_PROCESS is not None
            and index > MAX_SEQUENCES_TO_PROCESS
        ):
            break


        processed_sequences += 1
        # ---------------------------------------------
        # CONTEXT DELLA STESSA SEQUENZA
        # ---------------------------------------------

        context = build_context_from_sequence(
            sequence,
            original_context,
        )

        # ---------------------------------------------
        # BiLSTM -> detection output -> RL state
        # ---------------------------------------------

        raw_state, detection_output = (
            generate_rl_state(
                sequence=sequence,
                context=context,
                model=lstm_model,
                scaler=scaler,
            )
        )


        # ---------------------------------------------
        # BiLSTM statistics
        # ---------------------------------------------

        probability = float(
            detection_output[
                "attack_probability"
            ]
        )

        probabilities.append(
            probability
        )

        predicted_label = int(
            probability
            >= DETECTION_THRESHOLD
        )

        true_label = int(
            true_label
        )


        

        # ---------------------------------------------
        # NORMALIZZAZIONE DQN
        # ---------------------------------------------

        normalized_state = (
            normalize_rl_state(
                raw_state
            )
        )

        # ---------------------------------------------
        # Q-VALUES
        # ---------------------------------------------

        q_values = agent.model.predict(
            np.expand_dims(
                normalized_state,
                axis=0,
            ),
            verbose=0,
        )[0]

        # ---------------------------------------------
        # DECISIONE
        # ---------------------------------------------
        #
        # Usiamo direttamente argmax.
        # Quindi NON c'è nessuna scelta casuale.
        # ---------------------------------------------

        action = int(np.argmax(q_values))

        action_name = ACTIONS[action]

        action_counts[action_name] += 1

        # ALLOW = NORMAL
        # Qualsiasi azione di mitigazione = ATTACK
        #

        dqn_predicted_label = int(
            action_name != "ALLOW"
        )

        dqn_y_true.append(
            true_label
        )

        dqn_y_pred.append(
            dqn_predicted_label
        )


        # =====================================================
        # DQN ACTION ANALYSIS
        # =====================================================

        if true_label == 0:

            normal_action_counts[
                action_name
            ] += 1

        else:

            attack_action_counts[
                action_name
            ] += 1


        if (
            true_label == 1
            and predicted_label == 0
        ):

            false_negative_action_counts[
                action_name
            ] += 1




        # =====================================================
        # DQN ACTIONS BY BiLSTM CONFUSION MATRIX
        # =====================================================

        if true_label == 0 and predicted_label == 0:

            # TRUE NEGATIVE
            tn_action_counts[action_name] += 1

        elif true_label == 0 and predicted_label == 1:

            # FALSE POSITIVE
            fp_action_counts[action_name] += 1

        elif true_label == 1 and predicted_label == 0:

            # FALSE NEGATIVE
            fn_action_counts[action_name] += 1

        elif true_label == 1 and predicted_label == 1:

            # TRUE POSITIVE
            tp_action_counts[action_name] += 1



        




        # =================================================
        # STATE DISTRIBUTION ANALYSIS
        # =================================================

        if true_label == 0:

            normal_states.append(
                raw_state.copy()
            )

        else:

            attack_states.append(
                raw_state.copy()
            )









        if (
            true_label == 1
            and predicted_label == 1
        ):
            tp += 1

        elif (
            true_label == 0
            and predicted_label == 0
        ):
            tn += 1

        elif (
            true_label == 0
            and predicted_label == 1
        ):
            fp += 1

        elif (
            true_label == 1
            and predicted_label == 0
        ):
            fn += 1

        # ---------------------------------------------
        # PRINT PRIME N SEQUENZE
        # ---------------------------------------------

        if (
            MAX_SEQUENCES_TO_PRINT is None
            or index <= MAX_SEQUENCES_TO_PRINT
        ):

            print_result(
                index=index,
                sequence=sequence,
                true_label=true_label,
                context=context,
                detection_output=detection_output,
                raw_state=raw_state,
                normalized_state=normalized_state,
                q_values=q_values,
                action_name=action_name,
            )

        if index % 500 == 0:

            print(
                f"\n[PROGRESS] "
                f"Processate {index}/"
                f"{len(sequences)} sequenze..."
            )

    # =================================================
    # SUMMARY
    # =================================================

    total = processed_sequences
    #total = len(sequences)

    correct = tp + tn

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    average_probability = (
        float(
            np.mean(
                probabilities
            )
        )
        if probabilities
        else 0.0
    )

    print(
        "\n\n"
        + "=" * 70
    )

    print(
        "PIPELINE SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        "Sequenze processate:",
        total
    )

    print(
        "Probabilità media attacco:",
        f"{average_probability:.4f}"
    )

    print(
        "BiLSTM accuracy:",
        f"{accuracy * 100:.2f}%"
    )



    # =================================================
    # DQN CLASSIFICATION METRICS
    # =================================================

    dqn_accuracy = accuracy_score(
        dqn_y_true,
        dqn_y_pred
    )

    dqn_precision = precision_score(
        dqn_y_true,
        dqn_y_pred,
        zero_division=0
    )

    dqn_recall = recall_score(
        dqn_y_true,
        dqn_y_pred,
        zero_division=0
    )

    dqn_f1 = f1_score(
        dqn_y_true,
        dqn_y_pred,
        zero_division=0
    )

    dqn_cm = confusion_matrix(
        dqn_y_true,
        dqn_y_pred
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "DQN CLASSIFICATION REPORT"
    )

    print(
        "=" * 70
    )

    print(
        classification_report(
            dqn_y_true,
            dqn_y_pred,
            digits=2,
            zero_division=0
        )
    )

    print(
        "Confusion matrix:"
    )

    print(
        dqn_cm
    )

    print(
        f"\nAccuracy : {dqn_accuracy:.4f}"
    )

    print(
        f"Precision: {dqn_precision:.4f}"
    )

    print(
        f"Recall   : {dqn_recall:.4f}"
    )

    print(
        f"F1-score : {dqn_f1:.4f}"
    )








    print(
        "\n--- CONFUSION MATRIX ---"
    )

    print(
        "True Positive :",
        tp
    )

    print(
        "True Negative :",
        tn
    )

    print(
        "False Positive:",
        fp
    )

    print(
        "False Negative:",
        fn
    )

    print(
        "\n--- DQN ACTION DISTRIBUTION ---"
    )


    print_action_distribution(
        "DQN ACTIONS - TRUE NORMAL",
        normal_action_counts
    )

    print_action_distribution(
        "DQN ACTIONS - TRUE ATTACK",
        attack_action_counts
    )

    print_action_distribution(
        "DQN ACTIONS - BiLSTM FALSE NEGATIVES",
        false_negative_action_counts
    )




    print_action_distribution(
        "DQN ACTIONS - TRUE NEGATIVES (TN)",
        tn_action_counts
    )

    print_action_distribution(
        "DQN ACTIONS - FALSE POSITIVES (FP)",
        fp_action_counts
    )

    print_action_distribution(
        "DQN ACTIONS - FALSE NEGATIVES (FN)",
        fn_action_counts
    )

    print_action_distribution(
        "DQN ACTIONS - TRUE POSITIVES (TP)",
        tp_action_counts
    )


    






    # =====================================================
    # DQN CORRECTION / PRESERVATION METRICS
    # =====================================================

    tn_total = sum(tn_action_counts.values())
    fp_total = sum(fp_action_counts.values())
    fn_total = sum(fn_action_counts.values())
    tp_total = sum(tp_action_counts.values())


    # Il DQN preserva correttamente un normale
    tn_preservation_rate = (
        tn_action_counts["ALLOW"] / tn_total * 100
        if tn_total > 0
        else 0.0
    )


    # La BiLSTM genera un falso positivo,
    # ma il DQN evita una mitigazione scegliendo ALLOW
    fp_correction_rate = (
        fp_action_counts["ALLOW"] / fp_total * 100
        if fp_total > 0
        else 0.0
    )


    # La BiLSTM manca un attacco,
    # ma il DQN decide comunque di intervenire
    fn_recovery_count = (
        fn_total
        -
        fn_action_counts["ALLOW"]
    )

    fn_recovery_rate = (
        fn_recovery_count / fn_total * 100
        if fn_total > 0
        else 0.0
    )


    # La BiLSTM riconosce correttamente l'attacco
    # e il DQN NON lo lascia passare
    tp_response_count = (
        tp_total
        -
        tp_action_counts["ALLOW"]
    )

    tp_response_rate = (
        tp_response_count / tp_total * 100
        if tp_total > 0
        else 0.0
    )


    print("\n" + "=" * 70)
    print("DQN / BiLSTM INTERACTION METRICS")
    print("=" * 70)

    print(
        f"TN preservation rate : "
        f"{tn_preservation_rate:.2f}%"
    )

    print(
        f"FP correction rate   : "
        f"{fp_correction_rate:.2f}%"
    )

    print(
        f"FN recovery rate     : "
        f"{fn_recovery_rate:.2f}% "
        f"({fn_recovery_count}/{fn_total})"
    )

    print(
        f"TP response rate     : "
        f"{tp_response_rate:.2f}% "
        f"({tp_response_count}/{tp_total})"
    )















    for (
        action_name,
        count
    ) in action_counts.items():

        percentage = (
            count
            / total
            * 100
            if total > 0
            else 0.0
        )

        print(
            f"{action_name:<20}: "
            f"{count:<6} "
            f"({percentage:.2f}%)"
        )

    print(
        "=" * 70
    )

    print_state_distribution(
        "STATE DISTRIBUTION - TRUE NORMAL",
        normal_states
    )

    print_state_distribution(
        "STATE DISTRIBUTION - TRUE ATTACK",
        attack_states
    )

    # =================================================
    # FIGURE FINALI PER LA TESI
    # =================================================

    save_all_thesis_figures(
        normal_states=normal_states,
        attack_states=attack_states,
        normal_action_counts=normal_action_counts,
        attack_action_counts=attack_action_counts,
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
    )


# =====================================================
# ENTRY POINT
# =====================================================

if __name__ == "__main__":

    run_pipeline()