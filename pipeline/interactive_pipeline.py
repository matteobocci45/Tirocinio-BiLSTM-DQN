"""
Online Cyber Defense - interactive pipeline

Run from the project root with:

    python -m pipeline.interactive_pipeline

Design goals
------------
- Does NOT retrain or modify the frozen BiLSTM/DQN models.
- Reuses the existing frozen scaler and categorical encoders.
- Maintains independent online history for each (user, ip) pair.
- Recomputes temporal features from the relevant entity history.
- Uses the event timestamp, never the current computer clock.
- Keeps COLD START clearly separated from DQN decisions.
- Supports an empty session or preload from logsprova.json.
- Persists manually inserted raw events across executions via online_events.jsonl.

IMPORTANT
---------
This module assumes that detection.feature_engineering exposes a function named
`feature_engineering`. Its exact return type may be:
    - a list of enriched dicts,
    - a NumPy array,
    - or a tuple whose first item is one of the above.

The adapter `_run_feature_engineering()` handles those common cases.

The online sequence is constructed locally from the enriched entity history using
the thesis rules:
    entity = (user, ip)
    temporal window = 5 minutes
    min length = 5
    max length = 50

That avoids refitting anything and avoids depending on a training-only return
format from sequence_builder.py. If your existing sequence_builder.py contains
additional padding semantics, set PADDING_SIDE below to match it.
"""

from __future__ import annotations

import json
import pickle
import re
import sys
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

try:
    from tensorflow.keras.models import load_model
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "TensorFlow/Keras non disponibile. Attiva lo stesso ambiente usato per il training."
    ) from exc

try:
    from detection.feature_engineering import feature_engineering
except ImportError as exc:
    raise RuntimeError(
        "Impossibile importare detection.feature_engineering.feature_engineering. "
        "Esegui questo modulo dalla root del progetto con "
        "`python -m pipeline.interactive_pipeline`."
    ) from exc


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

LSTM_MODEL_PATH = MODELS_DIR / "lstm.keras"
DQN_MODEL_PATH = MODELS_DIR / "dqn.keras"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
ENCODERS_PATH = MODELS_DIR / "encoders.pkl"
DEFAULT_LOGS_PATH = PROJECT_ROOT / "logsprova.json"
AUDIT_LOG_PATH = Path(__file__).resolve().with_name("online_decisions.jsonl")
ONLINE_EVENTS_PATH = Path(__file__).resolve().with_name("online_events.jsonl")

SEQUENCE_WINDOW_SECONDS = 5 * 60
MIN_SEQUENCE_LENGTH = 5
MAX_SEQUENCE_LENGTH = 50

# If the BiLSTM has a fixed timestep dimension, short valid sequences are padded.
# Keep this consistent with the offline sequence_builder.py used for training.
PADDING_SIDE = "pre"  # "pre" or "post"

REQUESTS_SCALE = 4486.0

# Online historical-risk memory.
#
# Semantics:
#   historical_risk[i] = risk accumulated BEFORE event i
#   event i failed      = updates the memory for event i+1
#
# This preserves causality and avoids target leakage.
RISK_DECAY = 0.95
RISK_FAILED_INCREMENT = 0.10
RISK_REQUESTS_THRESHOLD = 948.0
RISK_REQUESTS_INCREMENT = 0.15

# Contextual cold-start policy.
# These weights are intentionally explicit and interpretable. They are NOT
# learned by the DQN and should be documented separately in the thesis.
COLD_START_WEIGHTS = {
    "failed": 0.20,
    "failure_ratio": 0.10,
    "is_night": 0.10,
    "privileged": 0.15,
    "critical_resource": 0.20,
    "no_certificate": 0.05,
    "historical_risk": 0.15,
    "request_pressure": 0.05,
}

COLD_START_THRESHOLDS = {
    "ALLOW_MAX": 0.20,
    "MONITOR_MAX": 0.40,
    "MFA_MAX": 0.65,
}

# Exact rules used by preprocessing/create_logs.py.
PRIVILEGED_ROLE_NAMES = {
    "Admin",
    "Administrator",
    "SuperAdmin",
    "SecurityAdmin",
}

CRITICAL_RESOURCES = {
    "AspNetUsers",
    "Users",
    "Identity",
    "Authentication",
    "Credentials",
    "Security",
    "Admin",
}

# Extended taxonomy used ONLY by the rule-based COLD-START policy.
# It must never replace CRITICAL_RESOURCES in the frozen BiLSTM/DQN path.
CONTEXTUAL_CRITICAL_RESOURCES = {
    "AspNetUsers",
    "Users",
    "Identity",
    "Authentication",
    "Credentials",
    "Security",
    "Admin",
    "User",
    "Role",
    "Claim",
    "Permission",
    "Auth",
    "Session",
    "Token",
    "Password",
    "Secret",
    "Certificate",
    "Audit",
    "Backup",
    "Configuration",
    "Setting",
}

ACTION_NAMES = (
    "ALLOW",
    "MONITOR",
    "MFA",
    "BLOCK_SESSION",
    "BLOCK_IP",
    "QUARANTINE",
)

FEATURE_NAMES = (
    "event_id",
    "source_id",
    "role_id",
    "resource_id",
    "delta",
    "delta_log",
    "hour",
    "weekday",
    "is_night",
    "failed",
    "requests_5m",
    "has_certificate",
    "privileged",
    "critical_resource",
    "historical_risk",
)

NUMERIC_START_INDEX = 4
EXPECTED_FEATURES = 15
EXPECTED_RL_STATE = 9


# =============================================================================
# SMALL UTILITIES
# =============================================================================

def _banner(title: str) -> None:
    print("\n" + "=" * 62)
    print(title)
    print("=" * 62)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _parse_bool(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value != 0)
    if isinstance(value, (float, np.floating)):
        return int(value != 0.0)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "si", "sì", "failed", "failure"}:
            return 1
        if normalized in {"0", "false", "no", "n", "success", "ok"}:
            return 0
    return 0


def _first_present(mapping: Mapping[str, Any], names: Sequence[str]) -> Any:
    for name in names:
        if name in mapping and mapping[name] not in (None, ""):
            return mapping[name]
    return None


def _parse_timestamp(value: Any) -> datetime:
    """
    Parse the timestamp contained in the event.

    Supported examples:
        2025-12-21T22:53:26
        2025-12-21T21:53:26.765Z
        2025-12-21 22:53:26
        21-Dec-2025 22:53:26

    The raw log format is parsed without relying on the OS locale.
    """
    if isinstance(value, datetime):
        return value

    if isinstance(value, (int, float, np.integer, np.floating)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)

    if not isinstance(value, str):
        raise ValueError(f"tipo timestamp non supportato: {type(value).__name__}")

    text = value.strip()
    if not text:
        raise ValueError("timestamp vuoto")

    # logsprova.json format, e.g. 21-Dec-2025 22:53:26
    raw_match = re.fullmatch(
        r"(\d{1,2})-([A-Za-z]{3})-(\d{4})[ T]"
        r"(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?",
        text,
    )

    if raw_match:
        day, month_name, year, hour, minute, second, fraction = raw_match.groups()
        months = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4,
            "may": 5, "jun": 6, "jul": 7, "aug": 8,
            "sep": 9, "oct": 10, "nov": 11, "dec": 12,
        }
        month = months.get(month_name.lower())
        if month is None:
            raise ValueError(f"mese non riconosciuto nel timestamp: {month_name!r}")

        microsecond = int((fraction + "000000")[:6]) if fraction else 0
        return datetime(
            int(year), month, int(day),
            int(hour), int(minute), int(second), microsecond
        )

    iso_text = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        return datetime.fromisoformat(iso_text)
    except ValueError:
        pass

    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y/%m/%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass

    raise ValueError(
        f"timestamp non riconosciuto: {value!r}. "
        "Formati accettati includono '21-Dec-2025 22:53:26' "
        "e '2025-12-21T22:53:26'."
    )


def _timestamp_sort_key(event: Mapping[str, Any]) -> float:
    dt = _event_datetime(event)
    if dt.tzinfo is not None:
        return dt.timestamp()
    # For naive historical simulations, use a timezone-independent ordinal-like value.
    return (
        dt.toordinal() * 86400
        + dt.hour * 3600
        + dt.minute * 60
        + dt.second
        + dt.microsecond / 1_000_000
    )


def _event_datetime(event: Mapping[str, Any]) -> datetime:
    value = _first_present(
        event,
        ("timestamp", "time", "datetime", "@timestamp", "event_timestamp"),
    )
    if value is None:
        raise ValueError(
            "Evento senza timestamp. Chiavi accettate: "
            "timestamp, time, datetime, @timestamp, event_timestamp."
        )
    return _parse_timestamp(value)


def _nested_get(mapping: Mapping[str, Any], *path: str) -> Any:
    current: Any = mapping
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _clean_identity_value(value: Any) -> str | None:
    """
    Treat None and blank strings as missing, but preserve valid values.
    This is crucial for heterogeneous raw logs where e.g. message.User may
    exist as "" while message.authUsername contains the real identity.
    """
    if value is None:
        return None

    value = str(value).strip()
    if not value:
        return None

    return value


def _extract_user(event: Mapping[str, Any]) -> str:
    """
    Resolve user across the heterogeneous schemas present in logsprova.json.

    IMPORTANT: a present-but-empty candidate does NOT stop fallback resolution.
    """
    candidates = [
        # Already-flat / correlated-session variants
        event.get("user"),
        event.get("username"),
        event.get("user_name"),
        event.get("user_id"),
        event.get("account"),
        event.get("User"),
        event.get("authUsername"),

        # ApplicationAudit / Blazor
        _nested_get(event, "message", "User"),

        # OPNsense / captive portal
        _nested_get(event, "message", "authUsername"),

        # Generic nested alternatives
        _nested_get(event, "message", "username"),
        _nested_get(event, "message", "user"),
        _nested_get(event, "message", "UserContext", "claims", "name"),
    ]

    for candidate in candidates:
        value = _clean_identity_value(candidate)
        if value is not None:
            return value

    raise ValueError(
        "Evento senza user utilizzabile. Provati campi flat/correlati "
        "(user, username, user_name, user_id, account, User, authUsername) "
        "e nested message.User, message.authUsername, message.username, "
        "message.user, message.UserContext.claims.name."
    )


def _normalize_remote_endpoint(value: Any) -> str | None:
    if value is None:
        return None

    endpoint = str(value).strip()
    if not endpoint:
        return None

    # [IPv6]:port
    if endpoint.startswith("[") and "]" in endpoint:
        return endpoint[1:endpoint.index("]")]

    # Common IPv4:port format.
    if endpoint.count(":") == 1:
        host, maybe_port = endpoint.rsplit(":", 1)
        if maybe_port.isdigit():
            return host

    # Bare IPv4 / IPv6.
    return endpoint


def _extract_ip(event: Mapping[str, Any]) -> str:
    """
    Resolve IP across flat, correlated-session, Blazor and OPNsense schemas.
    Blank candidates are ignored so later fallbacks are still attempted.
    """
    candidates = [
        # Already-flat / correlated-session variants
        event.get("ip"),
        event.get("source_ip"),
        event.get("src_ip"),
        event.get("client_ip"),
        event.get("ip_address"),
        event.get("IP"),
        event.get("ipBehindProxy"),

        # ApplicationAudit / Blazor
        _nested_get(event, "message", "IP"),

        # OPNsense / captive portal
        _nested_get(event, "message", "ip"),
        _nested_get(event, "message", "ipBehindProxy"),
    ]

    for candidate in candidates:
        value = _clean_identity_value(candidate)
        if value is not None:
            return value

    # Last fallback: request metadata endpoint, possibly "IP:port".
    endpoint = _normalize_remote_endpoint(
        _nested_get(event, "message", "RequestMetadata", "remoteEndpoint")
    )
    endpoint = _clean_identity_value(endpoint)
    if endpoint is not None:
        return endpoint

    raise ValueError(
        "Evento senza IP utilizzabile. Provati campi flat/correlati "
        "(ip, source_ip, src_ip, client_ip, ip_address, IP, ipBehindProxy) "
        "e nested message.IP, message.ip, message.ipBehindProxy, "
        "message.RequestMetadata.remoteEndpoint."
    )


def _entity_key(event: Mapping[str, Any]) -> Tuple[str, str]:
    return _extract_user(event), _extract_ip(event)


def _extract_failed(event: Mapping[str, Any]) -> int:
    """
    Exact online equivalent of detection/feature_engineering.py:

        failed = int(log.get("status_code", 200) >= 400)

    EventType names such as `captiveportal.auth_failed` do NOT override this
    rule, because the frozen model was trained with status_code semantics.
    """
    status_code = event.get("status_code", 200)

    try:
        status_code = int(status_code)
    except (TypeError, ValueError):
        status_code = 200

    return int(status_code >= 400)


def _load_pickle_compatible(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"File non trovato: {path}")

    if joblib is not None:
        try:
            return joblib.load(path)
        except Exception:
            pass

    with path.open("rb") as fh:
        return pickle.load(fh)


# =============================================================================
# MODEL / ARTIFACT LOADING
# =============================================================================

def _resolve_scaler_object(obj: Any) -> Any:
    """
    Return the actual fitted scaler object.

    Some training pipelines persist:
        StandardScaler(...)
    directly, while others persist a wrapper such as:
        {"scaler": StandardScaler(...), ...}

    We require the final object to expose `.transform()`.
    """
    if hasattr(obj, "transform") and callable(getattr(obj, "transform")):
        return obj

    if isinstance(obj, Mapping):
        # Common explicit wrapper keys first.
        preferred_keys = (
            "scaler",
            "standard_scaler",
            "numeric_scaler",
            "feature_scaler",
            "x_scaler",
        )

        for key in preferred_keys:
            candidate = obj.get(key)
            if hasattr(candidate, "transform") and callable(
                getattr(candidate, "transform")
            ):
                print(f"  Scaler wrapper key detected: {key!r}")
                return candidate

        # Fallback: if exactly one value behaves like a scaler, use it.
        candidates = [
            (key, value)
            for key, value in obj.items()
            if hasattr(value, "transform")
            and callable(getattr(value, "transform"))
        ]

        if len(candidates) == 1:
            key, candidate = candidates[0]
            print(f"  Scaler object auto-detected under key: {key!r}")
            return candidate

        if len(candidates) > 1:
            keys = [key for key, _ in candidates]
            raise ValueError(
                "scaler.pkl contiene più oggetti con metodo transform(); "
                f"non posso scegliere in modo sicuro. Chiavi candidate: {keys}"
            )

        raise ValueError(
            "scaler.pkl è un dizionario ma non contiene alcun oggetto con "
            "metodo transform(). Chiavi disponibili: "
            f"{list(obj.keys())}"
        )

    raise TypeError(
        "Formato scaler.pkl non supportato: "
        f"{type(obj).__name__}. Atteso uno scaler con .transform() oppure "
        "un dict che lo contiene."
    )


@dataclass
class FrozenArtifacts:
    lstm: Any
    dqn: Any
    scaler: Any
    encoders: Any


def load_frozen_artifacts() -> FrozenArtifacts:
    print("Caricamento artefatti congelati...")

    for path in (LSTM_MODEL_PATH, DQN_MODEL_PATH, SCALER_PATH, ENCODERS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Artefatto mancante: {path}")

    # compile=False avoids requiring training-only optimizer/loss objects.
    lstm = load_model(LSTM_MODEL_PATH, compile=False)
    dqn = load_model(DQN_MODEL_PATH, compile=False)
    scaler_payload = _load_pickle_compatible(SCALER_PATH)
    scaler = _resolve_scaler_object(scaler_payload)
    encoders = _load_pickle_compatible(ENCODERS_PATH)

    print(f"  BiLSTM   : {LSTM_MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  DQN      : {DQN_MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Scaler   : {SCALER_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Scaler type: {type(scaler).__name__}")
    print(f"  Encoders : {ENCODERS_PATH.relative_to(PROJECT_ROOT)}")
    print("Artefatti caricati senza fit/retraining.")

    return FrozenArtifacts(lstm=lstm, dqn=dqn, scaler=scaler, encoders=encoders)


# =============================================================================
# EVENT VALIDATION / NORMALIZATION
# =============================================================================

def _offline_extract_role(message: Mapping[str, Any]) -> str:
    """Exact equivalent of create_logs.extract_role()."""
    context = message.get("UserContext", {})
    if not isinstance(context, Mapping):
        context = {}

    roles = context.get("roles", [])
    if isinstance(roles, list) and roles:
        return str(roles[-1])

    return "User"


def _offline_is_privileged(role: str) -> int:
    """Exact equivalent of create_logs.is_privileged()."""
    return int(role in PRIVILEGED_ROLE_NAMES)


def _offline_extract_resource(message: Mapping[str, Any]) -> str:
    """Exact equivalent of create_logs.extract_resource()."""
    resource_context = message.get("ResourceContext", {})

    if isinstance(resource_context, Mapping):
        return str(
            resource_context.get(
                "table",
                resource_context.get("database", "UNKNOWN"),
            )
        )

    return "UNKNOWN"


def _offline_check_critical_resource(resource: Any) -> int:
    """Exact equivalent of create_logs.check_critical_resource()."""
    if not resource:
        return 0

    resource_text = str(resource).lower()

    for item in CRITICAL_RESOURCES:
        if item.lower() in resource_text:
            return 1

    return 0


def _contextual_check_critical_resource(resource: Any) -> int:
    """
    Extended criticality used ONLY in COLD START.

    The frozen model-facing `critical_resource` still comes from
    CRITICAL_RESOURCES, preserving train/inference consistency.
    """
    if not resource:
        return 0

    resource_text = str(resource).lower()

    for item in CONTEXTUAL_CRITICAL_RESOURCES:
        if item.lower() in resource_text:
            return 1

    return 0


def _flatten_online_raw_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Convert one interactive raw JSON event to the SAME schema consumed by
    feature_engineering.py after preprocessing/create_logs.py.

    The input is a single raw event rather than a correlated session, so user/ip
    are resolved from the supported raw schemas (Blazor/ApplicationAudit,
    OPNsense/captive portal, or already-flat fields). All feature-bearing fields
    then follow create_logs.py exactly.
    """
    raw = deepcopy(dict(event))
    event_dt = _event_datetime(raw)
    user = _extract_user(raw)
    ip = _extract_ip(raw)

    message = raw.get("message")
    if not isinstance(message, Mapping):
        # Already-flat events are also supported.
        message = {}

    # If this is already a flat event, preserve its values. Otherwise derive
    # them exactly as create_logs.py does from the nested message.
    if "event_type" in raw:
        event_type = raw.get("event_type", "UNKNOWN")
    else:
        event_type = message.get("EventType", "UNKNOWN")

    source = raw.get("source")

    if "status_code" in raw:
        status_code = raw.get("status_code", 200)
    else:
        status_code = message.get("HTTPStatusCode", 200)

    try:
        status_code = int(status_code)
    except (TypeError, ValueError):
        status_code = 200

    if "role" in raw:
        role = str(raw.get("role") or "User")
    else:
        role = _offline_extract_role(message)

    if "privileged" in raw:
        privileged = int(_parse_bool(raw.get("privileged")))
    else:
        privileged = _offline_is_privileged(role)

    if "resource" in raw:
        resource = raw.get("resource") or "UNKNOWN"
    else:
        resource = _offline_extract_resource(message)

    if "critical_resource" in raw:
        critical_resource = int(_parse_bool(raw.get("critical_resource")))
    else:
        critical_resource = _offline_check_critical_resource(resource)

    if "certificate" in raw:
        certificate = raw.get("certificate")
    else:
        certificate = message.get("ClientCertificate")

    if "route" in raw:
        route = raw.get("route") or "UNKNOWN"
    else:
        route = message.get("Route", "UNKNOWN")

    # create_logs.py stores Fingerprint exactly; support lowercase fingerprint
    # only for OPNsense raw logs, where that is the original field name.
    if "fingerprint" in raw:
        fingerprint = raw.get("fingerprint", {})
    else:
        fingerprint = message.get("Fingerprint", message.get("fingerprint", {}))

    # Device fields are not BiLSTM features, but keep useful flat context.
    os_value = raw.get("os")
    if os_value is None and isinstance(fingerprint, Mapping):
        os_value = fingerprint.get("platform")

    flat = {
        "user": user,
        "ip": ip,
        "event_time": event_dt,
        "event_type": event_type,
        "source": source,
        "status_code": status_code,
        "role": role,
        "privileged": privileged,
        "resource": resource,
        "critical_resource": critical_resource,
        "certificate": certificate,
        "os": os_value,
        "mac": raw.get("mac"),
        "hostname": raw.get("hostname", raw.get("host")),
        "fingerprint": fingerprint,
        "route": route,

        # Keep the original raw timestamp for display/debug convenience.
        "timestamp": _first_present(
            raw,
            ("time", "timestamp", "datetime", "@timestamp", "event_timestamp"),
        ),

        # Internal online identity only.
        "_online_event_id": raw.get("_online_event_id", str(uuid.uuid4())),
    }

    # Do NOT copy Suspicious/_suspicious into the flat inference event.
    # `user_risk` from create_logs.calculate_initial_user_risk() is also omitted:
    # feature_engineering.py does not use it for the 15-D model input, and its
    # historical version depended on Suspicious.
    return flat



def validate_event(event: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Validate + flatten one online event into the exact schema expected by the
    existing feature_engineering.py.
    """
    if not isinstance(event, Mapping):
        raise ValueError("L'evento deve essere un oggetto JSON.")

    flat = _flatten_online_raw_event(event)

    # Final mandatory checks.
    if not isinstance(flat["event_time"], datetime):
        raise ValueError("event_time non è stato normalizzato a datetime.")

    return flat


def read_event_from_console() -> Dict[str, Any]:
    print("\nInserisci un singolo evento JSON.")
    print("Puoi incollarlo su una sola riga oppure su più righe.")
    print("Termina un JSON multilinea con una riga contenente solo END.")
    print("Scrivi CANCEL per annullare.\n")

    first = input("> ").strip()
    if first.upper() == "CANCEL":
        raise KeyboardInterrupt

    lines = [first]

    # Fast path: valid one-line JSON.
    try:
        parsed = json.loads(first)
        if not isinstance(parsed, Mapping):
            raise ValueError("L'evento deve essere un oggetto JSON.")
        return dict(parsed)
    except json.JSONDecodeError:
        pass

    while True:
        line = input("... ")
        if line.strip().upper() == "END":
            break
        lines.append(line)

    text = "\n".join(lines)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"JSON non valido: {exc.msg} (linea {exc.lineno}, colonna {exc.colno})"
        ) from exc

    if not isinstance(parsed, Mapping):
        raise ValueError("L'evento deve essere un oggetto JSON.")

    return dict(parsed)


# =============================================================================
# HISTORY PRELOAD
# =============================================================================

def _extract_event_list_from_json(
    payload: Any,
    *,
    skip_unusable: bool = False,
) -> List[Dict[str, Any]]:
    """
    Extract events from logsprova.json.

    Interactive/manual events remain strict: they must have timestamp + user + ip.

    Historical preload can be tolerant (`skip_unusable=True`): raw log records
    that cannot be assigned to a concrete (user, ip) entity are skipped instead
    of aborting the whole session. This is preferable to inventing an
    UNKNOWN_USER/UNKNOWN_IP entity, which would mix unrelated traffic and corrupt
    temporal features such as delta, requests_5m and historical_risk.
    """
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        items = None
        for key in ("events", "logs", "data"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                items = candidate
                break

        if items is None:
            if any(
                k in payload
                for k in ("timestamp", "time", "event_time", "datetime", "@timestamp")
            ):
                items = [payload]
            else:
                raise ValueError(
                    "Formato JSON non riconosciuto. Atteso una lista di eventi oppure "
                    "un oggetto con chiave events/logs/data."
                )
    else:
        raise ValueError("Il file JSON deve contenere una lista o un oggetto.")

    result: List[Dict[str, Any]] = []
    skipped: List[str] = []

    for index, item in enumerate(items):
        try:
            result.append(validate_event(item))
        except Exception as exc:
            if skip_unusable:
                skipped.append(f"indice {index}: {type(exc).__name__}: {exc}")
                continue
            raise ValueError(
                f"Evento non compatibile all'indice {index}: {exc}"
            ) from exc

    if skip_unusable and skipped:
        print()
        print("Preload history: alcuni record raw sono stati esclusi.")
        print(f"  Record totali : {len(items)}")
        print(f"  Caricati      : {len(result)}")
        print(f"  Esclusi       : {len(skipped)}")
        print(
            "  Motivo        : impossibile costruire una valida entità "
            "(user, ip) o timestamp"
        )
        print()
        print("Primi record esclusi:")
        for message in skipped[:10]:
            print(f"  - {message}")
        if len(skipped) > 10:
            print(f"  ... altri {len(skipped) - 10} record esclusi")
        print()

    if not result:
        raise ValueError(
            "Nessun evento utilizzabile trovato nel file storico: "
            "non è possibile inizializzare la memoria online."
        )

    return result


def load_events_from_json(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"File storico non trovato: {path}")

    with path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)

    events = _extract_event_list_from_json(payload, skip_unusable=True)
    print(f"Eventi storici utilizzabili caricati: {len(events)} da {path}.")
    return events


def append_online_raw_event(
    raw_event: Mapping[str, Any],
    path: Path = ONLINE_EVENTS_PATH,
) -> Path:
    """
    Persist one manually inserted RAW event.

    The event is deliberately stored before flattening/feature engineering.
    On the next preload it is validated and transformed again by the normal
    inference pipeline.
    """
    if not isinstance(raw_event, Mapping):
        raise ValueError("L'evento online da salvare deve essere un oggetto JSON.")

    # Console input originates from json.loads(), therefore it should already
    # be JSON-safe. deepcopy prevents accidental mutation while serializing.
    payload = deepcopy(dict(raw_event))

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                payload,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )

    return path


def load_online_events_jsonl(
    path: Path = ONLINE_EVENTS_PATH,
) -> List[Dict[str, Any]]:
    """
    Load persisted online RAW events and normalize them with validate_event().

    Corrupt/unusable lines are skipped individually so one damaged line cannot
    prevent the historical session from starting.
    """
    if not path.exists():
        print(f"Nessun evento online persistito trovato: {path}")
        return []

    events: List[Dict[str, Any]] = []
    skipped: List[str] = []
    total = 0

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            total += 1

            try:
                raw = json.loads(stripped)
                if not isinstance(raw, Mapping):
                    raise ValueError("la riga JSONL non contiene un oggetto JSON")
                events.append(validate_event(raw))
            except Exception as exc:
                skipped.append(
                    f"riga {line_number}: {type(exc).__name__}: {exc}"
                )

    print()
    print("Persistent online events:")
    print(f"  Record totali : {total}")
    print(f"  Caricati      : {len(events)}")
    print(f"  Esclusi       : {len(skipped)}")
    print(f"  File          : {path}")

    if skipped:
        print()
        print("Primi eventi online persistiti esclusi:")
        for message in skipped[:10]:
            print(f"  - {message}")
        if len(skipped) > 10:
            print(f"  ... altri {len(skipped) - 10} record esclusi")

    print()
    return events



def _raw_security_context(event: Mapping[str, Any]) -> Dict[str, Any]:
    """Diagnostic/security context from the already-flat online event."""
    certificate = event.get("certificate")

    return {
        "roles": [event.get("role", "User")],
        "privileged": int(event.get("privileged", 0)),
        "has_certificate": int(certificate is not None),
        "critical_resource": int(event.get("critical_resource", 0)),
        "route": event.get("route"),
        "event_type": event.get("event_type"),
        "resource_database": None,
        "resource_table": event.get("resource"),
        "resource_server": None,
        "certificate_present": certificate is not None,
    }


def _merge_unambiguous_raw_context(
    engineered_context: Mapping[str, float],
    event: Mapping[str, Any],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    context = dict(engineered_context)
    raw = _raw_security_context(event)

    # These fields are unambiguous in the raw/normalized event.
    context["failed"] = float(_extract_failed(event))
    context["privileged"] = float(raw["privileged"])
    context["has_certificate"] = float(raw["has_certificate"])

    # Do not guess critical_resource: override only if explicitly present.
    if raw["critical_resource"] is not None:
        context["critical_resource"] = float(raw["critical_resource"])

    return context, raw


# =============================================================================
# FEATURE ENGINEERING ADAPTER
# =============================================================================

def _looks_like_feature_item(item: Any) -> bool:
    """Return True when one item can plausibly represent the 15 BiLSTM features."""
    if isinstance(item, Mapping):
        # Direct named features.
        if all(name in item for name in FEATURE_NAMES):
            return True

        # Common nested representations.
        for key in ("features", "feature_vector", "vector", "x", "X"):
            if key in item:
                try:
                    if _looks_like_feature_item(item[key]):
                        return True
                except Exception:
                    pass

        return False

    try:
        arr = np.asarray(item)
    except Exception:
        return False

    return arr.ndim >= 1 and arr.reshape(-1).size == EXPECTED_FEATURES


def _looks_like_feature_container(candidate: Any) -> bool:
    """
    Detect common containers returned by feature_engineering:
      ndarray shape (N, 15)
      list of 15-D vectors
      list of dicts containing the 15 named features
      dict with key features/X/x/data
    """
    if candidate is None:
        return False

    if isinstance(candidate, np.ndarray):
        return candidate.ndim == 2 and candidate.shape[1] == EXPECTED_FEATURES

    if isinstance(candidate, Mapping):
        # A wrapper around the feature matrix.
        for key in ("features", "X", "x", "data", "feature_vectors"):
            if key in candidate and _looks_like_feature_container(candidate[key]):
                return True
        # A single feature item isn't the expected history container here.
        return False

    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
        if len(candidate) == 0:
            return False
        return _looks_like_feature_item(candidate[0])

    return False


def _extract_feature_container(candidate: Any) -> Any:
    """Unwrap a known dict wrapper if necessary."""
    if isinstance(candidate, Mapping):
        for key in ("features", "X", "x", "data", "feature_vectors"):
            if key in candidate and _looks_like_feature_container(candidate[key]):
                return candidate[key]
    return candidate


def _describe_structure(value: Any) -> str:
    """Compact diagnostic description for unexpected project return types."""
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"

    if isinstance(value, Mapping):
        keys = list(value.keys())
        return f"dict(keys={keys[:20]})"

    if isinstance(value, tuple):
        parts = ", ".join(
            f"{i}:{_describe_structure(v)}" for i, v in enumerate(value[:8])
        )
        return f"tuple({parts})"

    if isinstance(value, list):
        if not value:
            return "list(len=0)"
        return f"list(len={len(value)}, first={_describe_structure(value[0])})"

    return type(value).__name__


def _collect_feature_candidates(result: Any) -> List[Tuple[str, Any]]:
    """
    Collect every plausible N x 15 feature container from the return value of
    feature_engineering(), instead of selecting the first one blindly.
    """
    found: List[Tuple[str, Any]] = []
    seen: set[int] = set()

    def visit(value: Any, path: str) -> None:
        obj_id = id(value)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if _looks_like_feature_container(value):
            found.append((path, _extract_feature_container(value)))

        if isinstance(value, tuple):
            for i, item in enumerate(value):
                visit(item, f"{path}[{i}]")

        elif isinstance(value, Mapping):
            for key, item in value.items():
                # Avoid recursively exploding arbitrary per-event dictionaries.
                if key in (
                    "features", "X", "x", "data", "feature_vectors",
                    "enriched_events", "events", "inputs", "values",
                ):
                    visit(item, f"{path}.{key}")

    visit(result, "result")
    return found


def _feature_container_to_matrix(candidate: Any) -> np.ndarray:
    candidate = _extract_feature_container(candidate)

    if isinstance(candidate, np.ndarray):
        arr = np.asarray(candidate, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != EXPECTED_FEATURES:
            raise ValueError(
                f"Container ndarray con shape {arr.shape}; atteso (N, 15)."
            )
        return arr

    if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
        rows = [_feature_vector_from_enriched(item) for item in candidate]
        if not rows:
            return np.empty((0, EXPECTED_FEATURES), dtype=np.float32)
        return np.stack(rows, axis=0).astype(np.float32)

    raise TypeError(
        f"Container feature non convertibile in matrice: {type(candidate).__name__}"
    )


def _semantic_penalty(raw_matrix: np.ndarray, target_event: Mapping[str, Any]) -> float:
    """
    Lower is better. This checks invariants of the known 15-D schema.

    The strongest signal is feature[6] == event timestamp hour.
    """
    if raw_matrix.ndim != 2 or raw_matrix.shape[1] != EXPECTED_FEATURES:
        return float("inf")

    if raw_matrix.shape[0] == 0:
        return float("inf")

    row = raw_matrix[-1]
    expected_hour = float(_event_datetime(target_event).hour)

    penalty = abs(float(row[6]) - expected_hour) * 100.0

    # Soft plausibility checks.
    if not 0.0 <= float(row[7]) <= 6.0:
        penalty += 25.0

    for idx in (8, 9, 11, 12, 13):
        value = float(row[idx])
        if not -0.01 <= value <= 1.01:
            penalty += 20.0

    if float(row[10]) < -0.01:
        penalty += 20.0

    if float(row[14]) < -0.01:
        penalty += 10.0

    return penalty


def _inverse_numeric_features(matrix: np.ndarray, scaler: Any) -> np.ndarray:
    if not hasattr(scaler, "inverse_transform"):
        raise TypeError(
            "Lo scaler non espone inverse_transform(), necessario per capire "
            "se feature_engineering() ha già standardizzato le feature."
        )

    raw = matrix.astype(np.float32, copy=True)
    raw[:, NUMERIC_START_INDEX:] = np.asarray(
        scaler.inverse_transform(raw[:, NUMERIC_START_INDEX:]),
        dtype=np.float32,
    )
    return raw


def _prepare_feature_matrices(
    feature_result: Any,
    causal_history: Sequence[Mapping[str, Any]],
    scaler: Any,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    Return:
        raw_matrix   : 15-D features in human/original scale
        model_matrix : exact representation to feed the BiLSTM before padding
        info         : diagnostic description

    For every candidate returned by feature_engineering(), test both hypotheses:
      A) candidate is RAW   -> scale columns 4..14 for the model
      B) candidate is SCALED -> inverse-transform columns 4..14 for context

    The hypothesis whose recovered feature[6] matches the event hour is selected.
    """
    candidates = _collect_feature_candidates(feature_result)

    if not candidates:
        raise ValueError(
            "Nessun container N x 15 individuato nel return di "
            "feature_engineering(). Struttura: "
            f"{_describe_structure(feature_result)}"
        )

    expected_len = len(causal_history)
    target_event = causal_history[-1]
    scored: List[Tuple[float, np.ndarray, np.ndarray, str]] = []

    for path, candidate in candidates:
        try:
            matrix = _feature_container_to_matrix(candidate)
        except Exception:
            continue

        if matrix.shape[0] != expected_len:
            continue

        # Hypothesis A: feature_engineering returned raw features.
        raw_direct = matrix.astype(np.float32, copy=True)
        direct_penalty = _semantic_penalty(raw_direct, target_event)

        try:
            model_from_raw = _scale_lstm_sequence(raw_direct, scaler)
            scored.append(
                (
                    direct_penalty,
                    raw_direct,
                    model_from_raw,
                    f"{path} | RAW -> scaled once",
                )
            )
        except Exception:
            pass

        # Hypothesis B: feature_engineering returned already-standardized features.
        try:
            recovered_raw = _inverse_numeric_features(matrix, scaler)
            inverse_penalty = _semantic_penalty(recovered_raw, target_event)
            scored.append(
                (
                    inverse_penalty,
                    recovered_raw,
                    matrix.astype(np.float32, copy=True),
                    f"{path} | ALREADY SCALED -> inverse only for context",
                )
            )
        except Exception:
            pass

    if not scored:
        raise ValueError(
            "Sono stati trovati possibili container a 15 feature, ma nessuno "
            f"ha lunghezza {expected_len} o è compatibile con lo scaler."
        )

    scored.sort(key=lambda item: item[0])
    best_penalty, raw_matrix, model_matrix, info = scored[0]

    expected_hour = float(_event_datetime(target_event).hour)
    recovered_hour = float(raw_matrix[-1, 6])

    if abs(recovered_hour - expected_hour) > 0.25:
        alternatives = "\n".join(
            f"  score={score:.3f}  {desc}"
            for score, _, _, desc in scored[:8]
        )
        raise RuntimeError(
            "Non riesco ancora ad allineare semanticamente le 15 feature.\n"
            f"Hour attesa dal timestamp: {expected_hour:.0f}\n"
            f"Migliore hour recuperata: {recovered_hour:.6f}\n"
            "Candidati provati:\n"
            f"{alternatives}\n"
            "Inference interrotta per evitare una decisione non valida."
        )

    return raw_matrix, model_matrix, info


def _strip_ground_truth_for_inference(event: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Return a deep copy with explicit ground-truth / annotation fields removed.

    In particular, raw logs may contain message.Suspicious and
    message.SuspiciousDescription. They are annotations and must not be used to
    produce online features or historical risk.
    """
    cleaned = deepcopy(dict(event))

    ground_truth_keys = {
        "suspicious",
        "_suspicious",
        "suspiciousdescription",
        "label",
        "target",
        "ground_truth",
        "is_attack",
    }

    def scrub(obj: Any) -> None:
        if isinstance(obj, dict):
            for key in list(obj.keys()):
                if str(key).lower() in ground_truth_keys:
                    obj.pop(key, None)
                else:
                    scrub(obj[key])
        elif isinstance(obj, list):
            for item in obj:
                scrub(item)

    scrub(cleaned)
    return cleaned


def _run_feature_engineering(
    entity_history: Sequence[Mapping[str, Any]],
    encoders: Any,
) -> Any:
    """
    Call the EXISTING project feature engineering using the frozen encoders.

    We intentionally feed the complete relevant (user, ip) history, not only the
    newest event, so delta/requests_5m/historical_risk retain temporal context.
    """
    raw_history = [_strip_ground_truth_for_inference(e) for e in entity_history]

    attempts = (
        lambda: feature_engineering(raw_history, encoders=encoders),
        lambda: feature_engineering(raw_history, encoders),
    )

    last_error: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_error = exc

    raise TypeError(
        "Firma di feature_engineering() non compatibile con gli adapter previsti. "
        "Atteso uno dei formati:\n"
        "  feature_engineering(events, encoders=encoders)\n"
        "  feature_engineering(events, encoders)\n"
        f"Errore originale: {last_error}"
    )


def _len_enriched(enriched: Any) -> int:
    if isinstance(enriched, np.ndarray):
        return int(enriched.shape[0])
    return len(enriched)


def _enriched_item(enriched: Any, index: int) -> Any:
    if isinstance(enriched, np.ndarray):
        return enriched[index]
    return enriched[index]


def _feature_vector_from_enriched(item: Any) -> np.ndarray:
    """
    Convert one feature-engineering output item into the exact 15-D vector
    expected by the frozen BiLSTM.
    """
    if isinstance(item, Mapping):
        # Direct named representation.
        if all(name in item for name in FEATURE_NAMES):
            return np.asarray(
                [item[name] for name in FEATURE_NAMES],
                dtype=np.float32,
            )

        # Common nested representations.
        for key in ("features", "feature_vector", "vector", "x", "X"):
            if key in item:
                return _feature_vector_from_enriched(item[key])

        present = [name for name in FEATURE_NAMES if name in item]
        raise KeyError(
            "Elemento di feature_engineering non contiene il vettore BiLSTM. "
            f"Feature note presenti={present}; chiavi disponibili={list(item.keys())[:30]}. "
            "Probabilmente il vettore è in un altro elemento del return di "
            "feature_engineering()."
        )

    arr = np.asarray(item, dtype=np.float32).reshape(-1)
    if arr.size != EXPECTED_FEATURES:
        raise ValueError(
            f"Elemento feature con {arr.size} valori; attesi {EXPECTED_FEATURES}. "
            f"Tipo originale={type(item).__name__}."
        )

    return arr


def _context_from_enriched(item: Any) -> Dict[str, float]:
    """
    Read RL context fields from an enriched event, supporting dict or vector.
    """
    if isinstance(item, Mapping):
        return {
            "requests_5m": _safe_float(item.get("requests_5m")),
            "hour": _safe_float(item.get("hour")),
            "is_night": _safe_float(item.get("is_night")),
            "privileged": _safe_float(item.get("privileged")),
            "critical_resource": _safe_float(item.get("critical_resource")),
            "has_certificate": _safe_float(item.get("has_certificate")),
            "historical_risk": _safe_float(item.get("historical_risk")),
            "failed": _safe_float(item.get("failed")),
        }

    vec = _feature_vector_from_enriched(item)
    return {
        "requests_5m": float(vec[10]),
        "hour": float(vec[6]),
        "is_night": float(vec[8]),
        "privileged": float(vec[12]),
        "critical_resource": float(vec[13]),
        "has_certificate": float(vec[11]),
        "historical_risk": float(vec[14]),
        "failed": float(vec[9]),
    }



def _compute_causal_historical_risk(
    history: Sequence[Mapping[str, Any]],
    requests_5m_values: Sequence[float],
) -> np.ndarray:
    """
    Replicate the offline historical-risk logic causally.

    For each event:
      1. decay the risk memory carried from previous events;
      2. expose that value as historical_risk for the CURRENT event;
      3. update the memory ONLY for the NEXT event:
           +0.10 if current event failed
           +0.15 if current requests_5m > 948
      4. clip memory to [0, 1].

    This follows the offline update block:

        new_risk = historical_risk

        if failed:
            new_risk += 0.10

        if requests_5m > 948:
            new_risk += 0.15

        new_risk = clip(new_risk, 0, 1)

    `Suspicious`, `_suspicious`, labels and attack ground truth are never used.
    """
    if len(history) != len(requests_5m_values):
        raise ValueError(
            "history e requests_5m_values devono avere la stessa lunghezza."
        )

    risks = np.zeros(len(history), dtype=np.float32)
    memory = 0.0

    for i, event in enumerate(history):
        # Risk carried from previous activity decays before the current event.
        memory = float(np.clip(memory * RISK_DECAY, 0.0, 1.0))

        # What the CURRENT event is allowed to see.
        historical_risk = memory
        risks[i] = historical_risk

        # Update ONLY the memory for the NEXT event.
        new_risk = historical_risk

        failed = int(_extract_failed(event))
        if failed:
            new_risk += RISK_FAILED_INCREMENT

        requests_5m = max(0.0, _safe_float(requests_5m_values[i]))
        if requests_5m > RISK_REQUESTS_THRESHOLD:
            new_risk += RISK_REQUESTS_INCREMENT

        memory = float(np.clip(new_risk, 0.0, 1.0))

    return risks


def _apply_online_temporal_overrides(
    raw_feature_matrix: np.ndarray,
    causal_history: Sequence[Mapping[str, Any]],
    scaler: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Make the two temporal/security features whose online semantics are known
    explicit and internally consistent:

      feature 9  = failed
      feature 14 = causal historical_risk

    Then regenerate the model-ready matrix from the corrected RAW matrix.
    This ensures BiLSTM and DQN observe the same corrected history and prevents
    double scaling.
    """
    if raw_feature_matrix.shape != (len(causal_history), EXPECTED_FEATURES):
        raise ValueError(
            "Matrice feature incompatibile con la history causale: "
            f"shape={raw_feature_matrix.shape}, events={len(causal_history)}"
        )

    corrected_raw = raw_feature_matrix.astype(np.float32, copy=True)

    # Canonical failed flag for every event.
    corrected_raw[:, 9] = np.asarray(
        [_extract_failed(event) for event in causal_history],
        dtype=np.float32,
    )

    # Causal historical risk for every event.
    # requests_5m is feature 10 in the raw 15-D schema.
    corrected_raw[:, 14] = _compute_causal_historical_risk(
        history=causal_history,
        requests_5m_values=corrected_raw[:, 10],
    )

    # Rebuild the exact BiLSTM representation from corrected raw features.
    corrected_model = _scale_lstm_sequence(corrected_raw, scaler)

    return corrected_raw, corrected_model


# =============================================================================
# ONLINE SEQUENCE CONSTRUCTION
# =============================================================================

def _seconds_between(a: datetime, b: datetime) -> float:
    if a.tzinfo is None and b.tzinfo is None:
        return (a - b).total_seconds()

    # If one is timezone-aware and the other is naive, compare after stripping tz.
    if (a.tzinfo is None) != (b.tzinfo is None):
        a = a.replace(tzinfo=None)
        b = b.replace(tzinfo=None)

    return (a - b).total_seconds()


def _window_indices_for_target(
    history: Sequence[Mapping[str, Any]],
    target_index: int,
) -> List[int]:
    """
    Return indices belonging to the 5-minute window ending at target event.

    Only events at or before the target are used; future events do not leak into
    a historical simulation.
    """
    target_time = _event_datetime(history[target_index])
    indices: List[int] = []

    for i in range(target_index + 1):
        event_time = _event_datetime(history[i])
        age = _seconds_between(target_time, event_time)
        if 0 <= age <= SEQUENCE_WINDOW_SECONDS:
            indices.append(i)

    if len(indices) > MAX_SEQUENCE_LENGTH:
        indices = indices[-MAX_SEQUENCE_LENGTH:]

    return indices


def _scale_lstm_sequence(sequence: np.ndarray, scaler: Any) -> np.ndarray:
    """
    Scale only features 4..14. Categorical IDs 0..3 stay untouched.
    """
    if sequence.ndim != 2 or sequence.shape[1] != EXPECTED_FEATURES:
        raise ValueError(
            f"Sequenza BiLSTM con shape {sequence.shape}; "
            f"atteso (timesteps, {EXPECTED_FEATURES})."
        )

    result = sequence.astype(np.float32, copy=True)
    numeric = result[:, NUMERIC_START_INDEX:]

    if not hasattr(scaler, "transform"):
        raise TypeError(
            "Lo scaler risolto non espone .transform(): "
            f"{type(scaler).__name__}"
        )

    transformed = scaler.transform(numeric)
    result[:, NUMERIC_START_INDEX:] = np.asarray(transformed, dtype=np.float32)

    return result


def _model_expected_timesteps(model: Any) -> int | None:
    shape = getattr(model, "input_shape", None)

    if isinstance(shape, list):
        shape = shape[0] if shape else None

    if not shape or len(shape) < 3:
        return None

    timesteps = shape[1]
    if timesteps is None:
        return None

    try:
        return int(timesteps)
    except (TypeError, ValueError):
        return None


def _pad_sequence_if_needed(sequence: np.ndarray, model: Any) -> np.ndarray:
    expected = _model_expected_timesteps(model)

    if expected is None:
        return sequence

    if sequence.shape[0] > expected:
        # Preserve the most recent events because the target is the last event.
        return sequence[-expected:]

    if sequence.shape[0] == expected:
        return sequence

    pad_rows = expected - sequence.shape[0]
    padding = np.zeros((pad_rows, sequence.shape[1]), dtype=np.float32)

    if PADDING_SIDE == "pre":
        return np.vstack([padding, sequence])
    if PADDING_SIDE == "post":
        return np.vstack([sequence, padding])

    raise ValueError("PADDING_SIDE deve essere 'pre' oppure 'post'.")


def build_latest_online_sequence(
    raw_history: Sequence[Mapping[str, Any]],
    model_feature_matrix: np.ndarray,
    target_index: int,
    lstm_model: Any,
) -> Tuple[np.ndarray | None, List[int]]:
    """
    Build the latest temporal sequence from a matrix that is ALREADY in the
    representation expected by the frozen BiLSTM (numeric features scaled once).
    """
    indices = _window_indices_for_target(raw_history, target_index)

    if len(indices) < MIN_SEQUENCE_LENGTH:
        return None, indices

    sequence = model_feature_matrix[indices].astype(np.float32, copy=True)
    sequence = _pad_sequence_if_needed(sequence, lstm_model)

    return sequence, indices


# =============================================================================
# BILSTM + DQN
# =============================================================================

def _extract_attack_probability(prediction: Any) -> float:
    arr = np.asarray(prediction, dtype=np.float32).reshape(-1)

    if arr.size == 0:
        raise ValueError("BiLSTM ha restituito una predizione vuota.")

    if arr.size == 1:
        p = float(arr[0])
    elif arr.size == 2:
        # Common binary softmax output: [P(normal), P(attack)].
        p = float(arr[1])
    else:
        raise ValueError(
            f"Output BiLSTM inatteso con {arr.size} valori. "
            "Atteso sigmoid singolo o softmax binario."
        )

    return float(np.clip(p, 0.0, 1.0))


def predict_attack_probability(lstm_model: Any, sequence: np.ndarray) -> float:
    batch = sequence[np.newaxis, ...]
    prediction = lstm_model.predict(batch, verbose=0)
    return _extract_attack_probability(prediction)


def _failure_ratio_for_window(
    raw_history: Sequence[Mapping[str, Any]],
    window_indices: Sequence[int],
) -> float:
    if not window_indices:
        return 0.0

    failed = sum(_extract_failed(raw_history[i]) for i in window_indices)
    return float(failed / len(window_indices))


def build_dqn_state(
    attack_probability: float,
    failure_ratio: float,
    context: Mapping[str, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        raw_state        : human-readable values before DQN normalization
        normalized_state : final 9-D input passed to DQN
    """
    raw_state = np.asarray(
        [
            attack_probability,
            failure_ratio,
            context["requests_5m"],
            context["hour"],
            context["is_night"],
            context["privileged"],
            context["critical_resource"],
            context["has_certificate"],
            context["historical_risk"],
        ],
        dtype=np.float32,
    )

    if raw_state.size != EXPECTED_RL_STATE:
        raise AssertionError("RL state interno non ha 9 feature.")

    state = raw_state.copy()

    # IMPORTANT: failure_ratio is already in [0, 1].
    # Do NOT divide state[1] by 1000.
    state[2] /= REQUESTS_SCALE
    state[3] /= 23.0

    # Do NOT divide requests_5m by 1000.
    state = np.clip(state, 0.0, 1.0)

    return raw_state, state.astype(np.float32)


def predict_dqn_action(
    dqn_model: Any,
    normalized_state: np.ndarray,
) -> Tuple[np.ndarray, int]:
    batch = normalized_state[np.newaxis, ...]
    output = dqn_model.predict(batch, verbose=0)
    q_values = np.asarray(output, dtype=np.float32).reshape(-1)

    if q_values.size != len(ACTION_NAMES):
        raise ValueError(
            f"Il DQN ha restituito {q_values.size} output; "
            f"attesi {len(ACTION_NAMES)}."
        )

    action = int(np.argmax(q_values))
    return q_values, action



def _cold_start_request_pressure(requests_5m: float) -> float:
    """
    Compress requests_5m into [0, 1] using a logarithmic scale.

    A plain requests_5m / 4486 ratio would make moderate bursts contribute
    almost nothing. Log scaling preserves sensitivity to lower-but-relevant
    request volumes while still using the TRAIN P99 (4486) as the cap.
    """
    value = max(0.0, _safe_float(requests_5m))
    if value <= 0.0:
        return 0.0

    pressure = np.log1p(value) / np.log1p(REQUESTS_SCALE)
    return float(np.clip(pressure, 0.0, 1.0))


def contextual_cold_start_policy(
    context: Mapping[str, float],
    failure_ratio: float,
) -> Dict[str, Any]:
    """
    Interpretable fallback policy used ONLY when no valid BiLSTM sequence exists.

    It does not use attack_probability and it does not call the DQN.

    Possible temporary actions are intentionally capped at BLOCK_SESSION:
        ALLOW, MONITOR, MFA, BLOCK_SESSION

    BLOCK_IP and QUARANTINE remain reserved for the full BiLSTM + DQN path,
    where stronger temporal evidence is available.
    """
    failed = float(np.clip(_safe_float(context.get("failed")), 0.0, 1.0))
    failure_ratio = float(np.clip(_safe_float(failure_ratio), 0.0, 1.0))
    is_night = float(np.clip(_safe_float(context.get("is_night")), 0.0, 1.0))
    privileged = float(np.clip(_safe_float(context.get("privileged")), 0.0, 1.0))
    critical = float(
        np.clip(_safe_float(context.get("critical_resource")), 0.0, 1.0)
    )
    has_certificate = float(
        np.clip(_safe_float(context.get("has_certificate")), 0.0, 1.0)
    )
    historical_risk = float(
        np.clip(_safe_float(context.get("historical_risk")), 0.0, 1.0)
    )
    requests_5m = max(0.0, _safe_float(context.get("requests_5m")))
    request_pressure = _cold_start_request_pressure(requests_5m)

    signals = {
        "failed": failed,
        "failure_ratio": failure_ratio,
        "is_night": is_night,
        "privileged": privileged,
        "critical_resource": critical,
        "no_certificate": 1.0 - has_certificate,
        "historical_risk": historical_risk,
        "request_pressure": request_pressure,
    }

    components = {
        name: COLD_START_WEIGHTS[name] * value
        for name, value in signals.items()
    }

    score = float(np.clip(sum(components.values()), 0.0, 1.0))

    if score < COLD_START_THRESHOLDS["ALLOW_MAX"]:
        action = "ALLOW"
    elif score < COLD_START_THRESHOLDS["MONITOR_MAX"]:
        action = "MONITOR"
    elif score < COLD_START_THRESHOLDS["MFA_MAX"]:
        action = "MFA"
    else:
        action = "BLOCK_SESSION"

    reasons: List[str] = []

    if failed >= 0.5:
        reasons.append("current event failed")
    if failure_ratio >= 0.25:
        reasons.append(f"failure ratio={failure_ratio:.2f}")
    if is_night >= 0.5:
        reasons.append("night-time activity")
    if privileged >= 0.5:
        reasons.append("privileged user")
    if critical >= 0.5:
        reasons.append("critical resource")
    if has_certificate < 0.5:
        reasons.append("certificate absent")
    if historical_risk >= 0.25:
        reasons.append(f"historical risk={historical_risk:.2f}")
    if request_pressure >= 0.35:
        reasons.append(
            f"request pressure={request_pressure:.2f} "
            f"(requests_5m={requests_5m:.0f})"
        )

    if not reasons:
        reasons.append("no strong contextual risk signal")

    return {
        "score": score,
        "action": action,
        "signals": signals,
        "components": components,
        "reasons": reasons,
        "request_pressure": request_pressure,
    }


def _previous_event_gap_seconds(
    causal_history: Sequence[Mapping[str, Any]],
    target_index: int,
) -> float | None:
    if target_index <= 0:
        return None

    current_dt = _event_datetime(causal_history[target_index])
    previous_dt = _event_datetime(causal_history[target_index - 1])
    gap = _seconds_between(current_dt, previous_dt)
    return max(0.0, float(gap))


# =============================================================================
# ONLINE ENGINE
# =============================================================================

class OnlineCyberDefense:
    def __init__(self, artifacts: FrozenArtifacts) -> None:
        self.artifacts = artifacts
        self.history: MutableMapping[Tuple[str, str], List[Dict[str, Any]]] = (
            defaultdict(list)
        )

    def preload(self, events: Iterable[Mapping[str, Any]]) -> None:
        count = 0
        for raw_event in events:
            event = validate_event(raw_event)
            key = _entity_key(event)
            self.history[key].append(event)
            count += 1

        for key in self.history:
            self.history[key].sort(key=_timestamp_sort_key)

        print(
            f"History inizializzata: {count} eventi, "
            f"{len(self.history)} entità (user, ip)."
        )

    def known_entities(self) -> List[Tuple[str, str]]:
        return sorted(self.history.keys())

    def _find_target_index(
        self,
        entity_history: Sequence[Mapping[str, Any]],
        online_event_id: str,
    ) -> int:
        for i, item in enumerate(entity_history):
            if item.get("_online_event_id") == online_event_id:
                return i
        raise RuntimeError("Impossibile ritrovare il nuovo evento nella history.")

    def analyze_new_event(self, raw_event: Mapping[str, Any]) -> Dict[str, Any]:
        event = validate_event(raw_event)
        key = _entity_key(event)
        existed_before = key in self.history and len(self.history[key]) > 0

        entity_history = self.history[key]
        entity_history.append(event)
        entity_history.sort(key=_timestamp_sort_key)

        target_index = self._find_target_index(
            entity_history,
            event["_online_event_id"],
        )

        # No future information enters feature engineering for the target event.
        # This is especially important when simulating an event with an old timestamp.
        causal_history = entity_history[: target_index + 1]

        feature_result = _run_feature_engineering(
            causal_history,
            encoders=self.artifacts.encoders,
        )

        raw_feature_matrix, model_feature_matrix, feature_adapter_info = (
            _prepare_feature_matrices(
                feature_result=feature_result,
                causal_history=causal_history,
                scaler=self.artifacts.scaler,
            )
        )

        # feature_engineering.py is now fed the same flat schema as offline,
        # so its 15-D output is the source of truth. No duplicate online
        # recalculation of failed/historical_risk is necessary.
        raw_target_vector = raw_feature_matrix[target_index]
        context = _context_from_enriched(raw_target_vector)

        # Two deliberately separate notions of criticality:
        #
        # 1) model_critical_resource
        #    Frozen training-time definition used by BiLSTM + DQN.
        #
        # 2) contextual_critical_resource
        #    Extended online-only definition used by the cold-start policy.
        model_critical_resource = int(round(context["critical_resource"]))
        contextual_critical_resource = _contextual_check_critical_resource(
            event.get("resource")
        )

        sequence, window_indices = build_latest_online_sequence(
            raw_history=causal_history,
            model_feature_matrix=model_feature_matrix,
            target_index=target_index,
            lstm_model=self.artifacts.lstm,
        )

        failure_ratio = _failure_ratio_for_window(
            causal_history,
            window_indices,
        )

        previous_gap_seconds = _previous_event_gap_seconds(
            causal_history,
            target_index,
        )

        result: Dict[str, Any] = {
            "entity_key": key,
            "entity_status": "EXISTING" if existed_before else "NEW",
            "history_length": len(entity_history),
            "causal_history_length": len(causal_history),
            "window_length": len(window_indices),
            "target_timestamp": _event_datetime(event),
            "context": context,
            "event_type": event.get("event_type"),
            "resource": event.get("resource"),
            "model_critical_resource": model_critical_resource,
            "contextual_critical_resource": contextual_critical_resource,
            "failure_ratio": failure_ratio,
            "mode": "COLD START" if sequence is None else "FULL ANALYSIS",
            "feature_adapter_info": feature_adapter_info,
            "previous_gap_seconds": previous_gap_seconds,
        }

        if sequence is None:
            if not existed_before:
                cold_reason = "NEW_ENTITY"
            elif previous_gap_seconds is not None and previous_gap_seconds > SEQUENCE_WINDOW_SECONDS:
                cold_reason = "STALE_HISTORY"
            else:
                cold_reason = "INSUFFICIENT_RECENT_HISTORY"

            cold_context = dict(context)
            cold_context["critical_resource"] = float(
                contextual_critical_resource
            )

            cold_policy = contextual_cold_start_policy(
                context=cold_context,
                failure_ratio=failure_ratio,
            )

            result.update(
                {
                    "cold_start_reason": cold_reason,
                    "cold_start_score": cold_policy["score"],
                    "temporary_policy": cold_policy["action"],
                    "cold_start_reasons": cold_policy["reasons"],
                    "cold_start_components": cold_policy["components"],
                    "cold_start_signals": cold_policy["signals"],
                    "request_pressure": cold_policy["request_pressure"],
                }
            )
            return result

        attack_probability = predict_attack_probability(
            self.artifacts.lstm,
            sequence,
        )

        raw_state, normalized_state = build_dqn_state(
            attack_probability=attack_probability,
            failure_ratio=failure_ratio,
            context=context,
        )

        q_values, action = predict_dqn_action(
            self.artifacts.dqn,
            normalized_state,
        )

        result.update(
            {
                "attack_probability": attack_probability,
                "raw_rl_state": raw_state,
                "normalized_rl_state": normalized_state,
                "q_values": q_values,
                "action_id": action,
                "decision": ACTION_NAMES[action],
            }
        )

        return result

    def show_entity_history(self, key: Tuple[str, str]) -> None:
        events = self.history.get(key, [])
        if not events:
            print("Nessun evento per questa entità.")
            return

        _banner(f"HISTORY {key[0]} / {key[1]}")
        for idx, event in enumerate(events, start=1):
            ts = _event_datetime(event).isoformat(sep=" ")
            failed = _extract_failed(event)
            print(f"{idx:4d} | {ts:<26} | failed={failed}")



# =============================================================================
# DECISION AUDIT TRAIL
# =============================================================================

def _audit_json_value(value: Any) -> Any:
    """Convert NumPy/datetime values into JSON-safe plain Python values."""
    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, np.ndarray):
        return [_audit_json_value(item) for item in value.tolist()]

    if isinstance(value, (np.floating,)):
        return float(value)

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, Mapping):
        return {
            str(key): _audit_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_audit_json_value(item) for item in value]

    return value


def build_audit_record(result: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Build an append-only audit record AFTER the decision has already been made.

    This function does not affect feature engineering, BiLSTM, DQN, cold-start
    scoring, thresholds, or actions.
    """
    user, ip = result["entity_key"]
    context = result["context"]
    mode = str(result["mode"])

    if mode == "COLD START":
        effective_critical_resource = int(
            result.get(
                "contextual_critical_resource",
                context["critical_resource"],
            )
        )
        action = result["temporary_policy"]
        decision_source = "RULE_BASED_COLD_START"

    else:
        effective_critical_resource = int(
            round(context["critical_resource"])
        )
        action = result["decision"]
        decision_source = "DQN"

    record: Dict[str, Any] = {
        "decision_id": str(uuid.uuid4()),
        "decision_time": datetime.now().astimezone().isoformat(),
        "event_time": result["target_timestamp"].isoformat(),
        "user": user,
        "ip": ip,
        "entity_status": result["entity_status"],
        "mode": mode,
        "event_type": result.get("event_type"),
        "resource": result.get("resource"),

        # Context used to explain the decision
        "failed": int(round(context["failed"])),
        "failure_ratio": float(result["failure_ratio"]),
        "requests_5m": float(context["requests_5m"]),
        "hour": int(round(context["hour"])),
        "is_night": int(round(context["is_night"])),
        "privileged": int(round(context["privileged"])),
        "critical_resource": effective_critical_resource,
        "has_certificate": int(round(context["has_certificate"])),
        "historical_risk": float(context["historical_risk"]),

        # Keep the two definitions visible for auditability.
        "model_critical_resource": int(
            result.get(
                "model_critical_resource",
                round(context["critical_resource"]),
            )
        ),
        "contextual_critical_resource": int(
            result.get(
                "contextual_critical_resource",
                round(context["critical_resource"]),
            )
        ),

        "action": action,
        "decision_source": decision_source,
    }

    if mode == "COLD START":
        record.update(
            {
                "cold_start_reason": result.get("cold_start_reason"),
                "cold_start_score": float(result["cold_start_score"]),
                "request_pressure": float(
                    result.get("request_pressure", 0.0)
                ),
                "score_components": result.get(
                    "cold_start_components",
                    {},
                ),
                "risk_signals": result.get(
                    "cold_start_reasons",
                    [],
                ),
                "attack_probability": None,
                "rl_state": None,
                "dqn_q_values": None,
            }
        )

    else:
        record.update(
            {
                "attack_probability": float(
                    result["attack_probability"]
                ),
                "raw_rl_state": result["raw_rl_state"],
                "normalized_rl_state": result[
                    "normalized_rl_state"
                ],
                "dqn_q_values": result["q_values"],
                "action_id": int(result["action_id"]),
            }
        )

    return _audit_json_value(record)


def append_decision_audit(
    result: Mapping[str, Any],
    path: Path = AUDIT_LOG_PATH,
) -> Path:
    """
    Append exactly one JSON object per analyzed event.

    JSONL is used so the existing audit file never needs to be rewritten.
    """
    record = build_audit_record(result)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        )

    return path


# =============================================================================
# OUTPUT
# =============================================================================

def print_analysis(result: Mapping[str, Any]) -> None:
    user, ip = result["entity_key"]
    context = result["context"]

    _banner("ONLINE CYBER DEFENSE")
    print(f"User            : {user}")
    print(f"IP              : {ip}")
    print(f"Event timestamp : {result['target_timestamp']}")
    print()
    print(f"Entity status   : {result['entity_status']}")
    print(f"History length  : {result['history_length']}")
    print(f"5m window events: {result['window_length']}")
    print(f"Mode            : {result['mode']}")
    print(f"Feature source  : {result.get('feature_adapter_info', 'n/a')}")
    print()

    print("Temporal / context features")
    print("-" * 62)
    print(f"Failed           : {int(round(context['failed']))}")
    print(f"Failure ratio    : {result['failure_ratio']:.4f}")
    print(f"Requests 5m      : {context['requests_5m']:.4f}")
    print(f"Historical risk  : {context['historical_risk']:.4f}")
    print(f"Hour             : {context['hour']:.0f}")
    print(f"Is night         : {context['is_night']:.0f}")
    print(f"Privileged       : {context['privileged']:.0f}")
    if result["mode"] == "COLD START":
        displayed_critical_resource = int(
            result.get(
                "contextual_critical_resource",
                context["critical_resource"],
            )
        )
    else:
        displayed_critical_resource = int(
            round(context["critical_resource"])
        )

    print(f"Critical resource: {displayed_critical_resource}")
    print(f"Has certificate  : {context['has_certificate']:.0f}")

    if result["mode"] == "COLD START":
        print()
        print("BiLSTM: NOT EXECUTED")
        print("DQN   : NOT EXECUTED")
        print()
        print(
            "Insufficient temporal history for a valid BiLSTM sequence "
            f"(minimum {MIN_SEQUENCE_LENGTH} events inside "
            f"{SEQUENCE_WINDOW_SECONDS // 60} minutes)."
        )

        reason_labels = {
            "NEW_ENTITY": "new (user, ip) entity",
            "STALE_HISTORY": "existing entity, but previous activity is outside the 5m window",
            "INSUFFICIENT_RECENT_HISTORY": "not enough recent events inside the 5m window",
        }

        print(
            f"Cold-start reason : "
            f"{reason_labels.get(result.get('cold_start_reason'), result.get('cold_start_reason'))}"
        )

        gap = result.get("previous_gap_seconds")
        if gap is not None:
            print(f"Previous event gap: {gap:.1f} seconds")

        print()
        print("Contextual cold-start policy")
        print("-" * 62)
        print(f"Context risk score: {result['cold_start_score']:.4f}")
        print(f"Request pressure  : {result.get('request_pressure', 0.0):.4f}")
        print(f"Temporary decision: {result['temporary_policy']}")
        print("Decision source   : RULE-BASED COLD-START POLICY, not DQN")

        print()
        print("Risk signals")
        print("-" * 62)
        for reason in result.get("cold_start_reasons", []):
            print(f"- {reason}")

        print()
        print("Score components")
        print("-" * 62)
        component_order = (
            "failed",
            "failure_ratio",
            "is_night",
            "privileged",
            "critical_resource",
            "no_certificate",
            "historical_risk",
            "request_pressure",
        )
        components = result.get("cold_start_components", {})
        for name in component_order:
            print(f"{name:<20}: {float(components.get(name, 0.0)):.4f}")

        return

    print()
    print(
        f"BiLSTM attack probability: "
        f"{result['attack_probability']:.6f}"
    )

    print()
    print("DQN Q-values")
    print("-" * 62)
    q_values = result["q_values"]
    for idx, name in enumerate(ACTION_NAMES):
        print(f"{idx} {name:<15} {float(q_values[idx]): .6f}")

    print()
    print("Normalized RL state passed to DQN")
    print("-" * 62)
    state_names = (
        "attack_probability",
        "failure_ratio",
        "requests_5m_scaled",
        "hour_scaled",
        "is_night",
        "privileged",
        "critical_resource",
        "has_certificate",
        "historical_risk",
    )
    for name, value in zip(state_names, result["normalized_rl_state"]):
        print(f"{name:<24}: {float(value):.6f}")

    print()
    print(f"DECISION: {result['decision']} (action={result['action_id']})")


# =============================================================================
# INTERACTIVE MENUS
# =============================================================================

def choose_initial_history(engine: OnlineCyberDefense) -> None:
    _banner("ONLINE CYBER DEFENSE - INITIALIZATION")
    print("1 - New empty session")
    print("2 - Load logsprova.json as initial history")

    while True:
        choice = input("\nChoice: ").strip()

        if choice == "1":
            print("Nuova sessione vuota inizializzata.")
            return

        if choice == "2":
            path_text = input(
                f"JSON path [{DEFAULT_LOGS_PATH.relative_to(PROJECT_ROOT)}]: "
            ).strip()

            path = (
                Path(path_text).expanduser()
                if path_text
                else DEFAULT_LOGS_PATH
            )

            if not path.is_absolute():
                path = PROJECT_ROOT / path

            historical_events = load_events_from_json(path)
            persistent_online_events = load_online_events_jsonl(
                ONLINE_EVENTS_PATH
            )

            combined_events = (
                historical_events
                + persistent_online_events
            )

            if persistent_online_events:
                print(
                    "History combinata: "
                    f"{len(historical_events)} eventi storici + "
                    f"{len(persistent_online_events)} eventi online persistiti "
                    f"= {len(combined_events)} eventi utilizzabili."
                )

            engine.preload(combined_events)
            return

        print("Scelta non valida. Inserisci 1 oppure 2.")


def show_known_entities(engine: OnlineCyberDefense) -> None:
    entities = engine.known_entities()
    _banner("KNOWN ENTITIES")

    if not entities:
        print("Nessuna entità in memoria.")
        return

    for idx, key in enumerate(entities, start=1):
        print(
            f"{idx:4d} | user={key[0]:<20} | ip={key[1]:<18} | "
            f"events={len(engine.history[key])}"
        )


def choose_entity(engine: OnlineCyberDefense) -> Tuple[str, str] | None:
    entities = engine.known_entities()
    if not entities:
        print("Nessuna entità in memoria.")
        return None

    show_known_entities(engine)
    raw = input("\nNumero entità (ENTER per annullare): ").strip()

    if not raw:
        return None

    try:
        idx = int(raw)
    except ValueError:
        print("Indice non valido.")
        return None

    if not 1 <= idx <= len(entities):
        print("Indice fuori intervallo.")
        return None

    return entities[idx - 1]


def interactive_loop(engine: OnlineCyberDefense) -> None:
    while True:
        _banner("ONLINE CYBER DEFENSE")
        print(f"Module file      : {Path(__file__).resolve()}")
        print("1 - Insert new event")
        print("2 - Show known entities")
        print("3 - Show entity history")
        print("4 - Exit")

        choice = input("\nChoice: ").strip()

        if choice == "1":
            try:
                raw_event = read_event_from_console()
                result = engine.analyze_new_event(raw_event)

                try:
                    online_event_path = append_online_raw_event(raw_event)
                except (OSError, TypeError, ValueError) as event_store_exc:
                    online_event_path = None
                    print(
                        "\n[WARNING] Decisione calcolata, ma evento raw "
                        "non persistito:"
                    )
                    print(
                        f"  {type(event_store_exc).__name__}: "
                        f"{event_store_exc}"
                    )

                try:
                    audit_path = append_decision_audit(result)
                except OSError as audit_exc:
                    audit_path = None
                    print(
                        "\n[WARNING] Decisione calcolata, ma audit non salvato:"
                    )
                    print(
                        f"  {type(audit_exc).__name__}: {audit_exc}"
                    )

                print_analysis(result)

                print()
                if online_event_path is not None:
                    print(f"Event saved     : {online_event_path}")
                if audit_path is not None:
                    print(f"Audit saved     : {audit_path}")
            except KeyboardInterrupt:
                print("\nInserimento annullato.")
            except Exception as exc:
                print("\n[ERROR] Evento non analizzato:")
                print(f"  {type(exc).__name__}: {exc}")

        elif choice == "2":
            show_known_entities(engine)

        elif choice == "3":
            key = choose_entity(engine)
            if key is not None:
                engine.show_entity_history(key)

        elif choice == "4":
            print("Online Cyber Defense terminato.")
            return

        else:
            print("Scelta non valida.")


# =============================================================================
# MAIN
# =============================================================================

def main() -> int:
    try:
        _banner("ONLINE CYBER DEFENSE")
        print("Frozen-model interactive inference mode")
        print("Raw/scaled guard : automatic")
        print("Feature pipeline : raw JSON -> create_logs-equivalent flatten -> feature_engineering")
        print("Identity parser  : multi-schema, blank-aware fallback")
        print("Risk rule        : delegated to original feature_engineering.py")
        print("Cold-start policy: contextual rule-based")
        print("Cold criticality : extended taxonomy, isolated from frozen models")
        print("Feature adapter  : auto-detect 15-D output")
        print("No training / no encoder fit / no model modification")
        print(f"Online event store: {ONLINE_EVENTS_PATH}")
        print(f"Decision audit    : {AUDIT_LOG_PATH}\n")

        artifacts = load_frozen_artifacts()
        engine = OnlineCyberDefense(artifacts)

        choose_initial_history(engine)
        interactive_loop(engine)

        return 0

    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.")
        return 130
    except Exception as exc:
        print("\n[FATAL ERROR]")
        print(f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())