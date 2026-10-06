from collections import defaultdict, deque
from datetime import timedelta
import numpy as np

# ============================================================
# USER RISK PARAMETERS
# ============================================================

RISK_DECAY = 0.95

# ============================================================
# FEATURE ENGINEERING
# ============================================================

def feature_engineering(
        logs,
        encoders
):


    # ordinamento temporale globale

    logs = sorted(
        logs,
        key=lambda x:x["event_time"]
    )

    enriched=[]

    last_event_per_entity = {}

    entity_history = defaultdict(
        deque
    )

    entity_risk_memory = defaultdict(
        float
    )


    # ========================================================
    # PROCESS EVENTS
    # ========================================================

    for log in logs:

        user = log["user"]

        ip = log.get(
            "ip"
        ) or "UNKNOWN_IP"

        entity_key = (
            user,
            ip
        )

        timestamp = log["event_time"]

        # ----------------------------------------------------
        # TEMPORAL FEATURES
        # ----------------------------------------------------

        previous = last_event_per_entity.get(
            entity_key
        )

        if previous:
            delta = (
                timestamp
                -
                previous["event_time"]
            ).total_seconds()
        else:
            delta=0

        delta = max(
            delta,
            0
        )


        delta_log = np.log1p(
            delta
        )

        hour = timestamp.hour

        weekday = timestamp.weekday()

        is_night = int( hour < 6 or hour >=22)

        # ----------------------------------------------------
        # BEHAVIOR FEATURES
        # ----------------------------------------------------

        failed = int(
            log.get("status_code",200)>=400
        )

        history = entity_history[entity_key]

        # manteniamo solo 5 minuti

        while history and (

            timestamp -
            history[0]

        ).total_seconds() > 300:


            history.popleft()


        history.append(
            timestamp
        )

        requests_5m=len(
            history
        )


        # ----------------------------------------------------
        # ENCODING
        # ----------------------------------------------------

        event_id = encoders["event"].transform_one(
            log.get(
                "event_type"
            )
        )


        source_id = encoders["source"].transform_one(
            log.get(
                "source"
            )
        )


        role_id = encoders["role"].transform_one(
            log.get(
                "role"
            )
        )


        resource_id = encoders["resource"].transform_one(
            log.get(
                "resource"
            )
        )

        # ----------------------------------------------------
        # SECURITY FEATURES
        # ----------------------------------------------------


        has_certificate = int(
            log.get(
                "certificate"
            )
            is not None
        )


        privileged = int(
            log.get(
                "privileged",
                0
            )
        )


        critical_resource = int(
            log.get(
                "critical_resource",
                0
            )
        )


        # ----------------------------------------------------
        # USER RISK UPDATE
        # ----------------------------------------------------


        previous_risk = entity_risk_memory[entity_key]

        historical_risk  = (
            previous_risk
            *
            RISK_DECAY
        )


        historical_risk = float(
            np.clip(
                historical_risk,
                0.0,
                1.0
            )
        )

        
        
        # ====================================================
        # FINAL LSTM VECTOR
        # ====================================================

        features=[

            # categorical
            event_id,
            source_id,
            role_id,
            resource_id,

            # temporal

            delta,
            delta_log,
            hour,
            weekday,
            is_night,

            # behaviour
            failed,
            requests_5m,

            # security
            has_certificate,
            privileged,
            critical_resource,

            # risk
            historical_risk
        ]


        # ====================================================
        # CONTEXT FOR RL
        # ====================================================

        context={

            "user":
                user,

            "ip":
                log.get(
                    "ip"
                ),

            "role":
                log.get(
                    "role"
                ),

            "resource":
                log.get(
                    "resource"
                ),

            "route":
                log.get(
                    "route"
                ),

            "os":
                log.get(
                    "os"
                ),

            "fingerprint":
                log.get(
                    "fingerprint"
                ),

            "critical_resource":
                critical_resource,

            "privileged":
                privileged,


            "requests_5m": requests_5m,

            "failed_requests": failed,

            "is_night": is_night,

            "user_risk": historical_risk

        }

        enriched.append({

            "user":
                user,

            "time":
                timestamp,

            "features":
                features,

            "context":
                context,

            "label":
                int(
                    log.get(
                        "suspicious",
                        0
                    )
                )
        })


        # ====================================================
        # UPDATE RISK FOR NEXT EVENT
        # ====================================================

        new_risk = historical_risk


        if failed:

            new_risk += 0.10


        if requests_5m > 948:
                    new_risk +=0.15



        new_risk = float(
            np.clip(
                new_risk,
                0.0,
                1.0
            )
        )


        entity_risk_memory[entity_key] = new_risk


        last_event_per_entity[entity_key] = log

    return enriched