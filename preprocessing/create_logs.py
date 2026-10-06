from datetime import datetime
import json



# ============================================================
# DEFAULT RESOURCE CRITICALITY
# ============================================================

CRITICAL_RESOURCES = {

    "AspNetUsers",
    "Users",
    "Identity",
    "Authentication",
    "Credentials",
    "Security",
    "Admin"

}



# ============================================================
# TIME NORMALIZATION
# ============================================================

def parse_event_time(ts):

    if isinstance(ts, datetime):
        return ts.replace(
            tzinfo=None
        )


    try:

        return datetime.fromisoformat(
            ts
        ).replace(
            tzinfo=None
        )

    except:


        formats = [

            "%Y-%m-%dT%H:%M:%S",

            "%d-%b-%Y %H:%M:%S"

        ]


        for fmt in formats:

            try:

                return datetime.strptime(
                    ts,
                    fmt
                )

            except:

                pass



    raise ValueError(
        f"Timestamp non valido: {ts}"
    )



# ============================================================
# ROLE EXTRACTION
# ============================================================

def extract_role(event):


    context = event.get(
        "UserContext",
        {}
    )


    roles = context.get(
        "roles",
        []
    )


    if roles:

        return roles[-1]


    return "User"



# ============================================================
# PRIVILEGED USER
# ============================================================

def is_privileged(role):


    privileged_roles = {

        "Admin",

        "Administrator",

        "SuperAdmin",

        "SecurityAdmin"

    }


    return int(
        role in privileged_roles
    )



# ============================================================
# RESOURCE EXTRACTION
# ============================================================

def extract_resource(event):


    resource_context = event.get(
        "ResourceContext",
        {}
    )


    if isinstance(
        resource_context,
        dict
    ):

        return resource_context.get(
            "table",
            resource_context.get(
                "database",
                "UNKNOWN"
            )
        )


    return "UNKNOWN"



# ============================================================
# CRITICAL RESOURCE
# ============================================================

def check_critical_resource(resource):


    if not resource:

        return 0


    resource = str(
        resource
    ).lower()



    for item in CRITICAL_RESOURCES:

        if item.lower() in resource:

            return 1



    return 0



# ============================================================
# DEVICE EXTRACTION
# ============================================================

def extract_device_info(
        session,
        p0f
):


    device = {

        "mac":None,

        "hostname":None,

        "os":None

    }



    if p0f:


        first = p0f[0]


        device["mac"] = first.get(
            "mac"
        )


        device["hostname"] = first.get(
            "hostname"
        )


        device["os"] = first.get(
            "os"
        )



    return device



# ============================================================
# USER RISK INITIAL VERSION
# ============================================================

def calculate_initial_user_risk(event):


    risk = 0.0



    status = event.get(
        "HTTPStatusCode",
        200
    )


    suspicious = event.get(
        "Suspicious",
        False
    )


    if suspicious:

        risk += 0.5



    if status >=400:

        risk +=0.2



    if status ==429:

        risk +=0.3



    return min(
        risk,
        1.0
    )



# ============================================================
# MAIN FUNCTION
# ============================================================

def create_logs(
        grouped_logs
):


    flat_logs=[]



    for session in grouped_logs:


        user = session.get(
            "User"
        )


        ip = session.get(
            "IP"
        )


        p0f = session.get(
            "p0f",
            []
        )


        device = extract_device_info(
            session,
            p0f
        )



        events = session.get(
            "Events",
            []
        )



        for item in events:


            event = item.get(
                "event",
                {}
            )



            event_time = parse_event_time(
                item.get(
                    "time"
                )
            )



            role = extract_role(
                event
            )



            resource = extract_resource(
                event
            )



            fingerprint = event.get(
                "Fingerprint",
                {}
            )



            certificate = event.get(
                "ClientCertificate"
            )



            route = event.get(
                "Route",
                "UNKNOWN"
            )



            flat_logs.append({

                # --------------------
                # Identity
                # --------------------

                "user":
                    user,


                "ip":
                    ip,



                # --------------------
                # Event
                # --------------------

                "event_time":
                    event_time,


                "event_type":
                    event.get(
                        "EventType",
                        "UNKNOWN"
                    ),



                "source":
                    item.get(
                        "source"
                    ),



                "status_code":
                    event.get(
                        "HTTPStatusCode",
                        200
                    ),



                "suspicious":
                    int(
                        event.get(
                            "Suspicious",
                            False
                        )
                    ),



                # --------------------
                # Context
                # --------------------

                "role":
                    role,


                "privileged":
                    is_privileged(
                        role
                    ),



                "resource":
                    resource,


                "critical_resource":
                    check_critical_resource(
                        resource
                    ),



                "certificate":
                    certificate,



                # --------------------
                # Device
                # --------------------

                "os":
                    device["os"],


                "mac":
                    device["mac"],


                "hostname":
                    device["hostname"],



                "fingerprint":
                    fingerprint,



                # --------------------
                # Application
                # --------------------

                "route":
                    route,



                # --------------------
                # Risk
                # --------------------

                "user_risk":
                    calculate_initial_user_risk(
                        event
                    )

            })



    return flat_logs



# ============================================================
# SAVE
# ============================================================

def save_logs_to_json(
        logs,
        path="flat_logs.json"
):


    serializable=[]



    for log in logs:


        item=log.copy()



        item["event_time"] = (
            item["event_time"]
            .isoformat()
        )


        serializable.append(
            item
        )



    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:


        json.dump(
            serializable,
            f,
            indent=2,
            ensure_ascii=False
        )


    return path