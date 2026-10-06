from collections import defaultdict
from datetime import datetime, timedelta


SESSION_TIMEOUT_MINUTES = 30
CORRELATION_WINDOW_MINUTES = 5


# ============================================================
# TIME PARSER
# ============================================================

def parse_time(ts):

    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None)

    formats = [
        "%d-%b-%Y %H:%M:%S.%f",
        "%d-%b-%Y %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S"
    ]


    for fmt in formats:

        try:
            return datetime.strptime(ts, fmt)

        except:
            pass


    try:

        return datetime.fromisoformat(ts).replace(
            tzinfo=None
        )

    except Exception:

        raise ValueError(
            f"Formato timestamp non riconosciuto: {ts}"
        )



# ============================================================
# MAIN CORRELATION ENGINE
# ============================================================

def correlate_logs(events):


    p0f_by_ip = defaultdict(list)

    captive_by_ip = defaultdict(list)

    application_events = []



    # ========================================================
    # 1) DIVISIONE SORGENTI
    # ========================================================

    for event in events:


        source = event.get("source")

        message = event.get(
            "message",
            {}
        )


        timestamp = parse_time(
            event["time"]
        )



        # -------------------------------
        # P0F
        # -------------------------------

        if source == "p0f":


            for ip, info in message.items():


                if not isinstance(info, dict):
                    continue



                p0f_by_ip[ip].append({

                    "time": timestamp,

                    "mac": info.get(
                        "mac"
                    ),

                    "hostname": info.get(
                        "hostname"
                    ),

                    "os": info.get(
                        "os"
                    ),

                    "first_seen": info.get(
                        "first_seen"
                    ),

                    "last_seen": info.get(
                        "last_seen"
                    )

                })




        # -------------------------------
        # Captive Portal
        # -------------------------------

        elif source == "captiveportal":


            ip = message.get(
                "ip"
            )


            if ip:

                captive_by_ip[ip].append({

                    "time": timestamp,

                    "event": message

                })




        # -------------------------------
        # Application Logs
        # -------------------------------

        elif (
            "ApplicationAudit" in source
            or
            "AuthAudit" in source
        ):


            application_events.append({

                "time": timestamp,

                "source": source,

                "event": message

            })



    # ========================================================
    # 2) GROUP USER + IP
    # ========================================================


    sessions = defaultdict(list)



    for event in application_events:


        data = event["event"]


        user = data.get(
            "User",
            "UNKNOWN"
        )


        ip = data.get(
            "IP"
        )


        if not ip:
            continue



        sessions[
            (user, ip)
        ].append(event)



    # ========================================================
    # 3) CREAZIONE SESSIONI
    # ========================================================


    output = []

    session_id = 0



    for (user, ip), user_events in sessions.items():


        user_events.sort(
            key=lambda x:x["time"]
        )


        current = []

        user_sessions = []



        for event in user_events:


            if not current:

                current.append(event)

                continue



            delta = (
                event["time"]
                -
                current[-1]["time"]
            ).total_seconds()



            if delta > SESSION_TIMEOUT_MINUTES*60:


                user_sessions.append(
                    current
                )

                current = [
                    event
                ]

            else:

                current.append(
                    event
                )



        if current:

            user_sessions.append(
                current
            )



        # ====================================================
        # 4) ARRICCHIMENTO SESSIONE
        # ====================================================


        for session in user_sessions:


            session_id += 1


            start = session[0]["time"]

            end = session[-1]["time"]



            correlated_p0f = []


            for device in p0f_by_ip.get(
                ip,
                []
            ):


                if (
                    start - timedelta(
                        minutes=CORRELATION_WINDOW_MINUTES
                    )
                    <= device["time"]
                    <=
                    end + timedelta(
                        minutes=CORRELATION_WINDOW_MINUTES
                    )
                ):

                    correlated_p0f.append(
                        device
                    )




            correlated_captive = []


            for captive in captive_by_ip.get(
                ip,
                []
            ):


                if (
                    start - timedelta(
                        minutes=CORRELATION_WINDOW_MINUTES
                    )
                    <= captive["time"]
                    <=
                    end + timedelta(
                        minutes=CORRELATION_WINDOW_MINUTES
                    )
                ):


                    correlated_captive.append(
                        captive
                    )




            output.append({

                "SessionId":
                    f"S{session_id}",


                "User":
                    user,


                "IP":
                    ip,



                "TimeWindow": {

                    "start":
                        start.isoformat(),

                    "end":
                        end.isoformat()

                },



                "Events":[

                    {

                    "source":
                        e["source"],


                    "time":
                        e["time"].isoformat(),


                    "event":
                        e["event"]

                    }

                    for e in session

                ],



                "p0f":
                    correlated_p0f,



                "captive":
                    correlated_captive

            })



    return output