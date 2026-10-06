from datetime import datetime


def parse_time(ts):
    """
    Converte tutti i formati timestamp presenti nei log.

    Supportati:
    - 22-Dec-2025 00:42:33.123
    - 22-Dec-2025 00:42:33
    - ISO format
    """

    if isinstance(ts, datetime):
        return ts.replace(tzinfo=None)


    formats = [
        "%d-%b-%Y %H:%M:%S.%f",
        "%d-%b-%Y %H:%M:%S"
    ]


    for fmt in formats:

        try:
            dt = datetime.strptime(
                ts,
                fmt
            )

            return dt.replace(
                tzinfo=None
            )

        except ValueError:
            pass


    # fallback ISO
    dt = datetime.fromisoformat(ts)

    return dt.replace(
        tzinfo=None
    )