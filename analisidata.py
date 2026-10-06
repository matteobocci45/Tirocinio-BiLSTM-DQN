"""
audit_summary.py

Analisi essenziale di un dataset JSON di audit/log.
Output sul terminale + grafici PNG professionali/minimal. Nessun CSV o JSON.

Uso:
    python audit_summary.py dataset.json
    python audit_summary_professional.py dataset.json --top 0
    python audit_summary_professional.py dataset.json --no-plots

Dipendenze:
    pip install numpy matplotlib
"""

from __future__ import annotations

import argparse
import json
import sys
from array import array
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WIDTH = 96


def section(title: str) -> None:
    print("\n" + "=" * WIDTH)
    print(title)
    print("=" * WIDTH)


def subsection(title: str) -> None:
    print(f"\n{title}")
    print("-" * min(WIDTH, max(20, len(title))))


def fmt_int(v: int) -> str:
    return f"{int(v):,}".replace(",", ".")


def fmt_float(v: float, digits: int = 3) -> str:
    return f"{float(v):,.{digits}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def fmt_pct(v: float) -> str:
    return f"{fmt_float(v, 2)}%"


def fmt_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "N/A"
    seconds = float(seconds)
    if seconds < 1:
        return f"{seconds * 1000:.3f} ms"
    if seconds < 60:
        return f"{seconds:.3f} s"
    if seconds < 3600:
        return f"{seconds / 60:.3f} min"
    if seconds < 86400:
        return f"{seconds / 3600:.3f} h"
    return f"{seconds / 86400:.3f} giorni"


def short(value: Any, length: int = 42) -> str:
    s = str(value).replace("\n", " ").replace("\r", " ")
    return s if len(s) <= length else s[: length - 3] + "..."


def clean(value: Any) -> str:
    return "" if value is None else str(value)


def parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes"}:
            return True
        if v in {"false", "0", "no"}:
            return False
    return None


def parse_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    s = str(value).strip()

    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except ValueError:
        pass

    for fmt in (
        "%d-%b-%Y %H:%M:%S",
        "%d-%b-%Y %H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S.%f",
        "%d/%m/%Y %H:%M:%S",
    ):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def iter_json_objects(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Dict[str, Any]]:
    """Parser streaming: array JSON, oggetti concatenati o separati da virgole."""
    decoder = json.JSONDecoder()
    buffer = ""
    pos = 0
    eof = False

    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        while True:
            if not eof and len(buffer) - pos < chunk_size // 4:
                more = f.read(chunk_size)
                if more:
                    buffer = buffer[pos:] + more
                    pos = 0
                else:
                    eof = True

            while pos < len(buffer) and (buffer[pos].isspace() or buffer[pos] in "[,]"):
                pos += 1

            if pos >= len(buffer):
                if eof:
                    break
                buffer = ""
                pos = 0
                continue

            if buffer[pos] == "]":
                pos += 1
                continue

            try:
                obj, end = decoder.raw_decode(buffer, pos)
            except json.JSONDecodeError:
                if eof:
                    if buffer[pos:].strip():
                        print("[WARN] Parte finale del JSON troncata/non valida ignorata.", file=sys.stderr)
                    break
                buffer = buffer[pos:]
                pos = 0
                more = f.read(chunk_size)
                if more:
                    buffer += more
                else:
                    eof = True
                continue

            pos = end
            if isinstance(obj, dict):
                yield obj
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, dict):
                        yield item

            if pos > chunk_size * 2:
                buffer = buffer[pos:]
                pos = 0


class Profiler:
    def __init__(self) -> None:
        self.total = 0
        self.normal = 0
        self.suspicious = 0
        self.unknown = 0

        self.ip_security = defaultdict(Counter)
        self.user_security = defaultdict(Counter)
        self.roles = Counter()
        self.role_security = defaultdict(Counter)

        self.unique_ips = set()
        self.unique_users = set()
        self.unique_sessions = set()
        self.ip_user = set()

        self.ip_to_users = defaultdict(set)
        self.user_to_ips = defaultdict(set)

        self.first_event: Optional[datetime] = None
        self.last_event: Optional[datetime] = None
        self.timestamps = array("d")
        self.events_per_second = Counter()
        self.events_per_minute = Counter()
        self.events_per_minute_security = defaultdict(Counter)

        self.session_span = {}
        self.ip_span = {}
        self.user_span = {}

        self.http_status = Counter()
        self.http_status_security = defaultdict(Counter)
        self.databases = Counter()
        self.tables = Counter()

        self.cert_present = 0
        self.cert_missing = 0

    @staticmethod
    def update_span(store, key: str, dt: datetime) -> None:
        if not key:
            return
        if key not in store:
            store[key] = [dt, dt]
            return
        if dt < store[key][0]:
            store[key][0] = dt
        if dt > store[key][1]:
            store[key][1] = dt

    def process(self, event: Dict[str, Any]) -> None:
        self.total += 1

        msg = event.get("message") if isinstance(event.get("message"), dict) else {}
        resource = msg.get("ResourceContext") if isinstance(msg.get("ResourceContext"), dict) else {}
        user_context = msg.get("UserContext") if isinstance(msg.get("UserContext"), dict) else {}

        # Sicurezza
        suspicious = parse_bool(msg.get("Suspicious"))
        if suspicious is True:
            label = "SOSP"
            self.suspicious += 1
        elif suspicious is False:
            label = "NORM"
            self.normal += 1
        else:
            label = "UNK"
            self.unknown += 1

        # IP / utente / sessione
        ip = clean(msg.get("IP"))
        user = clean(msg.get("User"))
        session = clean(msg.get("SessionId"))

        if ip:
            self.unique_ips.add(ip)
            self.ip_security[ip][label] += 1
        if user:
            self.unique_users.add(user)
            self.user_security[user][label] += 1
        if session:
            self.unique_sessions.add(session)

        roles = user_context.get("roles") if isinstance(user_context.get("roles"), list) else []
        for role in roles:
            if role is not None and str(role).strip():
                role_name = str(role).strip()
                self.roles[role_name] += 1
                self.role_security[role_name][label] += 1

        if ip and user:
            self.ip_user.add((ip, user))
            self.ip_to_users[ip].add(user)
            self.user_to_ips[user].add(ip)

        # HTTP status
        status = msg.get("HTTPStatusCode")
        if status is not None:
            status_key = str(status)
            self.http_status[status_key] += 1
            self.http_status_security[status_key][label] += 1

        # Risorse
        database = clean(resource.get("database"))
        table = clean(resource.get("table"))
        if database:
            self.databases[database] += 1
        if table:
            self.tables[table] += 1

        # Certificati
        if msg.get("ClientCertificate") is None:
            self.cert_missing += 1
        else:
            self.cert_present += 1

        # Tempo
        dt = parse_datetime(event.get("time"))
        if dt is not None:
            if self.first_event is None or dt < self.first_event:
                self.first_event = dt
            if self.last_event is None or dt > self.last_event:
                self.last_event = dt

            self.timestamps.append(dt.timestamp())
            self.events_per_second[dt.strftime("%Y-%m-%d %H:%M:%S")] += 1

            minute_key = dt.strftime("%Y-%m-%d %H:%M")
            self.events_per_minute[minute_key] += 1
            self.events_per_minute_security[minute_key][label] += 1

            self.update_span(self.session_span, session, dt)
            self.update_span(self.ip_span, ip, dt)
            self.update_span(self.user_span, user, dt)


def security_table(title: str, mapping, top: int) -> None:
    subsection(title)
    if not mapping:
        print("Nessun dato.")
        return

    rows = []
    for entity, counts in mapping.items():
        sosp = counts.get("SOSP", 0)
        norm = counts.get("NORM", 0)
        unk = counts.get("UNK", 0)
        total = sosp + norm + unk
        rate = sosp / total * 100 if total else 0
        rows.append((entity, total, sosp, norm, unk, rate))

    rows.sort(key=lambda x: (x[5], x[2], x[1]), reverse=True)
    if top > 0:
        rows = rows[:top]

    print(f"{'ENTITÀ':<42}{'TOT':>10}{'SOSP':>10}{'NORM':>10}{'UNK':>8}{'% SOSP':>10}")
    print("-" * WIDTH)
    for entity, total, sosp, norm, unk, rate in rows:
        print(
            f"{short(entity, 42):<42}"
            f"{fmt_int(total):>10}{fmt_int(sosp):>10}{fmt_int(norm):>10}"
            f"{fmt_int(unk):>8}{fmt_pct(rate):>10}"
        )


def duration_distribution(title: str, spans) -> None:
    subsection(title)
    if not spans:
        print("Nessun dato.")
        return

    durations = np.asarray(
        [(last - first).total_seconds() for first, last in spans.values()],
        dtype=float,
    )
    p25, median, p75, p95 = np.percentile(durations, [25, 50, 75, 95])

    print(f"Elementi        : {fmt_int(len(durations))}")
    print(f"Min             : {fmt_duration(float(np.min(durations)))}")
    print(f"P25             : {fmt_duration(float(p25))}")
    print(f"Mediana         : {fmt_duration(float(median))}")
    print(f"P75             : {fmt_duration(float(p75))}")
    print(f"P95             : {fmt_duration(float(p95))}")
    print(f"Max             : {fmt_duration(float(np.max(durations)))}")
    print(f"Media           : {fmt_duration(float(np.mean(durations)))}")


def print_counter(title: str, counter: Counter, total: int) -> None:
    subsection(title)
    if not counter:
        print("Nessun dato.")
        return
    for key, count in counter.most_common():
        pct = count / total * 100 if total else 0
        print(f"{short(key, 55):<55}{fmt_int(count):>12}{fmt_pct(pct):>12}")



def print_suspicious_concentration(title: str, mapping) -> None:
    subsection(title)

    rows = []
    for entity, counter in mapping.items():
        suspicious = counter.get("SOSP", 0)
        if suspicious > 0:
            rows.append((entity, suspicious))

    rows.sort(key=lambda x: x[1], reverse=True)
    total_suspicious = sum(count for _, count in rows)

    if total_suspicious == 0:
        print("Nessun evento sospetto disponibile.")
        return

    print(f"Eventi sospetti totali considerati : {fmt_int(total_suspicious)}")
    print(f"Soggetti con almeno 1 sospetto     : {fmt_int(len(rows))}")
    print()
    print(f"{'TOP':<10}{'EVENTI SOSPETTI':>20}{'% DEL TOTALE SOSPETTO':>28}")
    print("-" * 60)

    for n in (1, 3, 5, 10):
        effective_n = min(n, len(rows))
        if effective_n == 0:
            continue
        cumulative = sum(count for _, count in rows[:effective_n])
        percentage = cumulative / total_suspicious * 100
        print(
            f"Top {effective_n:<6}"
            f"{fmt_int(cumulative):>20}"
            f"{fmt_pct(percentage):>28}"
        )

    first_entity, first_count = rows[0]
    print()
    print(f"Primo contributore              : {short(first_entity, 50)}")
    print(
        f"Eventi sospetti del primo       : "
        f"{fmt_int(first_count)} "
        f"({fmt_pct(first_count / total_suspicious * 100)})"
    )


def print_report(p: Profiler, top: int) -> None:
    section("1. DATASET")
    print(f"Record totali             : {fmt_int(p.total)}")
    print(f"Record sospetti           : {fmt_int(p.suspicious)} ({fmt_pct(p.suspicious / p.total * 100 if p.total else 0)})")
    print(f"Record normali            : {fmt_int(p.normal)} ({fmt_pct(p.normal / p.total * 100 if p.total else 0)})")
    if p.unknown:
        print(f"Nota: {fmt_int(p.unknown)} record hanno Suspicious mancante/non interpretabile.")

    section("2. IP, UTENTI E SESSIONI")
    print(f"IP unici                  : {fmt_int(len(p.unique_ips))}")
    print(f"Utenti unici              : {fmt_int(len(p.unique_users))}")
    print(f"Sessioni uniche           : {fmt_int(len(p.unique_sessions))}")
    print(f"Combinazioni IP-Utente    : {fmt_int(len(p.ip_user))}")

    multi_user_ips = sum(1 for users in p.ip_to_users.values() if len(users) > 1)
    multi_ip_users = sum(1 for ips in p.user_to_ips.values() if len(ips) > 1)
    print(f"IP associati a più utenti : {fmt_int(multi_user_ips)}")
    print(f"Utenti associati a più IP : {fmt_int(multi_ip_users)}")

    print_counter("Ruoli applicativi", p.roles, sum(p.roles.values()) if p.roles else 0)

    section("3. SICUREZZA")
    security_table("Tasso di sospetto per IP", p.ip_security, top)
    security_table("Tasso di sospetto per utente", p.user_security, top)
    security_table("Tasso di sospetto per ruolo", p.role_security, top)

    print_suspicious_concentration(
        "Concentrazione degli eventi sospetti per IP",
        p.ip_security,
    )

    print_suspicious_concentration(
        "Concentrazione degli eventi sospetti per utente",
        p.user_security,
    )

    section("4. TEMPO")
    print(f"Primo evento              : {p.first_event.isoformat() if p.first_event else 'N/A'}")
    print(f"Ultimo evento             : {p.last_event.isoformat() if p.last_event else 'N/A'}")

    if p.first_event and p.last_event:
        total_seconds = (p.last_event - p.first_event).total_seconds()
        print(f"Durata totale dataset     : {fmt_duration(total_seconds)}")
        valid_events = len(p.timestamps)
        if total_seconds > 0:
            print(f"Eventi/secondo medi       : {fmt_float(valid_events / total_seconds, 6)}")
            print(f"Eventi/minuto medi        : {fmt_float(valid_events / (total_seconds / 60), 3)}")

    if p.events_per_second:
        second, count = p.events_per_second.most_common(1)[0]
        print(f"Picco stesso secondo      : {fmt_int(count)} eventi ({second})")
    if p.events_per_minute:
        minute, count = p.events_per_minute.most_common(1)[0]
        print(f"Picco stesso minuto       : {fmt_int(count)} eventi ({minute})")

    # Confronto temporale normali vs sospetti
    if p.events_per_minute_security:
        subsection("Eventi normali vs sospetti nel tempo")

        total_normal_time = sum(
            counts.get("NORM", 0)
            for counts in p.events_per_minute_security.values()
        )
        total_suspicious_time = sum(
            counts.get("SOSP", 0)
            for counts in p.events_per_minute_security.values()
        )

        print(f"Eventi normali con timestamp   : {fmt_int(total_normal_time)}")
        print(f"Eventi sospetti con timestamp  : {fmt_int(total_suspicious_time)}")

        normal_peak_minute, normal_peak_value = max(
            (
                (minute_key, counts.get("NORM", 0))
                for minute_key, counts in p.events_per_minute_security.items()
            ),
            key=lambda x: x[1],
        )

        suspicious_peak_minute, suspicious_peak_value = max(
            (
                (minute_key, counts.get("SOSP", 0))
                for minute_key, counts in p.events_per_minute_security.items()
            ),
            key=lambda x: x[1],
        )

        print(
            f"Picco normali nello stesso min : "
            f"{fmt_int(normal_peak_value)} ({normal_peak_minute})"
        )
        print(
            f"Picco sospetti nello stesso min: "
            f"{fmt_int(suspicious_peak_value)} ({suspicious_peak_minute})"
        )

        temporal_rows = []
        for minute_key, counts in p.events_per_minute_security.items():
            norm = counts.get("NORM", 0)
            sosp = counts.get("SOSP", 0)
            unk = counts.get("UNK", 0)
            total = norm + sosp + unk
            rate = sosp / total * 100 if total else 0
            temporal_rows.append(
                (minute_key, total, sosp, norm, unk, rate)
            )

        temporal_rows.sort(
            key=lambda x: (x[2], x[5], x[1]),
            reverse=True,
        )

        rows_to_show = temporal_rows if top == 0 else temporal_rows[:top]

        print()
        print(
            f"{'MINUTO':<19}"
            f"{'TOT':>10}"
            f"{'SOSP':>10}"
            f"{'NORM':>10}"
            f"{'UNK':>8}"
            f"{'% SOSP':>10}"
        )
        print("-" * WIDTH)

        for minute_key, total, sosp, norm, unk, rate in rows_to_show:
            print(
                f"{minute_key:<19}"
                f"{fmt_int(total):>10}"
                f"{fmt_int(sosp):>10}"
                f"{fmt_int(norm):>10}"
                f"{fmt_int(unk):>8}"
                f"{fmt_pct(rate):>10}"
            )

    duration_distribution("Durata delle sessioni", p.session_span)
    duration_distribution("Durata di attività degli IP", p.ip_span)
    duration_distribution("Durata di attività degli utenti", p.user_span)

    section("5. HTTP STATUS")
    print_counter("Distribuzione HTTP status", p.http_status, p.total)

    subsection("HTTP status: normali vs sospetti")

    if p.http_status_security:
        rows = []

        for status, counts in p.http_status_security.items():
            sosp = counts.get("SOSP", 0)
            norm = counts.get("NORM", 0)
            unk = counts.get("UNK", 0)
            total = sosp + norm + unk
            rate = sosp / total * 100 if total else 0
            rows.append((status, total, sosp, norm, unk, rate))

        def http_sort_key(row):
            try:
                return (0, int(row[0]))
            except ValueError:
                return (1, row[0])

        rows.sort(key=http_sort_key)

        print(
            f"{'STATUS':<12}"
            f"{'TOT':>12}"
            f"{'SOSP':>12}"
            f"{'NORM':>12}"
            f"{'UNK':>10}"
            f"{'% SOSP':>12}"
        )
        print("-" * WIDTH)

        for status, total, sosp, norm, unk, rate in rows:
            print(
                f"{status:<12}"
                f"{fmt_int(total):>12}"
                f"{fmt_int(sosp):>12}"
                f"{fmt_int(norm):>12}"
                f"{fmt_int(unk):>10}"
                f"{fmt_pct(rate):>12}"
            )
    else:
        print("Nessun dato.")

    section("6. RISORSE")
    print_counter("Database", p.databases, p.total)
    print_counter("Table", p.tables, p.total)

    section("7. CERTIFICATI CLIENT")
    print(f"Certificati presenti      : {fmt_int(p.cert_present)}")
    print(f"Certificati mancanti/null : {fmt_int(p.cert_missing)}")
    usage = p.cert_present / p.total * 100 if p.total else 0
    print(f"Percentuale di utilizzo   : {fmt_pct(usage)}")



# ============================================================================
# GRAFICI PROFESSIONALI / MINIMAL
# ============================================================================

# Palette volutamente neutra e coerente.
CHARCOAL = "#30343B"
LIGHT_GREY = "#AEB4BA"
GRID_GREY = "#D9DDE1"
TEXT_GREY = "#22262B"


def _professional_axes(ax, grid_axis="y"):
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_GREY)
    ax.spines["bottom"].set_color(GRID_GREY)
    ax.tick_params(colors=TEXT_GREY, labelsize=9)
    ax.grid(axis=grid_axis, color=GRID_GREY, linewidth=0.7, alpha=0.55)
    ax.set_axisbelow(True)


def _save_figure(fig, output_path: Path):
    fig.tight_layout()
    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def _annotate_vertical(ax, bars, labels):
    max_height = max((bar.get_height() for bar in bars), default=0)
    pad = max(max_height * 0.015, 0.5)

    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + pad,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
            color=TEXT_GREY,
            fontweight="semibold",
        )


def _annotate_horizontal(ax, bars, labels):
    max_width = max((bar.get_width() for bar in bars), default=0)
    pad = max(max_width * 0.012, 0.2)

    for bar, label in zip(bars, labels):
        ax.text(
            bar.get_width() + pad,
            bar.get_y() + bar.get_height() / 2,
            label,
            ha="left",
            va="center",
            fontsize=9,
            color=TEXT_GREY,
            fontweight="semibold",
        )


def _security_rows(mapping):
    rows = []

    for entity, counter in mapping.items():
        suspicious = counter.get("SOSP", 0)
        normal = counter.get("NORM", 0)
        unknown = counter.get("UNK", 0)
        total = suspicious + normal + unknown

        if total == 0:
            continue

        rate = suspicious / total * 100

        rows.append(
            (
                entity,
                total,
                suspicious,
                normal,
                unknown,
                rate,
            )
        )

    rows.sort(
        key=lambda x: (x[5], x[2], x[1]),
        reverse=True,
    )

    return rows


def _duration_summary(spans):
    if not spans:
        return None

    durations = np.asarray(
        [
            (last - first).total_seconds()
            for first, last in spans.values()
        ],
        dtype=float,
    )

    if durations.size == 0:
        return None

    return {
        "median": float(np.percentile(durations, 50)),
        "p95": float(np.percentile(durations, 95)),
        "max": float(np.max(durations)),
    }


def _duration_chart(title: str, spans, output_path: Path):
    stats = _duration_summary(spans)

    if not stats:
        return

    categories = ["Mediana", "P95", "Massimo"]
    values = [
        stats["median"],
        stats["p95"],
        stats["max"],
    ]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    bars = ax.bar(
        categories,
        values,
        color=CHARCOAL,
        width=0.56,
    )

    ax.set_title(
        title,
        loc="left",
        fontsize=14,
        fontweight="semibold",
        pad=14,
    )

    ax.set_ylabel("Durata (secondi)")
    _professional_axes(ax, "y")

    _annotate_vertical(
        ax,
        bars,
        [fmt_duration(value) for value in values],
    )

    max_value = max(values)

    ax.set_ylim(
        0,
        max_value * 1.20 if max_value > 0 else 1,
    )

    _save_figure(fig, output_path)



def _suspicious_pareto_chart(
    title: str,
    mapping,
    output_path: Path,
    plot_top: int = 12,
):
    rows = []

    for entity, counter in mapping.items():
        suspicious = counter.get("SOSP", 0)
        if suspicious > 0:
            rows.append((entity, suspicious))

    rows.sort(key=lambda x: x[1], reverse=True)

    if not rows:
        fig, ax = plt.subplots(figsize=(9.2, 4.8))
        ax.axis("off")
        ax.text(
            0.5, 0.56, title,
            ha="center", va="center",
            fontsize=14, fontweight="semibold",
            color=TEXT_GREY,
        )
        ax.text(
            0.5, 0.43,
            "Nessun evento sospetto disponibile.",
            ha="center", va="center",
            fontsize=10, color=TEXT_GREY,
        )
        _save_figure(fig, output_path)
        return

    total_suspicious = sum(count for _, count in rows)
    shown = rows[:plot_top]

    categories = [short(entity, 24) for entity, _ in shown]
    values = [count for _, count in shown]

    cumulative_percentages = []
    cumulative = 0
    for _, count in shown:
        cumulative += count
        cumulative_percentages.append(
            cumulative / total_suspicious * 100
        )

    x = np.arange(len(categories))

    fig, ax = plt.subplots(
        figsize=(max(10.2, len(categories) * 0.78 + 3), 5.8)
    )

    bars = ax.bar(
        x,
        values,
        color=CHARCOAL,
        width=0.62,
    )

    ax.set_title(
        title,
        loc="left",
        fontsize=14,
        fontweight="semibold",
        pad=14,
    )
    ax.set_ylabel("Eventi sospetti")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=28, ha="right")
    _professional_axes(ax, "y")

    _annotate_vertical(
        ax,
        bars,
        [fmt_int(value) for value in values],
    )

    max_value = max(values)
    ax.set_ylim(0, max_value * 1.22 if max_value > 0 else 1)

    ax2 = ax.twinx()
    ax2.plot(
        x,
        cumulative_percentages,
        color=LIGHT_GREY,
        linewidth=1.8,
        marker="o",
        markersize=4.5,
    )
    ax2.set_ylim(0, 105)
    ax2.set_ylabel(
        "Quota cumulata del totale sospetto (%)",
        color=TEXT_GREY,
    )
    ax2.tick_params(axis="y", colors=TEXT_GREY, labelsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color(GRID_GREY)

    for idx, percentage in enumerate(cumulative_percentages):
        show_label = (
            len(cumulative_percentages) <= 8
            or idx in {0, 2, 4, 9, len(cumulative_percentages) - 1}
        )
        if show_label:
            ax2.annotate(
                fmt_pct(percentage),
                xy=(x[idx], percentage),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=TEXT_GREY,
            )

    fig.text(
        0.01,
        0.01,
        (
            f"Totale eventi sospetti: {fmt_int(total_suspicious)}. "
            f"Visualizzati i primi {fmt_int(len(shown))} soggetti."
        ),
        fontsize=8.5,
        color=TEXT_GREY,
    )

    _save_figure(fig, output_path)


def make_plots(
    p: Profiler,
    output_dir: Path,
    plot_top: int = 12,
):

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # 1. Record normali / sospetti
    # ------------------------------------------------------------------------

    categories = [
        "Sospetti",
        "Normali",
    ]

    values = [
        p.suspicious,
        p.normal,
    ]

    total = p.total or 1

    exact_labels = [
        f"{fmt_int(p.suspicious)} ({fmt_pct(p.suspicious / total * 100)})",
        f"{fmt_int(p.normal)} ({fmt_pct(p.normal / total * 100)})",
    ]

    fig, ax = plt.subplots(figsize=(8.4, 5.0))

    bars = ax.bar(
        categories,
        values,
        color=[CHARCOAL, LIGHT_GREY],
        width=0.56,
    )

    ax.set_title(
        "Classificazione dei record",
        loc="left",
        fontsize=14,
        fontweight="semibold",
        pad=14,
    )

    ax.set_ylabel("Numero di record")
    _professional_axes(ax, "y")
    _annotate_vertical(ax, bars, exact_labels)

    max_value = max(values)

    ax.set_ylim(
        0,
        max_value * 1.20 if max_value > 0 else 1,
    )

    _save_figure(
        fig,
        output_dir / "01_record_classification.png",
    )

    # ------------------------------------------------------------------------
    # 2. Cardinalità principali
    # ------------------------------------------------------------------------

    ip_multi_user = sum(
        1
        for users in p.ip_to_users.values()
        if len(users) > 1
    )

    user_multi_ip = sum(
        1
        for ips in p.user_to_ips.values()
        if len(ips) > 1
    )

    categories = [
        "IP unici",
        "Utenti unici",
        "Sessioni uniche",
        "IP-Utente",
        "IP >1 utente",
        "Utenti >1 IP",
    ]

    values = [
        len(p.unique_ips),
        len(p.unique_users),
        len(p.unique_sessions),
        len(p.ip_user),
        ip_multi_user,
        user_multi_ip,
    ]

    fig, ax = plt.subplots(figsize=(10.2, 5.4))

    bars = ax.bar(
        categories,
        values,
        color=CHARCOAL,
        width=0.62,
    )

    ax.set_title(
        "Cardinalità e associazioni",
        loc="left",
        fontsize=14,
        fontweight="semibold",
        pad=14,
    )

    ax.set_ylabel("Numero")
    _professional_axes(ax, "y")
    ax.tick_params(axis="x", rotation=20)

    _annotate_vertical(
        ax,
        bars,
        [fmt_int(value) for value in values],
    )

    max_value = max(values)

    ax.set_ylim(
        0,
        max_value * 1.20 if max_value > 0 else 1,
    )

    _save_figure(
        fig,
        output_dir / "02_cardinality.png",
    )

    # ------------------------------------------------------------------------
    # 3. Sicurezza per IP
    # ------------------------------------------------------------------------

    ip_rows = _security_rows(
        p.ip_security,
    )[:plot_top]

    if ip_rows:

        entities = [
            short(row[0], 34)
            for row in ip_rows
        ][::-1]

        rates = [
            row[5]
            for row in ip_rows
        ][::-1]

        labels = [
            (
                f"{fmt_pct(row[5])} · "
                f"{fmt_int(row[2])}/{fmt_int(row[1])}"
            )
            for row in ip_rows
        ][::-1]

        fig, ax = plt.subplots(
            figsize=(
                10.2,
                max(5.0, len(ip_rows) * 0.46 + 1.5),
            )
        )

        bars = ax.barh(
            entities,
            rates,
            color=CHARCOAL,
            height=0.58,
        )

        ax.set_title(
            "Tasso di sospetto per IP",
            loc="left",
            fontsize=14,
            fontweight="semibold",
            pad=14,
        )

        ax.set_xlabel("Percentuale di eventi sospetti")
        ax.set_xlim(0, 112)

        _professional_axes(ax, "x")
        _annotate_horizontal(ax, bars, labels)

        _save_figure(
            fig,
            output_dir / "03_security_by_ip.png",
        )

    # ------------------------------------------------------------------------
    # 4. Sicurezza per utente
    # ------------------------------------------------------------------------

    user_rows = _security_rows(
        p.user_security,
    )[:plot_top]

    if user_rows:

        entities = [
            short(row[0], 34)
            for row in user_rows
        ][::-1]

        rates = [
            row[5]
            for row in user_rows
        ][::-1]

        labels = [
            (
                f"{fmt_pct(row[5])} · "
                f"{fmt_int(row[2])}/{fmt_int(row[1])}"
            )
            for row in user_rows
        ][::-1]

        fig, ax = plt.subplots(
            figsize=(
                10.2,
                max(5.0, len(user_rows) * 0.46 + 1.5),
            )
        )

        bars = ax.barh(
            entities,
            rates,
            color=CHARCOAL,
            height=0.58,
        )

        ax.set_title(
            "Tasso di sospetto per utente",
            loc="left",
            fontsize=14,
            fontweight="semibold",
            pad=14,
        )

        ax.set_xlabel("Percentuale di eventi sospetti")
        ax.set_xlim(0, 112)

        _professional_axes(ax, "x")
        _annotate_horizontal(ax, bars, labels)

        _save_figure(
            fig,
            output_dir / "04_security_by_user.png",
        )

    # ------------------------------------------------------------------------
    # 5. Eventi per minuto
    # ------------------------------------------------------------------------

    if p.events_per_minute:

        times = sorted(
            p.events_per_minute
        )

        values = [
            p.events_per_minute[t]
            for t in times
        ]

        x = np.arange(
            len(times)
        )

        fig, ax = plt.subplots(
            figsize=(11.6, 5.5)
        )

        ax.plot(
            x,
            values,
            color=CHARCOAL,
            linewidth=1.35,
        )

        peak_index = int(
            np.argmax(values)
        )

        peak_value = values[
            peak_index
        ]

        peak_time = times[
            peak_index
        ]

        ax.scatter(
            [peak_index],
            [peak_value],
            color=CHARCOAL,
            s=28,
            zorder=3,
        )

        # Se la serie è corta, mostra il valore esatto di ogni punto.
        # Su serie lunghe sarebbe illeggibile: in quel caso annota solo il picco.
        if len(values) <= 30:
            for xi, yi in zip(x, values):
                ax.annotate(
                    fmt_int(yi),
                    xy=(xi, yi),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    color=TEXT_GREY,
                )

        ax.annotate(
            (
                f"Picco: {fmt_int(peak_value)} eventi\n"
                f"{peak_time}"
            ),
            xy=(
                peak_index,
                peak_value,
            ),
            xytext=(10, 16),
            textcoords="offset points",
            fontsize=9,
            color=TEXT_GREY,
            bbox=dict(
                boxstyle="round,pad=0.35",
                fc="white",
                ec=GRID_GREY,
            ),
        )

        ax.set_title(
            "Volume di eventi nel tempo",
            loc="left",
            fontsize=14,
            fontweight="semibold",
            pad=14,
        )

        ax.set_ylabel("Eventi per minuto")
        ax.set_xlabel("Tempo")

        _professional_axes(ax, "y")

        tick_count = min(
            8,
            len(times),
        )

        if tick_count > 1:

            tick_indexes = np.linspace(
                0,
                len(times) - 1,
                tick_count,
                dtype=int,
            )

            ax.set_xticks(
                tick_indexes
            )

            ax.set_xticklabels(
                [
                    times[i]
                    for i in tick_indexes
                ],
                rotation=30,
                ha="right",
            )

        _save_figure(
            fig,
            output_dir / "05_events_per_minute.png",
        )

    # ------------------------------------------------------------------------
    # 5B. Normali vs sospetti nel tempo
    # ------------------------------------------------------------------------

    if p.events_per_minute:

        times = sorted(
            p.events_per_minute
        )

        normal_values = [
            p.events_per_minute_security[t].get("NORM", 0)
            for t in times
        ]

        suspicious_values = [
            p.events_per_minute_security[t].get("SOSP", 0)
            for t in times
        ]

        x = np.arange(
            len(times)
        )

        fig, ax = plt.subplots(
            figsize=(11.8, 5.7)
        )

        ax.plot(
            x,
            normal_values,
            color=LIGHT_GREY,
            linewidth=1.4,
            label="Normali",
        )

        ax.plot(
            x,
            suspicious_values,
            color=CHARCOAL,
            linewidth=1.6,
            label="Sospetti",
        )

        # Picco sospetto annotato con valore esatto
        if suspicious_values and max(suspicious_values) > 0:

            peak_index = int(
                np.argmax(suspicious_values)
            )

            peak_value = suspicious_values[
                peak_index
            ]

            peak_time = times[
                peak_index
            ]

            ax.scatter(
                [peak_index],
                [peak_value],
                color=CHARCOAL,
                s=30,
                zorder=3,
            )

            ax.annotate(
                (
                    f"Picco sospetti: "
                    f"{fmt_int(peak_value)}\n"
                    f"{peak_time}"
                ),
                xy=(
                    peak_index,
                    peak_value,
                ),
                xytext=(10, 16),
                textcoords="offset points",
                fontsize=9,
                color=TEXT_GREY,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    fc="white",
                    ec=GRID_GREY,
                ),
            )

        ax.set_title(
            "Eventi normali e sospetti nel tempo",
            loc="left",
            fontsize=14,
            fontweight="semibold",
            pad=14,
        )

        ax.set_ylabel(
            "Eventi per minuto"
        )

        ax.set_xlabel(
            "Tempo"
        )

        _professional_axes(
            ax,
            "y",
        )

        ax.legend(
            frameon=False,
            loc="upper right",
        )

        tick_count = min(
            8,
            len(times),
        )

        if tick_count > 1:

            tick_indexes = np.linspace(
                0,
                len(times) - 1,
                tick_count,
                dtype=int,
            )

            ax.set_xticks(
                tick_indexes
            )

            ax.set_xticklabels(
                [
                    times[i]
                    for i in tick_indexes
                ],
                rotation=30,
                ha="right",
            )

        _save_figure(
            fig,
            output_dir / "06_normal_vs_suspicious_over_time.png",
        )

    else:
        fig, ax = plt.subplots(figsize=(9.5, 4.8))
        ax.axis("off")
        ax.text(
            0.5,
            0.56,
            "Eventi normali e sospetti nel tempo",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="semibold",
            color=TEXT_GREY,
        )
        ax.text(
            0.5,
            0.43,
            "Nessun timestamp interpretabile disponibile nel dataset.",
            ha="center",
            va="center",
            fontsize=10,
            color=TEXT_GREY,
        )
        _save_figure(
            fig,
            output_dir / "06_normal_vs_suspicious_over_time.png",
        )

    # ------------------------------------------------------------------------
    # 5C. Tasso sospetto per ruolo
    # ------------------------------------------------------------------------

    role_rows = _security_rows(
        p.role_security,
    )

    if role_rows:

        categories = [
            short(row[0], 28)
            for row in role_rows
        ][::-1]

        rates = [
            row[5]
            for row in role_rows
        ][::-1]

        labels = [
            (
                f"{fmt_pct(row[5])} · "
                f"{fmt_int(row[2])}/{fmt_int(row[1])}"
            )
            for row in role_rows
        ][::-1]

        fig, ax = plt.subplots(
            figsize=(
                9.6,
                max(
                    4.8,
                    len(role_rows) * 0.5 + 1.5,
                ),
            )
        )

        bars = ax.barh(
            categories,
            rates,
            color=CHARCOAL,
            height=0.58,
        )

        ax.set_title(
            "Tasso di sospetto per ruolo",
            loc="left",
            fontsize=14,
            fontweight="semibold",
            pad=14,
        )

        ax.set_xlabel(
            "Percentuale di eventi sospetti"
        )

        ax.set_xlim(
            0,
            112,
        )

        _professional_axes(
            ax,
            "x",
        )

        _annotate_horizontal(
            ax,
            bars,
            labels,
        )

        _save_figure(
            fig,
            output_dir / "07_security_by_role.png",
        )

    else:
        # Crea comunque il file per rendere evidente che il dataset
        # non contiene UserContext.roles utilizzabili.
        fig, ax = plt.subplots(figsize=(8.8, 4.8))
        ax.axis("off")
        ax.text(
            0.5,
            0.56,
            "Tasso di sospetto per ruolo",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="semibold",
            color=TEXT_GREY,
        )
        ax.text(
            0.5,
            0.43,
            "Nessun ruolo applicativo disponibile nel dataset.",
            ha="center",
            va="center",
            fontsize=10,
            color=TEXT_GREY,
        )
        _save_figure(
            fig,
            output_dir / "07_security_by_role.png",
        )

    # ------------------------------------------------------------------------
    # 5D. HTTP status: normali vs sospetti
    # ------------------------------------------------------------------------

    if p.http_status_security:

        def http_status_sort(item):
            key = item[0]
            try:
                return (0, int(key))
            except ValueError:
                return (1, key)

        rows = sorted(
            p.http_status_security.items(),
            key=http_status_sort,
        )

        categories = [
            str(status)
            for status, _ in rows
        ]

        normal_values = [
            counter.get("NORM", 0)
            for _, counter in rows
        ]

        suspicious_values = [
            counter.get("SOSP", 0)
            for _, counter in rows
        ]

        x = np.arange(
            len(categories)
        )

        width = 0.36

        fig, ax = plt.subplots(
            figsize=(
                max(8.8, len(categories) * 0.8 + 3),
                5.4,
            )
        )

        normal_bars = ax.bar(
            x - width / 2,
            normal_values,
            width,
            label="Normali",
            color=LIGHT_GREY,
        )

        suspicious_bars = ax.bar(
            x + width / 2,
            suspicious_values,
            width,
            label="Sospetti",
            color=CHARCOAL,
        )

        ax.set_title(
            "HTTP status: normali vs sospetti",
            loc="left",
            fontsize=14,
            fontweight="semibold",
            pad=14,
        )

        ax.set_xlabel(
            "HTTP status"
        )

        ax.set_ylabel(
            "Numero di record"
        )

        ax.set_xticks(
            x
        )

        ax.set_xticklabels(
            categories
        )

        _professional_axes(
            ax,
            "y",
        )

        ax.legend(
            frameon=False,
            loc="upper right",
        )

        normal_labels = [
            fmt_int(value)
            for value in normal_values
        ]

        suspicious_labels = [
            fmt_int(value)
            for value in suspicious_values
        ]

        _annotate_vertical(
            ax,
            normal_bars,
            normal_labels,
        )

        _annotate_vertical(
            ax,
            suspicious_bars,
            suspicious_labels,
        )

        max_value = max(
            normal_values + suspicious_values
        )

        ax.set_ylim(
            0,
            max_value * 1.22 if max_value > 0 else 1,
        )

        _save_figure(
            fig,
            output_dir / "08_http_status_normal_vs_suspicious.png",
        )

    # ------------------------------------------------------------------------
    # 9-11. Durate
    # ------------------------------------------------------------------------

    _duration_chart(
        "Durata delle sessioni",
        p.session_span,
        output_dir / "09_session_duration.png",
    )

    _duration_chart(
        "Durata di attività degli IP",
        p.ip_span,
        output_dir / "10_ip_activity_duration.png",
    )

    _duration_chart(
        "Durata di attività degli utenti",
        p.user_span,
        output_dir / "11_user_activity_duration.png",
    )

    # ------------------------------------------------------------------------
    # 12. HTTP status
    # ------------------------------------------------------------------------

    if p.http_status:

        rows = p.http_status.most_common()

        categories = [
            str(key)
            for key, _ in rows
        ]

        values = [
            count
            for _, count in rows
        ]

        fig, ax = plt.subplots(
            figsize=(8.6, 5.0)
        )

        bars = ax.bar(
            categories,
            values,
            color=CHARCOAL,
            width=0.58,
        )

        ax.set_title(
            "Distribuzione HTTP status",
            loc="left",
            fontsize=14,
            fontweight="semibold",
            pad=14,
        )

        ax.set_xlabel("HTTP status")
        ax.set_ylabel("Numero di record")

        _professional_axes(ax, "y")

        labels = [
            (
                f"{fmt_int(value)} "
                f"({fmt_pct(value / p.total * 100 if p.total else 0)})"
            )
            for value in values
        ]

        _annotate_vertical(
            ax,
            bars,
            labels,
        )

        max_value = max(values)

        ax.set_ylim(
            0,
            max_value * 1.22 if max_value > 0 else 1,
        )

        _save_figure(
            fig,
            output_dir / "12_http_status.png",
        )

    # ------------------------------------------------------------------------
    # 13-14. Risorse
    # ------------------------------------------------------------------------

    resource_charts = [
        (
            "Database",
            p.databases,
            "13_databases.png",
        ),
        (
            "Table",
            p.tables,
            "14_tables.png",
        ),
    ]

    for title, counter, filename in resource_charts:

        rows = counter.most_common(
            plot_top
        )

        if not rows:
            continue

        categories = [
            short(key, 30)
            for key, _ in rows
        ][::-1]

        values = [
            count
            for _, count in rows
        ][::-1]

        fig, ax = plt.subplots(
            figsize=(
                9.8,
                max(4.8, len(rows) * 0.42 + 1.4),
            )
        )

        bars = ax.barh(
            categories,
            values,
            color=CHARCOAL,
            height=0.58,
        )

        ax.set_title(
            title,
            loc="left",
            fontsize=14,
            fontweight="semibold",
            pad=14,
        )

        ax.set_xlabel(
            "Numero di record"
        )

        _professional_axes(
            ax,
            "x",
        )

        labels = [
            (
                f"{fmt_int(value)} "
                f"({fmt_pct(value / p.total * 100 if p.total else 0)})"
            )
            for value in values
        ]

        _annotate_horizontal(
            ax,
            bars,
            labels,
        )

        max_value = max(values)

        ax.set_xlim(
            0,
            max_value * 1.34 if max_value > 0 else 1,
        )

        _save_figure(
            fig,
            output_dir / filename,
        )


    # ------------------------------------------------------------------------
    # 15. Ruoli applicativi
    # ------------------------------------------------------------------------

    if p.roles:

        rows = p.roles.most_common()
        categories = [str(role) for role, _ in rows]
        values = [count for _, count in rows]
        total_roles = sum(values) or 1

        fig, ax = plt.subplots(figsize=(8.6, 5.0))

        bars = ax.bar(
            categories,
            values,
            color=CHARCOAL,
            width=0.58,
        )

        ax.set_title(
            "Distribuzione dei ruoli applicativi",
            loc="left",
            fontsize=14,
            fontweight="semibold",
            pad=14,
        )

        ax.set_ylabel("Numero di occorrenze")
        _professional_axes(ax, "y")

        labels = [
            f"{fmt_int(value)} ({fmt_pct(value / total_roles * 100)})"
            for value in values
        ]

        _annotate_vertical(ax, bars, labels)

        max_value = max(values)
        ax.set_ylim(0, max_value * 1.20 if max_value > 0 else 1)

        _save_figure(
            fig,
            output_dir / "15_user_roles.png",
        )

    # ------------------------------------------------------------------------
    # 16-17. Concentrazione eventi sospetti
    # ------------------------------------------------------------------------

    _suspicious_pareto_chart(
        "Concentrazione degli eventi sospetti per IP",
        p.ip_security,
        output_dir / "16_suspicious_concentration_ip.png",
        plot_top,
    )

    _suspicious_pareto_chart(
        "Concentrazione degli eventi sospetti per utente",
        p.user_security,
        output_dir / "17_suspicious_concentration_user.png",
        plot_top,
    )

    # ------------------------------------------------------------------------
    # 18. Certificati
    # ------------------------------------------------------------------------

    categories = [
        "Presenti",
        "Mancanti / null",
    ]

    values = [
        p.cert_present,
        p.cert_missing,
    ]

    labels = [
        (
            f"{fmt_int(p.cert_present)} "
            f"({fmt_pct(p.cert_present / p.total * 100 if p.total else 0)})"
        ),
        (
            f"{fmt_int(p.cert_missing)} "
            f"({fmt_pct(p.cert_missing / p.total * 100 if p.total else 0)})"
        ),
    ]

    fig, ax = plt.subplots(
        figsize=(8.6, 5.0)
    )

    bars = ax.bar(
        categories,
        values,
        color=[CHARCOAL, LIGHT_GREY],
        width=0.56,
    )

    ax.set_title(
        "Utilizzo dei certificati client",
        loc="left",
        fontsize=14,
        fontweight="semibold",
        pad=14,
    )

    ax.set_ylabel(
        "Numero di record"
    )

    _professional_axes(
        ax,
        "y",
    )

    _annotate_vertical(
        ax,
        bars,
        labels,
    )

    max_value = max(values)

    ax.set_ylim(
        0,
        max_value * 1.20 if max_value > 0 else 1,
    )

    _save_figure(
        fig,
        output_dir / "18_client_certificates.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analisi essenziale di dataset JSON audit/log.")
    parser.add_argument("json_file", type=Path, help="File JSON da analizzare")
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Numero massimo di righe nelle tabelle sicurezza. 0 = tutte. Default: 30.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help="Mostra avanzamento ogni N record. 0 = disabilitato.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Non generare grafici PNG.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("audit_plots"),
        help="Cartella dei grafici PNG. Default: audit_plots.",
    )
    parser.add_argument(
        "--plot-top",
        type=int,
        default=12,
        help="Numero massimo di IP/utenti/risorse mostrati nei grafici. Default: 12.",
    )
    args = parser.parse_args()

    if not args.json_file.exists():
        raise SystemExit(f"File non trovato: {args.json_file}")
    if args.top < 0:
        raise SystemExit("--top deve essere >= 0")
    if args.plot_top <= 0:
        raise SystemExit("--plot-top deve essere > 0")

    profiler = Profiler()

    for i, event in enumerate(iter_json_objects(args.json_file), start=1):
        profiler.process(event)
        if args.progress_every > 0 and i % args.progress_every == 0:
            print(f"[INFO] Processati {fmt_int(i)} record...", file=sys.stderr)

    if profiler.total == 0:
        raise SystemExit("Nessun record JSON valido trovato.")

    print_report(profiler, args.top)

    if not args.no_plots:
        make_plots(
            profiler,
            args.plots_dir,
            args.plot_top,
        )
        print()
        print(
            f"Grafici salvati in: "
            f"{args.plots_dir.resolve()}"
        )


if __name__ == "__main__":
    main()