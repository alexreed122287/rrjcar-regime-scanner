"""
scheduled_scan.py — Headless scheduled scanner (V1 + V2 dual strategy)

Schedule: Weekdays at 10:00 AM CST and 1:00 PM CST
  - 10:00 AM: Initial scan (V1 then V2) — sends results immediately
  - 1:00 PM: Confirmation scan (V1 then V2) — cross-references AM results
    to highlight tickers that appear in BOTH scans as "Confirmed Signals"

Both strategies:
  - Bull Run only regime
  - ≥80% regime confidence
  - 3+ regime confirmation bars
  - No min volume requirement
  - Exclude healthcare/biotech (handled by screener)
  - Only tickers with options
  - Results include company name + sector/industry
  - Ranked top buy to least (by confirmations descending)

V1: 6/8 confirmations required
V2: 8/12 confirmations required
"""

import os
import sys
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from screener import scan_watchlist, WATCHLISTS, BULLISH_SIGNALS

# ── Scan Settings ──
MIN_CONFIDENCE = 0.80
ALLOWED_REGIMES = {"Bull Run"}
REGIME_CONFIRM_BARS = 3
V1_MIN_CONFS = 6
V2_MIN_CONFS = 8

RECIPIENTS = [
    "alexander.s.reed@gmail.com",
    "jasoncolvin7.0@gmail.com",
    "ruizrk@yahoo.com",
]

# File to persist AM scan results for PM confirmation
AM_RESULTS_FILE = os.path.join(os.path.dirname(__file__), ".am_scan_results.json")


def send_email(subject: str, body_html: str, recipients: list = None):
    """Send email via SMTP to multiple recipients."""
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    to_emails = recipients or RECIPIENTS

    if not smtp_user or not smtp_password:
        print("[Alert] SMTP not configured, skipping email")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = ", ".join(to_emails)
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, to_emails, msg.as_string())
        print(f"[Alert] Email sent to {', '.join(to_emails)}")
        return True
    except Exception as e:
        print(f"[Alert] Email failed: {e}")
        return False


def send_telegram(message: str):
    """Send Telegram message."""
    import requests
    bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

    if not bot_token or not chat_id:
        print("[Alert] Telegram not configured, skipping")
        return False

    try:
        r = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        r.raise_for_status()
        print("[Alert] Telegram sent")
        return True
    except Exception as e:
        print(f"[Alert] Telegram failed: {e}")
        return False


def filter_results(results: list, min_confs: int) -> list:
    """
    Filter scan results:
    - Bull Run regime only
    - ≥80% confidence
    - ≥min_confs confirmations
    - Must have options available
    Sort by confirmations descending (top buy first).
    """
    filtered = []
    for r in results:
        if r.get("regime_label") not in ALLOWED_REGIMES:
            continue
        if (r.get("regime_confidence") or 0) < MIN_CONFIDENCE:
            continue
        if (r.get("confirmations_met") or 0) < min_confs:
            continue
        if not r.get("has_options", True):
            continue
        filtered.append(r)

    filtered.sort(key=lambda r: -(r.get("confirmations_met") or 0))
    return filtered


def run_scan(strategy: str, min_confs: int) -> list:
    """Run a scan with specified strategy on ALL TICKERS and return filtered results."""
    tickers = WATCHLISTS.get("ALL TICKERS", [])
    if not tickers:
        print(f"[Scan] No tickers in 'ALL TICKERS' watchlist")
        return []

    total = len(tickers)
    confs_total = 8 if strategy == "v1" else 12
    print(f"[Scan] Starting {strategy.upper()} scan of {total:,} tickers at {datetime.now()}")

    def progress(batch_num, total_batches, running):
        bullish = [r for r in running if r.get("signal") in BULLISH_SIGNALS]
        scanned = min(batch_num * 200, total)
        print(f"[Scan] Batch {batch_num}/{total_batches} ({scanned:,}/{total:,}) — {len(bullish)} bullish")

    results = scan_watchlist(
        symbols=tickers,
        interval="1d",
        n_regimes=7,
        min_confirmations=min_confs,
        regime_confirm_bars=REGIME_CONFIRM_BARS,
        max_workers=10,
        strategy=strategy,
        batch_size=200,
        progress_callback=progress,
        bullish_only=True,
        min_avg_volume=0,  # No volume requirement
    )

    print(f"[Scan] Raw bullish: {len(results)} — applying filters...")
    filtered = filter_results(results, min_confs)
    print(f"[Scan] {strategy.upper()} hits: {len(filtered)} (Bull Run, ≥{MIN_CONFIDENCE*100:.0f}% conf, ≥{min_confs}/{confs_total} confs)")
    return filtered


def format_email(hits: list, scan_label: str, strategy: str, min_confs: int, confirmed_symbols: set = None) -> str:
    """Format scan hits as HTML email with company name, sector/industry."""
    now = datetime.now().strftime("%B %d, %Y %I:%M %p CT")
    confs_total = 8 if strategy == "v1" else 12

    if not hits:
        return f"""
        <div style="font-family:Arial,sans-serif;max-width:800px;margin:0 auto;background:#101114;color:#e5e7eb;padding:20px;border-radius:8px;">
            <h2 style="color:#2dd4bf;margin-top:0;">RRJCAR Regime Scanner</h2>
            <p><b>Scan Time:</b> {now}</p>
            <p>No hits found for {scan_label} today.</p>
        </div>
        """

    def ticker_row(r, rank):
        sym = r.get("symbol", "?")
        name = r.get("name") or ""
        sector = r.get("sector") or ""
        industry = r.get("industry") or ""
        sector_display = f"{sector} / {industry}" if sector and industry else sector or industry or "—"
        price = r.get("price", 0)
        regime = r.get("regime_label", "?")
        conf = r.get("regime_confidence", 0)
        confs = r.get("confirmations_met", 0)
        signal = r.get("signal", "?")
        change = r.get("change_1d") or 0
        color = "#2dd4bf" if change >= 0 else "#f87171"
        signal_short = signal.replace("LONG -- ", "")

        # Highlight confirmed signals (appear in both AM and PM scans)
        is_confirmed = confirmed_symbols and sym in confirmed_symbols
        confirm_badge = '<span style="background:#22c55e;color:#000;padding:1px 4px;border-radius:3px;font-size:9px;font-weight:700;margin-left:4px;">CONFIRMED</span>' if is_confirmed else ''

        return (
            f'<tr style="border-bottom:1px solid #1e2028;">'
            f'<td style="padding:6px;">{rank}</td>'
            f'<td style="padding:6px;"><b>{sym}</b>{confirm_badge}<br><span style="font-size:10px;color:#6b7280;">{name}</span></td>'
            f'<td style="padding:6px;font-size:10px;color:#9ca3af;">{sector_display}</td>'
            f'<td style="padding:6px;">${price:.2f}</td>'
            f'<td style="padding:6px;color:{color}">{change:+.1f}%</td>'
            f'<td style="padding:6px;">{regime}</td>'
            f'<td style="padding:6px;">{conf*100:.0f}%</td>'
            f'<td style="padding:6px;">{confs}/{confs_total}</td>'
            f'<td style="padding:6px;">{signal_short}</td>'
            f'</tr>'
        )

    rows_html = "".join(ticker_row(r, i + 1) for i, r in enumerate(hits))

    confirmed_count = ""
    if confirmed_symbols:
        n = sum(1 for h in hits if h.get("symbol") in confirmed_symbols)
        if n:
            confirmed_count = f' | <span style="color:#22c55e;">{n} confirmed from AM scan</span>'

    return f"""
    <div style="font-family:Arial,sans-serif;max-width:900px;margin:0 auto;background:#101114;color:#e5e7eb;padding:20px;border-radius:8px;">
        <h2 style="color:#2dd4bf;margin-top:0;">RRJCAR Regime Scanner — {strategy.upper()}</h2>
        <p style="color:#9ca3af;">{now} | {scan_label}</p>
        <p style="color:#9ca3af;font-size:12px;">
            Filters: Bull Run only | ≥{MIN_CONFIDENCE*100:.0f}% confidence | ≥{min_confs}/{confs_total} confs | {REGIME_CONFIRM_BARS}+ regime bars | Options required
        </p>
        <p><b style="color:#2dd4bf;">{len(hits)}</b> hits — ranked top buy to least{confirmed_count}</p>
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <tr style="color:#6b7280;font-size:11px;text-transform:uppercase;">
                <th style="text-align:left;padding:6px;">#</th>
                <th style="text-align:left;padding:6px;">Ticker</th>
                <th style="text-align:left;">Sector / Industry</th>
                <th style="text-align:left;">Price</th>
                <th style="text-align:left;">1D</th>
                <th style="text-align:left;">Regime</th>
                <th style="text-align:left;">Conf%</th>
                <th style="text-align:left;">Confs</th>
                <th style="text-align:left;">Signal</th>
            </tr>
            {rows_html}
        </table>
        <p style="color:#4b5563;font-size:11px;margin-top:20px;">
            Sent by RRJCAR Regime Scanner via GitHub Actions
        </p>
    </div>
    """


def format_telegram(hits: list, scan_label: str, strategy: str, min_confs: int) -> str:
    """Format hits as Telegram message."""
    now = datetime.now().strftime("%I:%M %p CT")
    confs_total = 8 if strategy == "v1" else 12

    if not hits:
        return f"<b>RRJCAR {strategy.upper()}</b> ({now})\nNo hits for {scan_label} today."

    lines = [
        f"<b>RRJCAR {strategy.upper()} — {scan_label}</b> ({now})",
        f"<b>{len(hits)}</b> hits (Bull Run, ≥{MIN_CONFIDENCE*100:.0f}% conf, ≥{min_confs}/{confs_total})\n",
    ]

    for r in hits[:25]:
        sym = r["symbol"]
        name = r.get("name") or ""
        confs = r.get("confirmations_met", 0)
        signal = r.get("signal", "?").replace("LONG -- ", "")
        lines.append(f"  <b>{sym}</b> ({name}) — {confs}/{confs_total} — {signal}")

    if len(hits) > 25:
        lines.append(f"\n... and {len(hits) - 25} more")

    return "\n".join(lines)


def save_am_results(v1_hits: list, v2_hits: list):
    """Save AM scan symbols to disk for PM confirmation."""
    data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "v1_symbols": [r.get("symbol") for r in v1_hits],
        "v2_symbols": [r.get("symbol") for r in v2_hits],
    }
    with open(AM_RESULTS_FILE, "w") as f:
        json.dump(data, f)
    print(f"[Confirm] Saved {len(data['v1_symbols'])} V1 + {len(data['v2_symbols'])} V2 AM symbols for PM confirmation")


def load_am_results() -> dict:
    """Load AM scan symbols for PM cross-reference."""
    if not os.path.exists(AM_RESULTS_FILE):
        return {"v1_symbols": [], "v2_symbols": []}
    try:
        with open(AM_RESULTS_FILE, "r") as f:
            data = json.load(f)
        # Only use if from today
        if data.get("date") == datetime.now().strftime("%Y-%m-%d"):
            print(f"[Confirm] Loaded {len(data['v1_symbols'])} V1 + {len(data['v2_symbols'])} V2 AM symbols")
            return data
    except Exception:
        pass
    return {"v1_symbols": [], "v2_symbols": []}


def run_session(session: str):
    """
    Run a full scan session (AM or PM).
    session: "am" or "pm"
    """
    today_str = datetime.now().strftime("%m/%d/%Y")
    is_pm = session == "pm"

    # Load AM results for PM confirmation
    am_data = load_am_results() if is_pm else {}
    am_v1_syms = set(am_data.get("v1_symbols", []))
    am_v2_syms = set(am_data.get("v2_symbols", []))

    session_label = "PM Confirmation" if is_pm else "AM"

    # ── V1 Scan (6/8) ──
    print("=" * 60)
    print(f"V1 SCAN ({session_label}) — {today_str}")
    print("=" * 60)

    v1_hits = run_scan("v1", V1_MIN_CONFS)

    confirmed_v1 = am_v1_syms if is_pm else None
    subject_v1 = f"V1 Hits {session_label} {today_str}"
    email_html = format_email(v1_hits, f"All Tickers — {session_label}", "v1", V1_MIN_CONFS, confirmed_v1)
    send_email(subject_v1, email_html)

    telegram_msg = format_telegram(v1_hits, session_label, "v1", V1_MIN_CONFS)
    send_telegram(telegram_msg)

    if v1_hits:
        print(f"\nV1 hits ({len(v1_hits)}): {', '.join(r['symbol'] for r in v1_hits)}")
        if is_pm and am_v1_syms:
            confirmed = [r["symbol"] for r in v1_hits if r["symbol"] in am_v1_syms]
            if confirmed:
                print(f"CONFIRMED from AM: {', '.join(confirmed)}")
    else:
        print("\nNo V1 hits.")

    # ── V2 Scan (8/12) ──
    print()
    print("=" * 60)
    print(f"V2 SCAN ({session_label}) — {today_str}")
    print("=" * 60)

    v2_hits = run_scan("v2", V2_MIN_CONFS)

    confirmed_v2 = am_v2_syms if is_pm else None
    subject_v2 = f"V2 Hits {session_label} {today_str}"
    email_html = format_email(v2_hits, f"All Tickers — {session_label}", "v2", V2_MIN_CONFS, confirmed_v2)
    send_email(subject_v2, email_html)

    telegram_msg = format_telegram(v2_hits, session_label, "v2", V2_MIN_CONFS)
    send_telegram(telegram_msg)

    if v2_hits:
        print(f"\nV2 hits ({len(v2_hits)}): {', '.join(r['symbol'] for r in v2_hits)}")
        if is_pm and am_v2_syms:
            confirmed = [r["symbol"] for r in v2_hits if r["symbol"] in am_v2_syms]
            if confirmed:
                print(f"CONFIRMED from AM: {', '.join(confirmed)}")
    else:
        print("\nNo V2 hits.")

    # Save AM results for PM confirmation
    if not is_pm:
        save_am_results(v1_hits, v2_hits)

    # ── Summary ──
    print()
    print("=" * 60)
    print(f"DONE ({session_label}) — V1: {len(v1_hits)} hits, V2: {len(v2_hits)} hits")
    print("=" * 60)


if __name__ == "__main__":
    # Determine session from CLI arg or time
    session = "am"
    if len(sys.argv) > 1 and sys.argv[1] in ("am", "pm"):
        session = sys.argv[1]
    else:
        # Auto-detect: if after 12 PM CT, it's PM session
        hour = datetime.now().hour
        if hour >= 12:
            session = "pm"

    run_session(session)
