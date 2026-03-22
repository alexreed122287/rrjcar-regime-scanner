"""
alerts.py — Email and Telegram notifications for regime changes.
"""

import smtplib
import json
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, List, Optional

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

from settings_manager import load_settings, get_setting

# Track last-known regimes to detect changes
_REGIME_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".regime_cache.json")


def _load_regime_cache() -> Dict[str, int]:
    if os.path.exists(_REGIME_CACHE_FILE):
        try:
            with open(_REGIME_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_regime_cache(cache: Dict[str, int]):
    with open(_REGIME_CACHE_FILE, "w") as f:
        json.dump(cache, f)


def detect_regime_changes(scan_results: List[Dict]) -> List[Dict]:
    """Compare current scan results to cached regimes. Return list of changed tickers."""
    cache = _load_regime_cache()
    changes = []
    new_cache = {}

    for r in scan_results:
        sym = r.get("symbol", "")
        regime_id = r.get("regime_id")
        if regime_id is None:
            continue
        new_cache[sym] = regime_id
        prev = cache.get(sym)
        if prev is not None and prev != regime_id:
            changes.append({
                "symbol": sym,
                "prev_regime": prev,
                "new_regime": regime_id,
                "regime_label": r.get("regime_label", ""),
                "price": r.get("price", 0),
                "confirmations": r.get("confirmations_met", 0),
            })

    _save_regime_cache(new_cache)
    return changes


REGIME_NAMES = {
    0: "Bull Run",
    1: "Mild Bull",
    2: "Bull Trend",
    3: "Neutral / Chop",
    4: "Mild Bear",
    5: "Bear Trend",
    6: "Crash / Capitulation",
}


def _regime_name(rid: int) -> str:
    return REGIME_NAMES.get(rid, f"Regime {rid}")


def _format_alert_text(changes: List[Dict]) -> str:
    """Format regime changes into a readable message."""
    lines = [f"Regime Scanner Alert — {datetime.now().strftime('%Y-%m-%d %H:%M')}",  ""]
    for c in changes:
        arrow = ">" if c["new_regime"] < c["prev_regime"] else "<"  # improving or worsening
        lines.append(
            f"{c['symbol']:6s}  {_regime_name(c['prev_regime'])} {arrow} {_regime_name(c['new_regime'])}  "
            f"@ ${c['price']:.2f}  (conf: {c['confirmations']})"
        )
    lines.append("")
    lines.append(f"{len(changes)} regime change(s) detected.")
    return "\n".join(lines)


def _format_alert_html(changes: List[Dict]) -> str:
    """HTML version for email."""
    rows = ""
    for c in changes:
        color = "#2dd4bf" if c["new_regime"] <= 2 else "#f87171" if c["new_regime"] >= 4 else "#fbbf24"
        rows += (
            f'<tr>'
            f'<td style="padding:4px 8px;font-weight:bold">{c["symbol"]}</td>'
            f'<td style="padding:4px 8px">{_regime_name(c["prev_regime"])}</td>'
            f'<td style="padding:4px 8px;color:{color};font-weight:bold">{_regime_name(c["new_regime"])}</td>'
            f'<td style="padding:4px 8px">${c["price"]:.2f}</td>'
            f'<td style="padding:4px 8px">{c["confirmations"]}</td>'
            f'</tr>'
        )
    return f"""
    <html><body style="font-family:monospace;background:#1a1a2e;color:#e0e0e0;padding:16px">
    <h2 style="color:#2dd4bf">Regime Scanner Alert</h2>
    <p>{datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
    <table style="border-collapse:collapse;width:100%">
    <tr style="border-bottom:1px solid #333">
        <th style="padding:4px 8px;text-align:left">Symbol</th>
        <th style="padding:4px 8px;text-align:left">From</th>
        <th style="padding:4px 8px;text-align:left">To</th>
        <th style="padding:4px 8px;text-align:left">Price</th>
        <th style="padding:4px 8px;text-align:left">Conf</th>
    </tr>
    {rows}
    </table>
    <p style="color:#888">{len(changes)} regime change(s) detected.</p>
    </body></html>
    """


def send_email_alert(changes: List[Dict], settings: Optional[Dict] = None) -> str:
    """Send alert via SMTP email. Returns status message."""
    s = settings or load_settings()
    to_email = s.get("alert_email", "")
    smtp_server = s.get("alert_smtp_server", "smtp.gmail.com")
    smtp_port = s.get("alert_smtp_port", 587)
    smtp_user = s.get("alert_smtp_user", "")
    smtp_pass = s.get("alert_smtp_password", "")

    if not to_email or not smtp_user or not smtp_pass:
        return "Email not configured (missing address or SMTP credentials)."

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"Regime Alert: {len(changes)} change(s) — {datetime.now().strftime('%H:%M')}"
    msg["From"] = smtp_user
    msg["To"] = to_email

    msg.attach(MIMEText(_format_alert_text(changes), "plain"))
    msg.attach(MIMEText(_format_alert_html(changes), "html"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_email, msg.as_string())
        return f"Email sent to {to_email}"
    except Exception as e:
        return f"Email failed: {e}"


def send_telegram_alert(changes: List[Dict], settings: Optional[Dict] = None) -> str:
    """Send alert via Telegram bot. Returns status message."""
    if not HAS_REQUESTS:
        return "Telegram requires `requests` package. Run: pip install requests"

    s = settings or load_settings()
    bot_token = s.get("alert_telegram_bot_token", "")
    chat_id = s.get("alert_telegram_chat_id", "")

    if not bot_token or not chat_id:
        return "Telegram not configured (missing bot token or chat ID)."

    text = _format_alert_text(changes)
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"

    try:
        resp = requests.post(url, json={
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "HTML",
        }, timeout=10)
        if resp.ok:
            return f"Telegram sent to chat {chat_id}"
        return f"Telegram error: {resp.text}"
    except Exception as e:
        return f"Telegram failed: {e}"


def process_alerts(scan_results: List[Dict], settings: Optional[Dict] = None) -> List[str]:
    """
    Main entry point: detect regime changes, filter by settings, send alerts.
    Returns list of status messages.
    """
    s = settings or load_settings()
    if not s.get("alerts_enabled", False):
        return []

    changes = detect_regime_changes(scan_results)
    if not changes:
        return []

    # Filter by user preferences
    min_conf = s.get("alert_min_confirmations", 6)
    filtered = []
    for c in changes:
        if c["confirmations"] < min_conf:
            continue
        if s.get("alert_on_bull_entry") and c["new_regime"] <= 2:
            filtered.append(c)
        elif s.get("alert_on_bear_entry") and c["new_regime"] >= 4:
            filtered.append(c)
        elif s.get("alert_on_regime_change"):
            filtered.append(c)

    if not filtered:
        return []

    statuses = []

    # Email
    if s.get("alert_email"):
        statuses.append(send_email_alert(filtered, s))

    # Telegram
    if s.get("alert_telegram_enabled"):
        statuses.append(send_telegram_alert(filtered, s))

    return statuses
