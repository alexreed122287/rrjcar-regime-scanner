"""
scheduled_scan.py — Headless scheduled scanner
Runs the full scan on "All Stocks (no ETFs)" and sends email/Telegram
with bullish tickers. Designed to run via GitHub Actions cron.
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


def send_email(subject: str, body_html: str):
    """Send email via SMTP."""
    smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "")
    smtp_password = os.environ.get("SMTP_PASSWORD", "")
    to_email = os.environ.get("ALERT_EMAIL", smtp_user)

    if not smtp_user or not smtp_password:
        print("[Alert] SMTP not configured, skipping email")
        return False

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = smtp_user
    msg["To"] = to_email
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, to_email, msg.as_string())
        print(f"[Alert] Email sent to {to_email}")
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


def run_scan():
    """Run the full scan and return bullish results."""
    tickers = WATCHLISTS.get("All Stocks (no ETFs)", [])
    if not tickers:
        print("[Scan] No tickers in 'All Stocks (no ETFs)' watchlist")
        return []

    total = len(tickers)
    print(f"[Scan] Starting scan of {total:,} tickers at {datetime.now()}")

    def progress(batch_num, total_batches, running):
        bullish = [r for r in running if r.get("signal") in BULLISH_SIGNALS]
        scanned = min(batch_num * 200, total)
        print(f"[Scan] Batch {batch_num}/{total_batches} done ({scanned:,}/{total:,}) — {len(bullish)} bullish so far")

    results = scan_watchlist(
        symbols=tickers,
        interval="1d",
        n_regimes=7,
        min_confirmations=6,
        regime_confirm_bars=2,
        max_workers=10,
        strategy="v2",
        batch_size=200,
        progress_callback=progress,
    )

    bullish = [r for r in results if r.get("signal") in BULLISH_SIGNALS]
    print(f"[Scan] Complete: {len(results)} scanned, {len(bullish)} bullish")
    return bullish


def format_email(bullish: list) -> str:
    """Format bullish results as HTML email."""
    now = datetime.now().strftime("%B %d, %Y %I:%M %p CT")

    if not bullish:
        return f"""
        <h2>RRJCAR Regime Scanner</h2>
        <p><b>Scan Time:</b> {now}</p>
        <p>No bullish signals found today.</p>
        """

    # Group by signal type
    enters = [b for b in bullish if b.get("signal") == "LONG -- ENTER"]
    confirming = [b for b in bullish if b.get("signal") == "LONG -- CONFIRMING"]
    holds = [b for b in bullish if b.get("signal") == "LONG -- HOLD"]

    def ticker_row(r):
        sym = r.get("symbol", "?")
        price = r.get("price", 0)
        regime = r.get("regime_label", "?")
        confs = r.get("confirmations_met", 0)
        change = r.get("price_1d_pct", 0) or 0
        color = "#2dd4bf" if change >= 0 else "#f87171"
        return f'<tr><td><b>{sym}</b></td><td>${price:.2f}</td><td style="color:{color}">{change:+.1f}%</td><td>{regime}</td><td>{confs}/12</td></tr>'

    rows_html = ""

    if enters:
        rows_html += '<tr><td colspan="5" style="background:#1a3a2a;color:#2dd4bf;padding:8px;font-weight:bold;">BUY NOW — ENTER</td></tr>'
        rows_html += "".join(ticker_row(r) for r in enters)

    if confirming:
        rows_html += '<tr><td colspan="5" style="background:#1a2a3a;color:#60a5fa;padding:8px;font-weight:bold;">CONFIRMING</td></tr>'
        rows_html += "".join(ticker_row(r) for r in confirming)

    if holds:
        rows_html += '<tr><td colspan="5" style="background:#2a2a1a;color:#fbbf24;padding:8px;font-weight:bold;">HOLD</td></tr>'
        rows_html += "".join(ticker_row(r) for r in holds)

    return f"""
    <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;background:#101114;color:#e5e7eb;padding:20px;border-radius:8px;">
        <h2 style="color:#2dd4bf;margin-top:0;">RRJCAR Regime Scanner</h2>
        <p style="color:#9ca3af;">{now} | All Stocks (no ETFs)</p>
        <p><b style="color:#2dd4bf;">{len(bullish)}</b> bullish signals found</p>
        <table style="width:100%;border-collapse:collapse;font-size:14px;">
            <tr style="color:#6b7280;font-size:12px;text-transform:uppercase;">
                <th style="text-align:left;padding:6px;">Ticker</th>
                <th style="text-align:left;">Price</th>
                <th style="text-align:left;">1D</th>
                <th style="text-align:left;">Regime</th>
                <th style="text-align:left;">Confs</th>
            </tr>
            {rows_html}
        </table>
        <p style="color:#4b5563;font-size:11px;margin-top:20px;">
            Sent by RRJCAR Regime Scanner via GitHub Actions
        </p>
    </div>
    """


def format_telegram(bullish: list) -> str:
    """Format bullish results as Telegram message."""
    now = datetime.now().strftime("%I:%M %p CT")

    if not bullish:
        return f"<b>RRJCAR Scanner</b> ({now})\nNo bullish signals today."

    enters = [b for b in bullish if b.get("signal") == "LONG -- ENTER"]
    confirming = [b for b in bullish if b.get("signal") == "LONG -- CONFIRMING"]

    lines = [f"<b>RRJCAR Scanner</b> ({now})", f"<b>{len(bullish)}</b> bullish signals\n"]

    if enters:
        lines.append("<b>BUY NOW:</b>")
        for r in enters[:20]:  # Telegram has message size limits
            lines.append(f"  <b>{r['symbol']}</b> ${r.get('price',0):.2f} — {r.get('regime_label','?')}")
        lines.append("")

    if confirming:
        lines.append("<b>CONFIRMING:</b>")
        for r in confirming[:20]:
            lines.append(f"  <b>{r['symbol']}</b> ${r.get('price',0):.2f} — {r.get('regime_label','?')}")

    if len(bullish) > 40:
        lines.append(f"\n... and {len(bullish) - 40} more")

    return "\n".join(lines)


if __name__ == "__main__":
    print("=" * 60)
    print("RRJCAR Scheduled Scan — All Stocks (no ETFs)")
    print("=" * 60)

    bullish = run_scan()

    # Send email
    subject = f"RRJCAR: {len(bullish)} Bullish Signals — {datetime.now().strftime('%m/%d')}"
    email_html = format_email(bullish)
    send_email(subject, email_html)

    # Send Telegram
    telegram_msg = format_telegram(bullish)
    send_telegram(telegram_msg)

    # Also save results to JSON for reference
    if bullish:
        output = [{
            "symbol": r.get("symbol"),
            "price": r.get("price"),
            "signal": r.get("signal"),
            "regime": r.get("regime_label"),
            "confirmations": r.get("confirmations_met"),
        } for r in bullish]
        print(f"\nBullish tickers: {', '.join(r['symbol'] for r in output)}")
    else:
        print("\nNo bullish signals found.")
