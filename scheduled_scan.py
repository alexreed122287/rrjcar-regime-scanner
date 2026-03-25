"""
scheduled_scan.py — Headless scheduled scanner
Runs two scans:
  1) "All Stocks (no ETFs)" with min avg volume 500K
  2) "All ETFs" without volume filter
Filters: Bull Run / Bull Trend regimes only, ≥70% confidence, ≥10 confirmations.
Sends results sorted by top buy (most confirmations) to least via email.
Designed to run via GitHub Actions cron weekdays at 1:25 PM CST.
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
MIN_CONFIDENCE = 0.70
ALLOWED_REGIMES = {"Bull Run", "Bull Trend"}
MIN_CONFIRMATIONS = 10
RECIPIENTS = [
    "alexander.s.reed@gmail.com",
    "jasoncolvin7.0@gmail.com",
    "ruizrk@yahoo.com",
]


def send_email(subject: str, body_html: str, recipients: list[str] = None):
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


def filter_results(results: list) -> list:
    """
    Filter scan results to only include:
    - Bull Run or Bull Trend regime
    - ≥70% regime confidence
    - ≥10 confirmations met
    Then sort by confirmations descending (top buy first).
    """
    filtered = []
    for r in results:
        if r.get("regime_label") not in ALLOWED_REGIMES:
            continue
        if (r.get("regime_confidence") or 0) < MIN_CONFIDENCE:
            continue
        if (r.get("confirmations_met") or 0) < MIN_CONFIRMATIONS:
            continue
        filtered.append(r)

    # Sort by confirmations descending (top buy first)
    filtered.sort(key=lambda r: -(r.get("confirmations_met") or 0))
    return filtered


def run_scan(watchlist_name: str, min_avg_volume: int = None) -> list:
    """Run a scan on the given watchlist and return filtered results."""
    tickers = WATCHLISTS.get(watchlist_name, [])
    if not tickers:
        print(f"[Scan] No tickers in '{watchlist_name}' watchlist")
        return []

    total = len(tickers)
    print(f"[Scan] Starting scan of {total:,} tickers ({watchlist_name}) at {datetime.now()}")

    def progress(batch_num, total_batches, running):
        bullish = [r for r in running if r.get("signal") in BULLISH_SIGNALS]
        scanned = min(batch_num * 200, total)
        print(f"[Scan] Batch {batch_num}/{total_batches} done ({scanned:,}/{total:,}) — {len(bullish)} bullish so far")

    results = scan_watchlist(
        symbols=tickers,
        interval="1d",
        n_regimes=7,
        min_confirmations=MIN_CONFIRMATIONS,
        regime_confirm_bars=2,
        max_workers=10,
        strategy="v2",
        batch_size=200,
        progress_callback=progress,
        bullish_only=True,
        min_avg_volume=min_avg_volume,
    )

    print(f"[Scan] Raw bullish: {len(results)} — applying regime/confidence/confirmation filters...")
    filtered = filter_results(results)
    print(f"[Scan] After filters: {len(filtered)} hits (Bull Run/Trend, ≥{MIN_CONFIDENCE*100:.0f}% conf, ≥{MIN_CONFIRMATIONS} confs)")
    return filtered


def format_email(hits: list, scan_label: str) -> str:
    """Format scan hits as HTML email, sorted top buy to least."""
    now = datetime.now().strftime("%B %d, %Y %I:%M %p CT")

    if not hits:
        return f"""
        <div style="font-family:Arial,sans-serif;max-width:600px;margin:0 auto;background:#101114;color:#e5e7eb;padding:20px;border-radius:8px;">
            <h2 style="color:#2dd4bf;margin-top:0;">RRJCAR Regime Scanner</h2>
            <p><b>Scan Time:</b> {now}</p>
            <p>No hits found for {scan_label} today.</p>
        </div>
        """

    def ticker_row(r, rank):
        sym = r.get("symbol", "?")
        price = r.get("price", 0)
        regime = r.get("regime_label", "?")
        conf = r.get("regime_confidence", 0)
        confs = r.get("confirmations_met", 0)
        signal = r.get("signal", "?")
        change = r.get("change_1d") or 0
        color = "#2dd4bf" if change >= 0 else "#f87171"
        signal_short = signal.replace("LONG -- ", "")
        return (
            f'<tr style="border-bottom:1px solid #1e2028;">'
            f'<td style="padding:6px;">{rank}</td>'
            f'<td style="padding:6px;"><b>{sym}</b></td>'
            f'<td style="padding:6px;">${price:.2f}</td>'
            f'<td style="padding:6px;color:{color}">{change:+.1f}%</td>'
            f'<td style="padding:6px;">{regime}</td>'
            f'<td style="padding:6px;">{conf*100:.0f}%</td>'
            f'<td style="padding:6px;">{confs}/12</td>'
            f'<td style="padding:6px;">{signal_short}</td>'
            f'</tr>'
        )

    rows_html = "".join(ticker_row(r, i + 1) for i, r in enumerate(hits))

    return f"""
    <div style="font-family:Arial,sans-serif;max-width:700px;margin:0 auto;background:#101114;color:#e5e7eb;padding:20px;border-radius:8px;">
        <h2 style="color:#2dd4bf;margin-top:0;">RRJCAR Regime Scanner</h2>
        <p style="color:#9ca3af;">{now} | {scan_label}</p>
        <p style="color:#9ca3af;font-size:12px;">
            Filters: Bull Run / Bull Trend only | ≥{MIN_CONFIDENCE*100:.0f}% confidence | ≥{MIN_CONFIRMATIONS} confirmations | Min volume: {'500K' if 'Stock' in scan_label else 'None'}
        </p>
        <p><b style="color:#2dd4bf;">{len(hits)}</b> hits — sorted top buy to least</p>
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <tr style="color:#6b7280;font-size:11px;text-transform:uppercase;">
                <th style="text-align:left;padding:6px;">#</th>
                <th style="text-align:left;padding:6px;">Ticker</th>
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


def format_telegram(hits: list, scan_label: str) -> str:
    """Format hits as Telegram message."""
    now = datetime.now().strftime("%I:%M %p CT")

    if not hits:
        return f"<b>RRJCAR Scanner</b> ({now})\nNo hits for {scan_label} today."

    lines = [
        f"<b>RRJCAR Scanner — {scan_label}</b> ({now})",
        f"<b>{len(hits)}</b> hits (Bull Run/Trend, ≥{MIN_CONFIDENCE*100:.0f}% conf, ≥{MIN_CONFIRMATIONS} confs)\n",
    ]

    for r in hits[:25]:  # Telegram message size limits
        sym = r["symbol"]
        confs = r.get("confirmations_met", 0)
        regime = r.get("regime_label", "?")
        signal = r.get("signal", "?").replace("LONG -- ", "")
        lines.append(f"  <b>{sym}</b> — {regime} — {confs}/12 — {signal}")

    if len(hits) > 25:
        lines.append(f"\n... and {len(hits) - 25} more")

    return "\n".join(lines)


if __name__ == "__main__":
    today_str = datetime.now().strftime("%m/%d/%Y")

    # ── Scan 1: All Stocks with min avg volume 500K ──
    print("=" * 60)
    print(f"SCAN 1: Stocks Hits — {today_str}")
    print("=" * 60)

    stock_hits = run_scan("All Stocks (no ETFs)", min_avg_volume=500_000)

    subject_stocks = f"Stocks Hits {today_str}"
    email_html = format_email(stock_hits, "All Stocks (no ETFs)")
    send_email(subject_stocks, email_html)

    telegram_msg = format_telegram(stock_hits, "Stocks Hits")
    send_telegram(telegram_msg)

    if stock_hits:
        print(f"\nStocks hits ({len(stock_hits)}): {', '.join(r['symbol'] for r in stock_hits)}")
    else:
        print("\nNo stock hits found.")

    # ── Scan 2: All ETFs without volume filter ──
    print()
    print("=" * 60)
    print(f"SCAN 2: ETF Hits — {today_str}")
    print("=" * 60)

    etf_hits = run_scan("All ETFs", min_avg_volume=0)

    subject_etfs = f"ETF Hits {today_str}"
    email_html = format_email(etf_hits, "All ETFs (no Stocks)")
    send_email(subject_etfs, email_html)

    telegram_msg = format_telegram(etf_hits, "ETF Hits")
    send_telegram(telegram_msg)

    if etf_hits:
        print(f"\nETF hits ({len(etf_hits)}): {', '.join(r['symbol'] for r in etf_hits)}")
    else:
        print("\nNo ETF hits found.")

    # ── Summary ──
    print()
    print("=" * 60)
    print(f"DONE — {len(stock_hits)} stock hits, {len(etf_hits)} ETF hits")
    print("=" * 60)
