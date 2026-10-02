"""Pair a phone (or any other device) with the assistant.

    python -m apps.cli.pair --url https://my-mac.tail1234.ts.net

Creates ACCESS_TOKEN in .env if there isn't one, then prints a pairing link and a
QR code. Scan it with the iPhone camera, open it in Safari, then Share > Add to
Home Screen. The link carries the token, so treat it like a password.

    python -m apps.cli.pair --rotate   # new token: every paired device must re-pair
"""

from __future__ import annotations

import argparse
import re
import secrets
import sys
from urllib.parse import quote

import qrcode

from core.config import ROOT

ENV = ROOT / ".env"


def read_token() -> str | None:
    if not ENV.exists():
        return None
    match = re.search(r"^ACCESS_TOKEN=(\S+)", ENV.read_text(), re.MULTILINE)
    return match.group(1) if match else None


def write_token(token: str) -> None:
    text = ENV.read_text() if ENV.exists() else ""
    line = f"ACCESS_TOKEN={token}"
    if re.search(r"^ACCESS_TOKEN=.*$", text, re.MULTILINE):
        text = re.sub(r"^ACCESS_TOKEN=.*$", line, text, flags=re.MULTILINE)
    else:
        text = text + ("" if text.endswith("\n") or not text else "\n") + line + "\n"
    ENV.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", help="address the phone uses to reach this Mac (https://...)")
    parser.add_argument("--rotate", action="store_true", help="replace the token")
    args = parser.parse_args()

    token = read_token()
    if token is None or args.rotate:
        token = secrets.token_urlsafe(24)
        write_token(token)
        print("New access token saved to .env. Restart the server so it takes effect.\n")

    if not args.url:
        print("Pass --url with the address your phone will use, for example your Tailscale")
        print("address: python -m apps.cli.pair --url https://<mac-name>.<tailnet>.ts.net")
        sys.exit(1)
    if not args.url.startswith("https://"):
        print("Note: without https://, the iPhone won't allow the microphone (typing still works).\n")

    link = f"{args.url.rstrip('/')}/?token={quote(token)}"
    qr = qrcode.QRCode(border=2)
    qr.add_data(link)
    qr.print_ascii(invert=True)
    print(f"\n{link}\n")
    print("On the iPhone: scan with the Camera, open in Safari, then Share > Add to Home Screen.")


if __name__ == "__main__":
    main()
