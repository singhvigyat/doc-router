"""Parse .eml files into a flat sender/recipient/subject/body dict."""

from __future__ import annotations

import email
from email import policy
from pathlib import Path


def _extract_body(message: email.message.EmailMessage) -> str:
    """Prefer the first text/plain part; fall back to text/html if needed.

    A single .eml file is one RFC 5322 message. Multipart MIME is nested
    *parts* (plain + html + attachments), not extra pages — we do not try
    to stitch a thread of replies together.
    """
    if message.is_multipart():
        html_fallback = ""
        for part in message.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                return str(part.get_content())
            if content_type == "text/html" and not html_fallback:
                html_fallback = str(part.get_content())
        return html_fallback

    content = message.get_content()
    return content if isinstance(content, str) else ""


def parse_email(path: str) -> dict:
    """Return sender, recipient, subject, and body from an .eml file.

    The file is opened in binary mode so the stdlib email parser can honor
    the charset declared in the MIME headers instead of assuming UTF-8.
    `policy.default` yields an EmailMessage that auto-decodes RFC 2047
    encoded headers (e.g. Subject: =?UTF-8?B?...?=).
    """
    with open(path, "rb") as handle:
        message = email.message_from_binary_file(handle, policy=policy.default)

    return {
        "sender": message.get("From", "") or "",
        "recipient": message.get("To", "") or "",
        "subject": message.get("Subject", "") or "",
        "body": _extract_body(message),
    }


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[2]
    sample_path = repo_root / "data" / "raw" / "samples" / "sample.eml"
    print(parse_email(str(sample_path)))
