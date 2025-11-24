"""Notification utilities for sending alerts via webhooks."""

import os
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)
WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL")


def send_alert(
    model_name: str, status: str, error_msg: Optional[str] = None, run_time: str = "N/A"
):
    """Send alert notification via webhook with model evaluation status."""
    if not WEBHOOK_URL:
        return

    is_failure = status == "FAILURE"
    emoji = ":x:" if is_failure else ":white_check_mark:"
    color = "#FF0000" if is_failure else "#36a64f"

    text = f"{emoji} **Nightly Eval {status}:** {model_name}"
    if is_failure:
        text += f"\n\n**Error:** `{error_msg}`"
    else:
        text += f"\n\n**Duration:** {run_time}"
        text += "\n**Action:** Results uploaded to S3."

    payload = {
        "text": text,
        "attachments": [
            {
                "color": color,
                "fields": [
                    {"title": "Model", "value": model_name, "short": True},
                    {"title": "Status", "value": status, "short": True},
                ],
            }
        ],
    }

    try:
        requests.post(WEBHOOK_URL, json=payload, timeout=5)
    except Exception as e:
        logger.error(f"Alert failed: {e}")
