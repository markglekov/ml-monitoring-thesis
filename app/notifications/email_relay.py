from __future__ import annotations

import json
import smtplib
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.common.config import settings
from app.common.logging import get_logger, setup_logging


logger = get_logger(__name__)

WEBHOOK_REQUESTS_TOTAL = Counter(
    "ml_monitoring_email_webhook_requests_total",
    "Total webhook requests received from Alertmanager.",
    ["result"],
)
EMAIL_NOTIFICATIONS_TOTAL = Counter(
    "ml_monitoring_email_notifications_total",
    "Total email notifications attempted by the relay.",
    ["result", "alert_status"],
)
EMAIL_REQUEST_DURATION_SECONDS = Histogram(
    "ml_monitoring_email_request_duration_seconds",
    "Outbound SMTP request duration.",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)
EMAIL_CONFIGURED = Gauge(
    "ml_monitoring_email_configured",
    "Whether the email relay has enough SMTP configuration to send messages.",
)
EMAIL_LAST_SUCCESS_TIMESTAMP = Gauge(
    "ml_monitoring_email_last_success_timestamp",
    "Unix timestamp of the latest successful email notification.",
)


class AlertmanagerAlert(BaseModel):
    """Single alert item from the Alertmanager webhook payload."""

    status: str
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)
    startsAt: datetime | None = None
    endsAt: datetime | None = None
    generatorURL: str | None = None


class AlertmanagerWebhookPayload(BaseModel):
    """Alertmanager webhook payload received by the email relay."""

    receiver: str
    status: str
    externalURL: str | None = None
    groupKey: str | None = None
    groupLabels: dict[str, str] = Field(default_factory=dict)
    commonLabels: dict[str, str] = Field(default_factory=dict)
    commonAnnotations: dict[str, str] = Field(default_factory=dict)
    alerts: list[AlertmanagerAlert] = Field(default_factory=list)
    truncatedAlerts: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize relay logging and static gauges once per process."""

    setup_logging()
    if settings.smtp_use_ssl and settings.smtp_use_starttls:
        raise RuntimeError("SMTP_USE_SSL and SMTP_USE_STARTTLS cannot both be enabled.")

    configured = is_configured()
    EMAIL_CONFIGURED.set(1.0 if configured else 0.0)
    logger.info(
        "Email relay startup completed. configured=%s host=%s port=%s recipients=%s starttls=%s ssl=%s",
        configured,
        settings.smtp_host,
        settings.smtp_port,
        len(settings.alert_email_to),
        settings.smtp_use_starttls,
        settings.smtp_use_ssl,
    )
    yield


app = FastAPI(
    title="ML Monitoring Email Relay",
    version="0.1.0",
    lifespan=lifespan,
)


def is_configured() -> bool:
    """Return whether the email relay has enough SMTP config to send messages."""

    return bool(settings.smtp_host and settings.smtp_from and settings.alert_email_to)


def format_datetime(value: datetime | None) -> str:
    """Render timestamps in a compact UTC format."""

    if value is None:
        return "n/a"
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.strftime("%Y-%m-%d %H:%M:%SZ")


def build_subject(payload: AlertmanagerWebhookPayload) -> str:
    """Build a concise subject line for an alert group."""

    status = payload.status.upper()
    alertname = payload.commonLabels.get("alertname", "multiple")
    service = payload.commonLabels.get("service", "multiple")
    severity = payload.commonLabels.get("severity", "mixed")
    subject = f"[{status}] {alertname} | service={service} | severity={severity}"
    return subject[:180]


def build_alert_block(alert: AlertmanagerAlert) -> str:
    """Build one formatted text block for a single alert."""

    lines = [
        f"- alertname: {alert.labels.get('alertname', 'unknown')}",
        f"  service: {alert.labels.get('service', 'unknown')}",
        f"  severity: {alert.labels.get('severity', 'unknown')}",
        f"  status: {alert.status}",
        f"  starts_at_utc: {format_datetime(alert.startsAt)}",
    ]
    summary = alert.annotations.get("summary") or alert.annotations.get("description")
    if summary:
        lines.append(f"  summary: {summary}")
    if alert.generatorURL:
        lines.append(f"  generator_url: {alert.generatorURL}")
    return "\n".join(lines)


def build_body(payload: AlertmanagerWebhookPayload) -> str:
    """Render Alertmanager payload into a readable plain-text email body."""

    summary = (
        payload.commonAnnotations.get("summary")
        or payload.commonAnnotations.get("description")
        or f"{len(payload.alerts)} alert(s) in group"
    )
    lines = [
        "ML Monitoring Alert",
        "",
        f"Status: {payload.status.upper()}",
        f"Receiver: {payload.receiver}",
        f"Alerts in group: {len(payload.alerts)}",
        f"Alertname: {payload.commonLabels.get('alertname', 'multiple')}",
        f"Service: {payload.commonLabels.get('service', 'multiple')}",
        f"Severity: {payload.commonLabels.get('severity', 'mixed')}",
        f"Summary: {summary}",
    ]

    if payload.externalURL:
        lines.append(f"Alertmanager URL: {payload.externalURL}")

    if payload.groupKey:
        lines.append(f"Group key: {payload.groupKey}")

    if payload.alerts:
        lines.extend(["", "Alerts:"])
        for alert in payload.alerts[:10]:
            lines.append(build_alert_block(alert))
            lines.append("")

    if payload.truncatedAlerts:
        lines.append(f"{payload.truncatedAlerts} additional alert(s) were truncated by Alertmanager.")

    return "\n".join(lines).strip()


def send_email_notification(subject: str, body: str, alert_status: str) -> dict[str, Any]:
    """Send one email notification via SMTP."""

    if not is_configured():
        raise RuntimeError("Email relay is not configured. SMTP_HOST, SMTP_FROM, and ALERT_EMAIL_TO are required.")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.smtp_from
    message["To"] = ", ".join(settings.alert_email_to)
    message.set_content(body)

    smtp_client_class = smtplib.SMTP_SSL if settings.smtp_use_ssl else smtplib.SMTP
    started_at = time.perf_counter()

    try:
        with smtp_client_class(settings.smtp_host, settings.smtp_port, timeout=settings.smtp_timeout_sec) as smtp:
            smtp.ehlo()
            if settings.smtp_use_starttls:
                smtp.starttls()
                smtp.ehlo()
            if settings.smtp_username:
                smtp.login(settings.smtp_username, settings.smtp_password or "")
            send_result = smtp.send_message(message)

        EMAIL_REQUEST_DURATION_SECONDS.observe(time.perf_counter() - started_at)
        if send_result:
            EMAIL_NOTIFICATIONS_TOTAL.labels(result="failed", alert_status=alert_status).inc()
            raise RuntimeError(f"SMTP rejected recipients: {json.dumps(send_result, ensure_ascii=False)}")

        EMAIL_NOTIFICATIONS_TOTAL.labels(result="sent", alert_status=alert_status).inc()
        EMAIL_LAST_SUCCESS_TIMESTAMP.set(time.time())
        return {
            "status": "sent",
            "recipients": list(settings.alert_email_to),
            "subject": subject,
        }
    except Exception as exc:
        EMAIL_REQUEST_DURATION_SECONDS.observe(time.perf_counter() - started_at)
        EMAIL_NOTIFICATIONS_TOTAL.labels(result="failed", alert_status=alert_status).inc()
        raise RuntimeError(f"SMTP request failed: {exc}") from exc


@app.get("/health")
def health() -> dict[str, Any]:
    """Expose relay readiness information."""

    return {
        "status": "ok",
        "configured": is_configured(),
        "smtp_host": settings.smtp_host,
        "smtp_port": settings.smtp_port,
        "smtp_use_starttls": settings.smtp_use_starttls,
        "smtp_use_ssl": settings.smtp_use_ssl,
        "recipient_count": len(settings.alert_email_to),
    }


@app.get("/metrics")
def metrics() -> Response:
    """Expose Prometheus metrics for the email relay."""

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/alertmanager/webhook")
def alertmanager_webhook(payload: AlertmanagerWebhookPayload) -> dict[str, Any]:
    """Receive grouped alerts from Alertmanager and forward them via email."""

    if not is_configured():
        WEBHOOK_REQUESTS_TOTAL.labels(result="ignored_unconfigured").inc()
        logger.warning("Email relay received webhook but SMTP settings are not configured.")
        return {
            "status": "ignored_unconfigured",
            "alerts_count": len(payload.alerts),
        }

    subject = build_subject(payload)
    body = build_body(payload)
    try:
        email_result = send_email_notification(subject=subject, body=body, alert_status=payload.status)
    except Exception as exc:
        WEBHOOK_REQUESTS_TOTAL.labels(result="failed").inc()
        logger.exception("Email relay failed to deliver alert group.")
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    WEBHOOK_REQUESTS_TOTAL.labels(result="sent").inc()
    logger.info(
        "Email relay delivered alert group. status=%s alerts=%s receiver=%s recipients=%s",
        payload.status,
        len(payload.alerts),
        payload.receiver,
        len(settings.alert_email_to),
    )
    return {
        "status": "sent",
        "alerts_count": len(payload.alerts),
        "recipients": email_result["recipients"],
        "subject": email_result["subject"],
    }
