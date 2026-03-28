from __future__ import annotations

from datetime import datetime, timezone

import pytest
from fastapi import HTTPException

from app.notifications import email_relay


def build_payload() -> email_relay.AlertmanagerWebhookPayload:
    return email_relay.AlertmanagerWebhookPayload(
        receiver="email-relay",
        status="firing",
        externalURL="http://alertmanager:9093",
        commonLabels={"alertname": "HighDrift", "service": "monitoring", "severity": "warning"},
        commonAnnotations={"summary": "Drift exceeded threshold"},
        alerts=[
            email_relay.AlertmanagerAlert(
                status="firing",
                labels={"alertname": "HighDrift", "service": "monitoring", "severity": "warning"},
                annotations={"summary": "Drift exceeded threshold"},
                startsAt=datetime(2026, 3, 28, 12, 0, tzinfo=timezone.utc),
            )
        ],
    )


def test_build_subject_and_body_include_key_alert_fields() -> None:
    payload = build_payload()

    subject = email_relay.build_subject(payload)
    body = email_relay.build_body(payload)

    assert subject == "[FIRING] HighDrift | service=monitoring | severity=warning"
    assert "ML Monitoring Alert" in body
    assert "Status: FIRING" in body
    assert "Alertmanager URL: http://alertmanager:9093" in body
    assert "summary: Drift exceeded threshold" in body


def test_alertmanager_webhook_ignores_unconfigured_relay(monkeypatch) -> None:
    monkeypatch.setattr(email_relay, "is_configured", lambda: False)

    response = email_relay.alertmanager_webhook(build_payload())

    assert response == {
        "status": "ignored_unconfigured",
        "alerts_count": 1,
    }


def test_alertmanager_webhook_returns_sent_result(monkeypatch) -> None:
    monkeypatch.setattr(email_relay, "is_configured", lambda: True)
    monkeypatch.setattr(
        email_relay,
        "send_email_notification",
        lambda subject, body, alert_status: {
            "status": "sent",
            "recipients": ["owner@example.com"],
            "subject": subject,
        },
    )

    response = email_relay.alertmanager_webhook(build_payload())

    assert response["status"] == "sent"
    assert response["alerts_count"] == 1
    assert response["recipients"] == ["owner@example.com"]
    assert response["subject"] == "[FIRING] HighDrift | service=monitoring | severity=warning"


def test_alertmanager_webhook_raises_http_502_on_send_failure(monkeypatch) -> None:
    monkeypatch.setattr(email_relay, "is_configured", lambda: True)

    def fail_send(subject: str, body: str, alert_status: str):
        raise RuntimeError("smtp failed")

    monkeypatch.setattr(email_relay, "send_email_notification", fail_send)

    with pytest.raises(HTTPException) as exc_info:
        email_relay.alertmanager_webhook(build_payload())

    assert exc_info.value.status_code == 502
    assert "smtp failed" in exc_info.value.detail
