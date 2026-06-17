"""
No-op Chroma product telemetry client.

chromadb 0.5.x still invokes posthog.capture() even when anonymized_telemetry=False.
posthog 7.x removed the legacy positional capture() signature, which produces:
  "capture() takes 1 positional argument but 3 were given"
"""

from __future__ import annotations

from overrides import override

from chromadb.telemetry.product import ProductTelemetryClient, ProductTelemetryEvent


class NoOpProductTelemetry(ProductTelemetryClient):
    """Drop all Chroma telemetry events without contacting posthog."""

    @override
    def capture(self, event: ProductTelemetryEvent) -> None:
        return