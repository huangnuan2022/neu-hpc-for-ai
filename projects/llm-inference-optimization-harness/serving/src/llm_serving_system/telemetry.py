from __future__ import annotations

import os

from opentelemetry import trace


def configure_tracing() -> None:
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    if not endpoint:
        return
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:
        raise RuntimeError("Install the telemetry extra to export OTLP traces") from exc

    provider = TracerProvider(resource=Resource.create({"service.name": "llm-serving-gateway"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
    trace.set_tracer_provider(provider)


def tracer() -> trace.Tracer:
    return trace.get_tracer("llm_serving_system.gateway")
