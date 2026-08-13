from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest


class ServingMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "llm_gateway_requests_total",
            "Gateway requests by result.",
            ("endpoint", "result"),
            registry=self.registry,
        )
        self.route_decisions = Counter(
            "llm_gateway_route_decisions_total",
            "Routing decisions by backend and reason.",
            ("backend", "reason"),
            registry=self.registry,
        )
        self.backend_errors = Counter(
            "llm_gateway_backend_errors_total",
            "Backend errors by stage.",
            ("backend", "stage"),
            registry=self.registry,
        )
        self.in_flight = Gauge(
            "llm_gateway_in_flight_requests",
            "Requests admitted by the gateway.",
            registry=self.registry,
        )
        self.backend_outstanding = Gauge(
            "llm_gateway_backend_outstanding_requests",
            "Outstanding requests assigned to each backend.",
            ("backend",),
            registry=self.registry,
        )
        self.ttft = Histogram(
            "llm_gateway_ttft_seconds",
            "Time from gateway admission to the first output token.",
            ("backend",),
            registry=self.registry,
        )
        self.e2e = Histogram(
            "llm_gateway_e2e_seconds",
            "End-to-end request latency.",
            ("backend", "result"),
            registry=self.registry,
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)
