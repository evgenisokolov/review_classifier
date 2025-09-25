import time
from fastapi import Request, Response
from prometheus_client import REGISTRY, Counter, Histogram

def _endpoint_label(request: Request) -> str:
    """
    Prefer the route's path template (stable) if available, else the raw path.
    For '/ping', both resolve to '/ping', which the test searches for.
    """
    try:
        route = request.scope.get("route")
        if route and getattr(route, "path", None):
            return route.path
    except Exception:
        pass
    # Fallbacks
    return request.url.path or request.scope.get("path", "/unknown")

def create_logging_middleware(logger, registry=REGISTRY):
    REQUEST_COUNT = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "endpoint", "http_status"],
        registry=registry,
    )
    REQUEST_LATENCY = Histogram(
        "http_request_duration_seconds",
        "Request duration in seconds",
        ["method", "endpoint"],
        registry=registry,
    )

    async def logging_middleware(request: Request, call_next):
        start = time.perf_counter()
        response: Response = await call_next(request)
        duration = time.perf_counter() - start

        method = request.method
        endpoint = _endpoint_label(request)
        status_code = response.status_code

        REQUEST_COUNT.labels(
            method=method,
            endpoint=endpoint,
            http_status=str(status_code),
        ).inc()

        REQUEST_LATENCY.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration)

        logger.info(
            "request",
            extra={
                "method": method,
                "endpoint": endpoint,
                "status": status_code,
                "duration_s": round(duration, 3),
            },
        )
        return response

    return logging_middleware