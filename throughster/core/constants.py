import httpx

DEFAULT_LIMITS = httpx.Limits(
    max_keepalive_connections=50,
    max_connections=100,
    keepalive_expiry=10,
)
DEFAULT_TIMEOUT = httpx.Timeout(timeout=600.0, connect=5.0, pool=None)
