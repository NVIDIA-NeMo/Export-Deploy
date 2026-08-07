import ipaddress
import socket
from urllib.parse import urlparse

# Ranges that must never be reachable via a request-controlled image URL.
# 169.254.0.0/16 is the cloud IMDS range (AWS/GCP/Azure 169.254.169.254) —
# the primary SSRF target in cloud deployments.
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),    # loopback — server's own local services
    ipaddress.ip_network("10.0.0.0/8"),     # RFC 1918 private
    ipaddress.ip_network("172.16.0.0/12"),  # RFC 1918 private
    ipaddress.ip_network("192.168.0.0/16"), # RFC 1918 private
    ipaddress.ip_network("169.254.0.0/16"), # link-local / cloud IMDS
    ipaddress.ip_network("::1/128"),        # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),       # IPv6 unique-local
]


def validate_image_url(url: str) -> None:
    """Raise ValueError if url is not a safe http/https URL.

    Rejects file://, non-http(s) schemes, and URLs that resolve to
    private/link-local/loopback ranges (SSRF guard).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"Unsupported image URL scheme '{parsed.scheme}'. "
            "Only http and https are allowed."
        )
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Image URL has no hostname.")
    try:
        resolved_ip = ipaddress.ip_address(socket.gethostbyname(hostname))
    except (socket.gaierror, ValueError) as exc:
        raise ValueError(f"Cannot resolve image URL hostname '{hostname}': {exc}") from exc
    for net in _BLOCKED_NETWORKS:
        if resolved_ip in net:
            raise ValueError(
                f"Image URL resolves to a blocked address ({resolved_ip}). "
                "Private, loopback, and link-local addresses are not allowed."
            )
