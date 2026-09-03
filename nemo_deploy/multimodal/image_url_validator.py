#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import base64
import ipaddress
import socket
from io import BytesIO
from urllib.parse import urljoin, urlparse

import urllib3

# Ranges that must never be reachable via a request-controlled image URL.
# 169.254.0.0/16 is the cloud IMDS range (AWS/GCP/Azure 169.254.169.254) —
# the primary SSRF target in cloud deployments.
_BLOCKED_NETWORKS = [
    ipaddress.ip_network("127.0.0.0/8"),  # loopback — server's own local services
    ipaddress.ip_network("10.0.0.0/8"),  # RFC 1918 private
    ipaddress.ip_network("172.16.0.0/12"),  # RFC 1918 private
    ipaddress.ip_network("192.168.0.0/16"),  # RFC 1918 private
    ipaddress.ip_network("169.254.0.0/16"),  # link-local / cloud IMDS
    ipaddress.ip_network("::1/128"),  # IPv6 loopback
    ipaddress.ip_network("fc00::/7"),  # IPv6 unique-local
    ipaddress.ip_network("fe80::/10"),  # IPv6 link-local (IPv6 cloud IMDS equivalent)
]


def _is_blocked_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True if ip must not be reachable via an image URL fetch."""
    if any(ip in net for net in _BLOCKED_NETWORKS):
        return True
    return not ip.is_global or ip.is_multicast or ip.is_reserved or ip.is_unspecified


def _resolve_and_validate(hostname: str) -> list[tuple[int, str]]:
    """Resolve hostname to every A/AAAA address and reject if any is blocked.

    Returns a list of (address_family, ip_string) tuples for all resolved
    addresses. Validating every returned address (rather than just the
    first) guards against DNS answers that mix a public address with a
    private/link-local one (DNS rebinding / multi-answer SSRF).
    """
    try:
        infos = socket.getaddrinfo(hostname, None)
    except socket.gaierror as exc:
        raise ValueError(f"Cannot resolve image URL hostname '{hostname}': {exc}") from exc
    resolved: list[tuple[int, str]] = []
    seen: set[str] = set()
    for family, _, _, _, sockaddr in infos:
        ip_str = sockaddr[0]
        if ip_str in seen:
            continue
        seen.add(ip_str)
        ip = ipaddress.ip_address(ip_str)
        if _is_blocked_ip(ip):
            raise ValueError(
                f"Image URL hostname '{hostname}' resolves to a blocked address ({ip}). "
                "Private, loopback, link-local, and other non-global addresses are not allowed."
            )
        resolved.append((family, ip_str))
    if not resolved:
        raise ValueError(f"Cannot resolve image URL hostname '{hostname}': no addresses returned.")
    return resolved


def validate_image_url(url: str) -> None:
    """Raise ValueError if url is not a safe http/https URL.

    Rejects file://, non-http(s) schemes, and URLs that resolve to
    private/link-local/loopback ranges (SSRF guard). This is a point-in-time
    check; use fetch_image_bytes_safely()/fetch_image_data_uri_safely() to
    actually retrieve the image, since those pin validation to the
    connection itself and guard redirects.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported image URL scheme '{parsed.scheme}'. Only http and https are allowed.")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("Image URL has no hostname.")
    _resolve_and_validate(hostname)


def _pinned_request(url: str, hostname: str, ip: str, family: int, timeout: float) -> urllib3.HTTPResponse:
    """Issue a GET request connected directly to ip, without following redirects.

    Connecting to the pre-validated IP directly (instead of letting the HTTP
    client re-resolve hostname) closes the TOCTOU window between validation
    and connection that DNS rebinding exploits. The Host header and TLS SNI
    still use the original hostname so virtual hosting and certificate
    validation behave normally.
    """
    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    pool_host = f"[{ip}]" if family == socket.AF_INET6 else ip
    headers = {"Host": hostname}
    if parsed.scheme == "https":
        pool = urllib3.HTTPSConnectionPool(
            pool_host,
            port=port,
            timeout=timeout,
            assert_hostname=hostname,
            server_hostname=hostname,
            cert_reqs="CERT_REQUIRED",
        )
    else:
        pool = urllib3.HTTPConnectionPool(pool_host, port=port, timeout=timeout)
    try:
        return pool.request("GET", path, headers=headers, redirect=False, preload_content=True)
    finally:
        pool.close()


def fetch_image_bytes_safely(url: str, timeout: float = 5, max_redirects: int = 5) -> bytes:
    """Safely fetch image bytes from url, guarding against SSRF.

    Unlike validate_image_url() followed by a separate request, this
    resolves and validates every hostname (including redirect targets)
    immediately before connecting, connects directly to the validated IP
    address, and never auto-follows redirects — each hop is re-validated
    from scratch.
    """
    current_url = url
    for _ in range(max_redirects + 1):
        parsed = urlparse(current_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"Unsupported image URL scheme '{parsed.scheme}'. Only http and https are allowed.")
        hostname = parsed.hostname
        if not hostname:
            raise ValueError("Image URL has no hostname.")
        resolved = _resolve_and_validate(hostname)
        family, ip = resolved[0]
        response = _pinned_request(current_url, hostname, ip, family, timeout)
        if response.status in (301, 302, 303, 307, 308):
            location = response.headers.get("Location")
            if not location:
                raise ValueError(f"Redirect response from '{current_url}' is missing a Location header.")
            current_url = urljoin(current_url, location)
            continue
        if response.status >= 400:
            raise ValueError(f"Failed to fetch image URL '{current_url}': HTTP {response.status}")
        return response.data
    raise ValueError(f"Exceeded max redirects ({max_redirects}) while fetching image URL '{url}'.")


def fetch_image_data_uri_safely(url: str, timeout: float = 5, max_redirects: int = 5) -> str:
    """Safely fetch an image and return it as a base64 data URI.

    Useful for handing an image to code (e.g. third-party libraries) that
    would otherwise perform its own, unguarded network fetch given a raw
    URL — the network I/O happens here, under the SSRF guard, instead.
    """
    from PIL import Image

    data = fetch_image_bytes_safely(url, timeout=timeout, max_redirects=max_redirects)
    image_format = (Image.open(BytesIO(data)).format or "JPEG").lower()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/{image_format};base64,{encoded}"
