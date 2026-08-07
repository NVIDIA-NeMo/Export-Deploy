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


import ipaddress
import socket
from urllib.parse import urlparse

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
]


def validate_image_url(url: str) -> None:
    """Raise ValueError if url is not a safe http/https URL.

    Rejects file://, non-http(s) schemes, and URLs that resolve to
    private/link-local/loopback ranges (SSRF guard).
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported image URL scheme '{parsed.scheme}'. Only http and https are allowed.")
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
