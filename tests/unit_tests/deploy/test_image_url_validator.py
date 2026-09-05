# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import importlib.util
import pathlib
import socket
from unittest.mock import MagicMock, patch

import pytest

# Load the validator directly by file path so we don't trigger nemo_deploy/__init__.py
# (which requires torch/triton). The module itself is pure stdlib + urllib3.
_validator_path = pathlib.Path(__file__).resolve().parents[3] / "nemo_deploy" / "multimodal" / "image_url_validator.py"
_spec = importlib.util.spec_from_file_location("image_url_validator", _validator_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
validate_image_url = _mod.validate_image_url
fetch_image_bytes_safely = _mod.fetch_image_bytes_safely


def _addrinfo(family, ip_str):
    return (family, socket.SOCK_STREAM, 6, "", (ip_str, 0))


def _mock_resolve(*ip_strs, family=socket.AF_INET):
    """Return a patch for socket.getaddrinfo resolving to the given IP(s)."""
    infos = [_addrinfo(family, ip) for ip in ip_strs]
    return patch.object(_mod.socket, "getaddrinfo", return_value=infos)


class TestBlockedSchemes:
    def test_file_scheme_rejected(self):
        with pytest.raises(ValueError, match="scheme"):
            validate_image_url("file:///etc/passwd")

    def test_ftp_scheme_rejected(self):
        with pytest.raises(ValueError, match="scheme"):
            validate_image_url("ftp://example.com/img.png")

    def test_no_scheme_rejected(self):
        with pytest.raises(ValueError, match="scheme"):
            validate_image_url("example.com/img.png")


class TestBlockedRanges:
    def test_loopback_ipv4_rejected(self):
        with _mock_resolve("127.0.0.1"):
            with pytest.raises(ValueError, match="blocked address"):
                validate_image_url("http://localhost/img.jpg")

    def test_loopback_other_subnet_rejected(self):
        with _mock_resolve("127.1.2.3"):
            with pytest.raises(ValueError, match="blocked address"):
                validate_image_url("http://internal.local/img.jpg")

    def test_cloud_imds_rejected(self):
        # 169.254.169.254 is the AWS/GCP/Azure metadata service
        with _mock_resolve("169.254.169.254"):
            with pytest.raises(ValueError, match="blocked address"):
                validate_image_url("http://169.254.169.254/latest/meta-data/")

    def test_link_local_rejected(self):
        with _mock_resolve("169.254.0.1"):
            with pytest.raises(ValueError, match="blocked address"):
                validate_image_url("http://169.254.0.1/img.jpg")

    def test_rfc1918_10_rejected(self):
        with _mock_resolve("10.0.0.1"):
            with pytest.raises(ValueError, match="blocked address"):
                validate_image_url("http://internal.corp/img.jpg")

    def test_rfc1918_172_rejected(self):
        with _mock_resolve("172.16.0.1"):
            with pytest.raises(ValueError, match="blocked address"):
                validate_image_url("http://internal.corp/img.jpg")

    def test_rfc1918_192_rejected(self):
        with _mock_resolve("192.168.1.1"):
            with pytest.raises(ValueError, match="blocked address"):
                validate_image_url("http://192.168.1.1/img.jpg")


class TestIPv6:
    def test_ipv6_loopback_rejected(self):
        with _mock_resolve("::1", family=socket.AF_INET6):
            with pytest.raises(ValueError, match="blocked address"):
                validate_image_url("http://ipv6-loopback/img.jpg")

    def test_ipv6_link_local_rejected(self):
        with _mock_resolve("fe80::1", family=socket.AF_INET6):
            with pytest.raises(ValueError, match="blocked address"):
                validate_image_url("http://ipv6-link-local/img.jpg")

    def test_ipv6_unique_local_rejected(self):
        with _mock_resolve("fc00::1", family=socket.AF_INET6):
            with pytest.raises(ValueError, match="blocked address"):
                validate_image_url("http://ipv6-ula/img.jpg")

    def test_ipv6_global_allowed(self):
        with _mock_resolve("2606:4700:4700::1111", family=socket.AF_INET6):
            validate_image_url("http://ipv6-public/img.jpg")  # must not raise


class TestMultipleDnsAnswers:
    def test_mixed_public_and_blocked_answer_rejected(self):
        """One public + one blocked answer for the same host must be rejected.

        Guards against DNS rebinding / multi-answer SSRF where only the
        first-returned address would look safe.
        """
        infos = [_addrinfo(socket.AF_INET, "93.184.216.34"), _addrinfo(socket.AF_INET, "169.254.169.254")]
        with patch.object(_mod.socket, "getaddrinfo", return_value=infos):
            with pytest.raises(ValueError, match="blocked address"):
                validate_image_url("http://multi-answer.example.com/img.jpg")

    def test_all_public_answers_allowed(self):
        infos = [_addrinfo(socket.AF_INET, "93.184.216.34"), _addrinfo(socket.AF_INET, "1.2.3.4")]
        with patch.object(_mod.socket, "getaddrinfo", return_value=infos):
            validate_image_url("http://multi-answer.example.com/img.jpg")  # must not raise


class TestAllowedUrls:
    def test_public_https_allowed(self):
        with _mock_resolve("93.184.216.34"):  # example.com
            validate_image_url("https://example.com/image.jpg")  # must not raise

    def test_public_http_allowed(self):
        with _mock_resolve("1.2.3.4"):
            validate_image_url("http://cdn.example.com/image.png")  # must not raise


class TestNoHostname:
    def test_url_without_hostname_rejected(self):
        with pytest.raises(ValueError, match="hostname"):
            validate_image_url("http:///image.jpg")

    def test_dns_failure_rejected(self):
        with patch.object(
            _mod.socket,
            "getaddrinfo",
            side_effect=socket.gaierror("Name or service not known"),
        ):
            with pytest.raises(ValueError, match="Cannot resolve"):
                validate_image_url("http://nonexistent.invalid/img.jpg")


def _mock_pinned_response(status, data=b"", location=None):
    response = MagicMock()
    response.status = status
    response.data = data
    response.headers = {"Location": location} if location else {}
    return response


class TestFetchImageBytesSafely:
    def test_fetch_success(self):
        with _mock_resolve("93.184.216.34"):
            with patch.object(_mod, "_pinned_request", return_value=_mock_pinned_response(200, b"imgdata")):
                result = fetch_image_bytes_safely("http://example.com/image.jpg")
        assert result == b"imgdata"

    def test_fetch_rejects_redirect_to_blocked_address(self):
        """A public URL redirecting to a blocked address must be rejected.

        Each redirect hop is independently resolved and validated, so the
        redirect target's own DNS resolution (to a blocked IP) is what
        trips the guard.
        """

        def fake_getaddrinfo(host, *_args, **_kwargs):
            if host == "public.example.com":
                return [_addrinfo(socket.AF_INET, "93.184.216.34")]
            return [_addrinfo(socket.AF_INET, "169.254.169.254")]

        with patch.object(_mod.socket, "getaddrinfo", side_effect=fake_getaddrinfo):
            with patch.object(
                _mod,
                "_pinned_request",
                return_value=_mock_pinned_response(302, location="http://169.254.169.254/latest/meta-data/"),
            ):
                with pytest.raises(ValueError, match="blocked address"):
                    fetch_image_bytes_safely("http://public.example.com/image.jpg")

    def test_fetch_follows_redirect_chain_to_public_address(self):
        def fake_getaddrinfo(host, *_args, **_kwargs):
            return [_addrinfo(socket.AF_INET, "93.184.216.34")]

        responses = [
            _mock_pinned_response(302, location="http://public.example.com/final.jpg"),
            _mock_pinned_response(200, data=b"final-bytes"),
        ]
        with patch.object(_mod.socket, "getaddrinfo", side_effect=fake_getaddrinfo):
            with patch.object(_mod, "_pinned_request", side_effect=responses):
                result = fetch_image_bytes_safely("http://public.example.com/image.jpg")
        assert result == b"final-bytes"

    def test_fetch_raises_on_missing_location_header(self):
        with _mock_resolve("93.184.216.34"):
            with patch.object(_mod, "_pinned_request", return_value=_mock_pinned_response(302)):
                with pytest.raises(ValueError, match="Location"):
                    fetch_image_bytes_safely("http://example.com/image.jpg")

    def test_fetch_raises_on_too_many_redirects(self):
        def fake_getaddrinfo(host, *_args, **_kwargs):
            return [_addrinfo(socket.AF_INET, "93.184.216.34")]

        with patch.object(_mod.socket, "getaddrinfo", side_effect=fake_getaddrinfo):
            with patch.object(
                _mod,
                "_pinned_request",
                return_value=_mock_pinned_response(302, location="http://public.example.com/next.jpg"),
            ):
                with pytest.raises(ValueError, match="redirects"):
                    fetch_image_bytes_safely("http://public.example.com/image.jpg", max_redirects=2)

    def test_fetch_raises_on_http_error_status(self):
        with _mock_resolve("93.184.216.34"):
            with patch.object(_mod, "_pinned_request", return_value=_mock_pinned_response(404)):
                with pytest.raises(ValueError, match="404"):
                    fetch_image_bytes_safely("http://example.com/image.jpg")
