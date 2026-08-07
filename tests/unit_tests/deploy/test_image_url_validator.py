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
from unittest.mock import patch

import pytest

# Load the validator directly by file path so we don't trigger nemo_deploy/__init__.py
# (which requires torch/triton). The module itself is pure stdlib.
_validator_path = pathlib.Path(__file__).resolve().parents[3] / "nemo_deploy" / "multimodal" / "image_url_validator.py"
_spec = importlib.util.spec_from_file_location("image_url_validator", _validator_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
validate_image_url = _mod.validate_image_url


def _mock_resolve(ip_str):
    """Return a patch for socket.gethostbyname that always resolves to ip_str."""
    return patch.object(_mod.socket, "gethostbyname", return_value=ip_str)


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
            "gethostbyname",
            side_effect=socket.gaierror("Name or service not known"),
        ):
            with pytest.raises(ValueError, match="Cannot resolve"):
                validate_image_url("http://nonexistent.invalid/img.jpg")
