import warnings

import pytest

from simchology.exceptions import RemoteNotAllowedError, SimchologyPrivacyWarning
from simchology.llm import netpolicy
from simchology.llm.netpolicy import enforce_remote_policy, host_of, is_local_url


@pytest.mark.parametrize("url", [
    "http://localhost:11434", "localhost:8009", "http://127.0.0.1:1234/v1", "http://127.8.9.1",
    "http://[::1]:8009", "http://0.0.0.0:11434", "https://kev.localhost/v1", "LOCALHOST",
])
def test_local_urls(url):
    assert is_local_url(url)


@pytest.mark.parametrize("url", [
    "https://api.openai.com/v1", "http://192.168.1.20:11434", "http://10.0.0.5:8009",
    "https://openrouter.ai/api", "my-server:8009",
])
def test_remote_urls(url):
    assert not is_local_url(url)


def test_host_of_without_scheme():
    assert host_of("example.org:8009") == "example.org"


def test_remote_refused_without_opt_in():
    with pytest.raises(RemoteNotAllowedError, match="allow_remote=True"):
        enforce_remote_policy("https://api.openai.com/v1", False, "The test method")


def test_local_never_warns():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        enforce_remote_policy("http://localhost:1234", False)


def test_privacy_warning_once_per_host(monkeypatch):
    monkeypatch.setattr(netpolicy, "_WARNED_HOSTS", set())
    with pytest.warns(SimchologyPrivacyWarning, match="api.example.com"):
        enforce_remote_policy("https://api.example.com/v1", True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        enforce_remote_policy("https://api.example.com/v2", True)
    with pytest.warns(SimchologyPrivacyWarning):
        enforce_remote_policy("https://other.example.com", True)
