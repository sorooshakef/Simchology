"""Keep participant data on the local machine unless the user explicitly opts in."""

from __future__ import annotations

import ipaddress
from urllib.parse import urlparse

from .._utils.warn import warn
from ..exceptions import RemoteNotAllowedError, SimchologyPrivacyWarning

_WARNED_HOSTS: set[str] = set()


def host_of(url: str) -> str:
    """The host name of ``url`` (also for scheme-less values such as ``"localhost:11434"``)."""
    url = str(url).strip()
    if "://" not in url:
        url = "http://" + url
    return (urlparse(url).hostname or "").lower()


def is_local_url(url: str) -> bool:
    """True if ``url`` points to this machine (localhost, 127.0.0.0/8, ::1, 0.0.0.0)."""
    host = host_of(url)
    if host in ("localhost", "0.0.0.0", "::", "") or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_unspecified


def enforce_remote_policy(url: str, allow_remote: bool, what: str = "This method") -> None:
    """Refuse to send texts to a non-local ``url`` unless ``allow_remote`` is True.

    When allowed, a :class:`SimchologyPrivacyWarning` is shown once per host, because
    participant texts will leave the machine.
    """
    if is_local_url(url):
        return
    host = host_of(url)
    if not allow_remote:
        raise RemoteNotAllowedError(
            f"{what} would send the texts to {host!r}, which is not this computer. "
            "Simchology keeps participant data on your machine by default. If your ethics "
            "approval and data-protection agreements allow sending the texts to this "
            "service, pass allow_remote=True (e.g. simchology.assess(..., allow_remote=True))."
        )
    if host not in _WARNED_HOSTS:
        _WARNED_HOSTS.add(host)
        warn(
            f"Participant texts will be sent to {host!r}. Make sure this is covered by your "
            "consent forms and data-protection agreements. Results of hosted models may "
            "also change when the provider updates or retires the model.",
            SimchologyPrivacyWarning,
        )
