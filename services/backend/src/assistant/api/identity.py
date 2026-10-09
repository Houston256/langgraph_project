from typing import Protocol

from fastapi import Request


class IdentityProvider(Protocol):
    def __call__(self, request: Request) -> str:
        """User id for the request."""
        ...


class TrustedHeaderIdentity:
    """User id from a header the proxy in front sets, else one fixed user.

    Only safe while the api is reachable through that proxy alone.
    """

    def __init__(self, header: str, default: str):
        self.header = header
        self.default = default

    def __call__(self, request: Request) -> str:
        return request.headers.get(self.header) or self.default


def current_user(request: Request) -> str:
    return request.app.state.identity(request)
