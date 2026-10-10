from typing import Protocol
from uuid import UUID, uuid4

from fastapi import Request, Response

VISITOR_COOKIE = "visitor"
VISITOR_MAX_AGE = 365 * 24 * 3600


class IdentityProvider(Protocol):
    def __call__(self, request: Request, response: Response) -> str:
        """User id for the request."""
        ...


class Identity:
    """User id from a trusted header if one is set, else from a random visitor cookie.

    The header is only safe while every request passes the proxy that sets it,
    so it's off unless configured.
    """

    def __init__(self, header: str | None, secure_cookie: bool):
        self.header = header
        self.secure_cookie = secure_cookie

    def __call__(self, request: Request, response: Response) -> str:
        if self.header and (user := request.headers.get(self.header)):
            return user
        visitor = request.cookies.get(VISITOR_COOKIE, "")
        if not _is_uuid(visitor):
            visitor = str(uuid4())
            response.set_cookie(
                VISITOR_COOKIE,
                visitor,
                max_age=VISITOR_MAX_AGE,
                path="/api",
                httponly=True,
                secure=self.secure_cookie,
                samesite="lax",
            )
        # Prefixed so a visitor id never equals a header user id.
        return f"visitor:{visitor}"


def _is_uuid(value: str) -> bool:
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False


def current_user(request: Request, response: Response) -> str:
    return request.app.state.identity(request, response)
