import re
from typing import Any

_EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def pre_signup_handler(event: dict[str, Any], _context: Any) -> dict[str, Any]:
    if event.get("triggerSource") == "PreSignUp_SignUp":
        attributes = event.get("request", {}).get("userAttributes", {})
        email = attributes.get("email", "")
        if not isinstance(email, str) or not _EMAIL_PATTERN.fullmatch(email):
            raise ValueError("A valid email address is required to create an account.")

        response = event.setdefault("response", {})
        response["autoConfirmUser"] = False
        response["autoVerifyEmail"] = False

    return event
