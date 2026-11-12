import pytest

from backend.app.cognito_triggers import pre_signup_handler


def test_pre_signup_requires_email_for_public_registration():
    event = {
        "triggerSource": "PreSignUp_SignUp",
        "request": {"userAttributes": {"username": "new-user"}},
    }

    with pytest.raises(ValueError, match="valid email address is required"):
        pre_signup_handler(event, None)


def test_pre_signup_allows_email_without_auto_confirming_account():
    event = {
        "triggerSource": "PreSignUp_SignUp",
        "request": {
            "userAttributes": {
                "username": "new-user",
                "email": "new-user@example.com",
            }
        },
        "response": {},
    }

    result = pre_signup_handler(event, None)

    assert result["response"]["autoConfirmUser"] is False
    assert result["response"]["autoVerifyEmail"] is False


def test_pre_signup_does_not_change_admin_created_users():
    event = {
        "triggerSource": "PreSignUp_AdminCreateUser",
        "request": {"userAttributes": {"username": "admin-user"}},
    }

    assert pre_signup_handler(event, None) == event
