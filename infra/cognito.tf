data "aws_caller_identity" "current" {}

locals {
  cognito_domain_prefix = format(
    "rag-%s-%s-%s",
    substr(replace(lower(var.project_name), "_", "-"), 0, 35),
    data.aws_caller_identity.current.account_id,
    replace(var.aws_region, "-", "")
  )
  frontend_origin = trimsuffix(var.frontend_origin, "/")
}

resource "aws_cognito_user_pool" "user_pool" {
  name                     = "${var.project_name}-users"
  auto_verified_attributes = ["email"]

  admin_create_user_config {
    allow_admin_create_user_only = false
  }

  password_policy {
    minimum_length                   = 12
    require_lowercase                = true
    require_numbers                  = true
    require_symbols                  = true
    require_uppercase                = true
    temporary_password_validity_days = 7
  }

  lambda_config {
    pre_sign_up = aws_lambda_function.cognito_pre_signup.arn
  }
}

resource "aws_cognito_user_pool_domain" "managed_login" {
  domain       = local.cognito_domain_prefix
  user_pool_id = aws_cognito_user_pool.user_pool.id
}

resource "aws_cognito_user_pool_client" "client" {
  name         = "${var.project_name}-client"
  user_pool_id = aws_cognito_user_pool.user_pool.id

  explicit_auth_flows                   = ["ALLOW_REFRESH_TOKEN_AUTH"]
  generate_secret                       = false
  allowed_oauth_flows_user_pool_client = true
  allowed_oauth_flows                  = ["code"]
  allowed_oauth_scopes                 = ["openid", "email"]
  callback_urls                        = ["${local.frontend_origin}/api/auth/callback"]
  logout_urls                          = [local.frontend_origin]
  supported_identity_providers         = ["COGNITO"]
  read_attributes                      = ["email", "email_verified"]
  write_attributes                     = ["email"]
  enable_token_revocation              = true
  prevent_user_existence_errors        = "ENABLED"

  access_token_validity  = 60
  id_token_validity      = 60
  refresh_token_validity = 30

  token_validity_units {
    access_token  = "minutes"
    id_token      = "minutes"
    refresh_token = "days"
  }
}

resource "aws_lambda_permission" "cognito_pre_signup" {
  statement_id   = "AllowCognitoPreSignup"
  action         = "lambda:InvokeFunction"
  function_name  = aws_lambda_function.cognito_pre_signup.function_name
  principal      = "cognito-idp.amazonaws.com"
  source_arn     = aws_cognito_user_pool.user_pool.arn
  source_account = data.aws_caller_identity.current.account_id
}
