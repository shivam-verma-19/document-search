# -----------------------------------------------
# WAF Web ACL
# -----------------------------------------------
resource "aws_wafv2_web_acl" "api_waf" {
  name  = "${var.project_name}-waf"
  scope = "REGIONAL"  # REGIONAL for API Gateway (CLOUDFRONT for CloudFront)

  default_action {
    allow {}  # Allow by default; rules below block/count
  }

  # ---- Rule 1: AWS Managed Common Rule Set ----
  # Blocks SQLi, XSS, bad user agents, etc.
  rule {
    name     = "AWSManagedRulesCommonRuleSet"
    priority = 1

    override_action {
      none {}  # Use the rule group's own actions (Block)
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesCommonRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "CommonRuleSet"
      sampled_requests_enabled   = true
    }
  }

  # ---- Rule 2: Known Bad Inputs ----
  # Blocks Log4Shell, SSRF, etc.
  rule {
    name     = "AWSManagedRulesKnownBadInputsRuleSet"
    priority = 2

    override_action {
      none {}
    }

    statement {
      managed_rule_group_statement {
        name        = "AWSManagedRulesKnownBadInputsRuleSet"
        vendor_name = "AWS"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "KnownBadInputs"
      sampled_requests_enabled   = true
    }
  }

  # ---- Rule 3: Rate Limiting ----
  # Blocks IPs that exceed 1000 requests per 5 minutes
  rule {
    name     = "RateLimitPerIP"
    priority = 3

    action {
      block {}
    }

    statement {
      rate_based_statement {
        limit              = 1000   # adjust as needed
        aggregate_key_type = "IP"
      }
    }

    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimitPerIP"
      sampled_requests_enabled   = true
    }
  }

  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${var.project_name}-waf"
    sampled_requests_enabled   = true
  }
}

# -----------------------------------------------
# Associate WAF with API Gateway stage
# -----------------------------------------------
resource "aws_wafv2_web_acl_association" "api_waf_assoc" {
  resource_arn = "${aws_apigatewayv2_api.http_api.execution_arn}/stages/$default"
  web_acl_arn  = aws_wafv2_web_acl.api_waf.arn
}

# -----------------------------------------------
# WAF Logging (optional but recommended)
# -----------------------------------------------
resource "aws_cloudwatch_log_group" "waf_logs" {
  # Must be prefixed with aws-waf-logs-
  name              = "aws-waf-logs-${var.project_name}"
  retention_in_days = 7
}

resource "aws_wafv2_web_acl_logging_configuration" "waf_logging" {
  log_destination_configs = [aws_cloudwatch_log_group.waf_logs.arn]
  resource_arn            = aws_wafv2_web_acl.api_waf.arn
}