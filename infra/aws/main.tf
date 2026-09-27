# One GPU machine that runs the remaining model steps and keeps its progress in
# an S3 bucket. No SSH and no open ports: you connect with Session Manager from
# the AWS console, and the machine's role can reach only this bucket and the
# one GitHub-token parameter.

data "aws_caller_identity" "current" {}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

data "aws_ssm_parameter" "ami" {
  name = var.ami_ssm_parameter
}

locals {
  account_id = data.aws_caller_identity.current.account_id
  bucket     = "wayfinder-${local.account_id}-${var.region}"
  # Built from the name rather than read with a data source: reading the
  # parameter would copy the token into the Terraform state in plain text.
  token_parameter_arn = "arn:aws:ssm:${var.region}:${local.account_id}:parameter${var.github_token_parameter}"
}

# --- Storage ---------------------------------------------------------------

resource "aws_s3_bucket" "work" {
  bucket = local.bucket
  # A bucket with results in it is never deleted by `terraform destroy`: the
  # destroy stops with an error instead. Empty it yourself once the results
  # are downloaded.
  force_destroy = false
}

resource "aws_s3_bucket_public_access_block" "work" {
  bucket                  = aws_s3_bucket.work.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "work" {
  bucket = aws_s3_bucket.work.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# --- Identity --------------------------------------------------------------

data "aws_iam_policy_document" "assume_ec2" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ec2.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "gpu" {
  name               = "wayfinder-gpu"
  assume_role_policy = data.aws_iam_policy_document.assume_ec2.json
}

data "aws_iam_policy_document" "gpu" {
  statement {
    sid       = "ListBucket"
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.work.arn]
  }
  statement {
    sid       = "ReadWriteObjects"
    actions   = ["s3:GetObject", "s3:PutObject", "s3:DeleteObject"]
    resources = ["${aws_s3_bucket.work.arn}/*"]
  }
  statement {
    sid       = "ReadGitHubToken"
    actions   = ["ssm:GetParameter"]
    resources = [local.token_parameter_arn]
  }
  statement {
    # SecureStrings are encrypted with the account's aws/ssm key; allow
    # decrypting only through SSM, not with the key directly.
    sid       = "DecryptTokenViaSsm"
    actions   = ["kms:Decrypt"]
    resources = ["*"]
    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["ssm.${var.region}.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "gpu" {
  name   = "wayfinder-gpu"
  role   = aws_iam_role.gpu.id
  policy = data.aws_iam_policy_document.gpu.json
}

# Lets you open a shell from the console (Session Manager) with no SSH key or
# open port.
resource "aws_iam_role_policy_attachment" "session_manager" {
  role       = aws_iam_role.gpu.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_instance_profile" "gpu" {
  name = "wayfinder-gpu"
  role = aws_iam_role.gpu.name
}

# --- Network ---------------------------------------------------------------

resource "aws_security_group" "gpu" {
  name        = "wayfinder-gpu"
  description = "No inbound access. Outbound only, for git, pip, Hugging Face and S3."
  vpc_id      = data.aws_vpc.default.id

  egress {
    from_port        = 0
    to_port          = 0
    protocol         = "-1"
    cidr_blocks      = ["0.0.0.0/0"]
    ipv6_cidr_blocks = ["::/0"]
  }
}

# --- Machine ---------------------------------------------------------------

resource "aws_instance" "gpu" {
  ami                         = data.aws_ssm_parameter.ami.insecure_value
  instance_type               = var.instance_type
  subnet_id                   = coalesce(var.subnet_id, data.aws_subnets.default.ids[0])
  vpc_security_group_ids      = [aws_security_group.gpu.id]
  iam_instance_profile        = aws_iam_instance_profile.gpu.name
  associate_public_ip_address = true

  # When the job finishes (or the time cap hits) the machine powers off and
  # stops billing for compute. Stopped, only its disk costs anything, until
  # `terraform destroy` removes it.
  instance_initiated_shutdown_behavior = "stop"

  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required" # IMDSv2 only
  }

  root_block_device {
    volume_size           = var.root_volume_gb
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  user_data = templatefile("${path.module}/user_data.sh.tftpl", {
    bucket          = aws_s3_bucket.work.id
    region          = var.region
    token_parameter = var.github_token_parameter
    public_repo_url = var.public_repo_url
    branch          = var.branch
    data_repo       = var.data_repo
    max_minutes     = var.max_hours * 60
  })
  user_data_replace_on_change = true

  tags = {
    Name = "wayfinder-gpu"
  }

  depends_on = [aws_iam_role_policy.gpu, aws_iam_role_policy_attachment.session_manager]
}

# --- Cost alert --------------------------------------------------------------

resource "aws_budgets_budget" "monthly" {
  count = var.budget_email == "" ? 0 : 1

  name         = "wayfinder-monthly"
  budget_type  = "COST"
  limit_amount = var.budget_limit_usd
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  # Count usage before credits. With credits included, the net cost stays at
  # $0 and the alert would never fire while the credits are being spent.
  cost_types {
    include_credit = false
    include_refund = false
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 80
    threshold_type             = "PERCENTAGE"
    notification_type          = "ACTUAL"
    subscriber_email_addresses = [var.budget_email]
  }

  notification {
    comparison_operator        = "GREATER_THAN"
    threshold                  = 100
    threshold_type             = "PERCENTAGE"
    notification_type          = "FORECASTED"
    subscriber_email_addresses = [var.budget_email]
  }
}
