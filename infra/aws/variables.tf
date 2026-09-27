variable "region" {
  description = "AWS region. Your G-instance quota is per region, so use the one where it was granted."
  type        = string
  default     = "us-east-1"
}

variable "instance_type" {
  description = "g4dn.xlarge has a T4, the same GPU as the Colab runs, so timings stay comparable."
  type        = string
  default     = "g4dn.xlarge"
}

variable "subnet_id" {
  description = "A subnet in the default VPC. Leave null to use the first default subnet; set one if that availability zone has no g4dn capacity."
  type        = string
  default     = null
}

variable "root_volume_gb" {
  description = "The Deep Learning image needs a large root volume; the model, packages and checkpoints add about 25 GB."
  type        = number
  default     = 120
}

variable "max_hours" {
  description = "Hard cap: the machine powers itself off after this many hours, whatever the job is doing."
  type        = number
  default     = 10
}

variable "github_token_parameter" {
  description = "Name of the SSM SecureString holding a read-only GitHub token for the private data repo. Create it yourself; Terraform never reads its value, so it never enters the state file."
  type        = string
  default     = "/wayfinder/github-token"
}

variable "public_repo_url" {
  description = "The public code repo, cloned at boot. The job runs infra/aws/job.sh from this branch."
  type        = string
  default     = "https://github.com/rafgaf-dev/wayfinder.git"
}

variable "branch" {
  type    = string
  default = "main"
}

variable "data_repo" {
  description = "owner/name of the private data repo."
  type        = string
  default     = "rafgaf-dev/wayfinder-data"
}

variable "budget_email" {
  description = "Email address for the cost alert. Leave empty to skip creating the budget."
  type        = string
  default     = ""
}

variable "budget_limit_usd" {
  description = "Monthly budget; alerts at 80% of actual spend and 100% of forecast spend."
  type        = string
  default     = "20"
}

variable "ami_ssm_parameter" {
  description = "Public SSM parameter that always points at the latest Deep Learning Base GPU image (Ubuntu 24.04: NVIDIA driver, CUDA, Python 3.12, SSM agent)."
  type        = string
  default     = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-24.04/latest/ami-id"
}
