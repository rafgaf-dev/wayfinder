# Running the model steps on AWS

One `g4dn.xlarge` machine (a T4, the same GPU as the Colab runs) finishes the teacher's labels, builds the
distilled splits, trains the student and predicts the test set. Progress lives in an S3 bucket, so a machine
that dies can be replaced and the job carries on. Expect about 5–6 hours, roughly $3 at on-demand prices.

The infrastructure is Terraform, in `infra/aws/`:
- **An S3 bucket**, private and encrypted, for inputs, progress, checkpoints, logs and results. `terraform
  destroy` never deletes it while it holds anything.
- **The GPU machine**, on AWS's Deep Learning Base image (Ubuntu 24.04, NVIDIA driver, CUDA, Python 3.12).
  It has **no SSH and no open ports**: you open a shell from the console with Session Manager.
- **An IAM role** that can reach only that bucket and one SSM parameter, the GitHub token.
- **A monthly cost alert** emailed to you. It counts usage before credits, so it fires while the credits are
  being spent.
- **Two automatic shutdowns:** the machine powers off when the job ends, and in any case after `max_hours`
  (default 10).

## 1. One-time setup on the Mac

```bash
brew install awscli hashicorp/tap/terraform
aws configure            # or: aws configure sso — use your own credentials; never share them
aws sts get-caller-identity
```

**GPU quota.** In the Service Quotas console, under Amazon EC2, **"Running On-Demand G and VT instances"** must
be at least **4** in your region (`g4dn.xlarge` has 4 vCPUs). New accounts often start at 0, and an increase
can take a while to be approved.

**The GitHub token**, a read-only fine-grained token with access to `wayfinder-data` only (Contents: read).
Store it as a SecureString, either in the console (Systems Manager → Parameter Store → Create parameter:
name `/wayfinder/github-token`, type SecureString, default key) or from the terminal without it landing in
your shell history:

```bash
read -rs "TOKEN?GitHub token: " && aws ssm put-parameter --name /wayfinder/github-token --type SecureString --value "$TOKEN" && unset TOKEN
```

Terraform only references the parameter by name and never reads its value, so the token never enters the
Terraform state file.

## 2. Hand over the teacher's progress

Stop the teacher on the PC with Ctrl+C: every finished item is already saved. Then get
`results\predictions\teacher-train.jsonl` and `teacher-train.meta.json` onto the Mac, into the same folder,
or upload them from the PC through the S3 console in step 4.

## 3. Create the machine

```bash
cd infra/aws
cp terraform.tfvars.example terraform.tfvars   # set region and budget_email
terraform init
terraform plan        # read what it will create
terraform apply
```

The outputs print the bucket name, the Session Manager link and the exact commands for the next steps. The
machine boots, installs its environment (about 10 minutes), then **waits** for the start signal. It won't
start labelling from zero before the teacher's progress is uploaded.

If `apply` fails with an "Unsupported" or "InsufficientInstanceCapacity" error, that availability zone has
no g4dn capacity. Set `subnet_id` in `terraform.tfvars` to a default subnet in another zone and apply again.

## 4. Start the job

From the repo root (the exact commands are in `terraform output next_steps`):

```bash
aws s3 cp results/predictions/teacher-train.jsonl     s3://BUCKET/predictions/
aws s3 cp results/predictions/teacher-train.meta.json s3://BUCKET/predictions/
echo go | aws s3 cp - s3://BUCKET/READY
```

## 5. Watch it

```bash
aws s3 cp s3://BUCKET/logs/job.log - | tail -20
```

The log is copied every 5 minutes. For a live view, open the `session_manager_url` output in the browser
and run `tail -f ~ubuntu/job.log`.

When it finishes, `s3://BUCKET/DONE` appears and the machine shuts down. If a step fails, `FAILED` appears
instead, and the log says why.

If the machine is lost part-way (a crash, or the time cap), run `terraform apply -replace=aws_instance.gpu`
to get a new one. It resumes from the bucket. Upload `READY` again if you deleted it.

## 6. Collect the results and clean up

```bash
aws s3 sync s3://BUCKET/predictions results/predictions --exclude "*" --include "lora-distilled*" --include "teacher-*" --include "distill_report.json"
cd infra/aws && terraform destroy
```

`destroy` removes the machine, its disk, the role and the budget. It stops with an error at the bucket
because the bucket isn't empty: that's deliberate, so the results survive. Once you have everything, empty
and delete the bucket in the S3 console.
