output "bucket" {
  value = aws_s3_bucket.work.id
}

output "instance_id" {
  value = aws_instance.gpu.id
}

output "session_manager_url" {
  description = "Open a shell on the machine in the browser (no SSH)."
  value       = "https://${var.region}.console.aws.amazon.com/systems-manager/session-manager/${aws_instance.gpu.id}?region=${var.region}"
}

output "next_steps" {
  value = <<-EOT
    1. Upload the teacher's partial files, then the start signal:
         aws s3 cp results/predictions/teacher-train.jsonl      s3://${aws_s3_bucket.work.id}/predictions/
         aws s3 cp results/predictions/teacher-train.meta.json  s3://${aws_s3_bucket.work.id}/predictions/
         echo go | aws s3 cp - s3://${aws_s3_bucket.work.id}/READY
    2. Follow the job's log:
         aws s3 cp s3://${aws_s3_bucket.work.id}/logs/job.log - | tail -20
    3. When s3://${aws_s3_bucket.work.id}/DONE appears, download the results:
         aws s3 sync s3://${aws_s3_bucket.work.id}/predictions results/predictions --exclude "*" --include "lora-distilled*" --include "teacher-*" --include "distill_report.json"
  EOT
}
