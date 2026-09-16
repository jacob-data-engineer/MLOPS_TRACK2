# Terraformer Guide: Reverse-Engineering SageMaker / ECR / IAM into HCL

> **Note:** this guide uses InstructIR's resources as a worked example
> throughout. InstructIR (and SwinIR) have since been fully removed from
> this project — the specific resource names below (`jurong-port-instructir-*`,
> `instructir-sagemaker-jl`) no longer exist in AWS. The Terraformer workflow
> itself is unchanged; substitute a real model's resource names when following
> along for a future model.

[Terraformer](https://github.com/GoogleCloudPlatform/terraformer) generates
Terraform HCL + a matching `.tfstate` from resources that already exist in
an AWS account — useful here if the InstructIR endpoint (or its ECR repo /
IAM role) was created manually, by a notebook, or by another team, and now
needs to be brought under this repo's `terraform/` management instead of
being redefined by hand and re-imported one resource at a time.

This is a **one-time reconciliation tool**, not a replacement for
`terraform/*.tf` above — the workflow below always ends with cleaned-up HCL
merged into the hand-written codebase, never with Terraformer's raw output
left as the source of truth.

---

## 1. Install Terraformer

Terraformer ships as a single binary; it needs a matching Terraform AWS
provider binary alongside it.

**macOS (Homebrew):**
```bash
brew install terraformer
```

**Linux / manual install:**
```bash
export PROVIDER=aws
curl -LO "https://github.com/GoogleCloudPlatform/terraformer/releases/latest/download/terraformer-all-linux-amd64"
chmod +x terraformer-all-linux-amd64
sudo mv terraformer-all-linux-amd64 /usr/local/bin/terraformer

# Terraformer shells out to the real AWS provider plugin to talk to the API.
terraform init   # run inside a scratch dir with a minimal aws provider block,
                  # or point terraformer at your existing terraform/.terraform/providers
```

Verify:
```bash
terraformer version
terraform version   # provider plugin must be Terraform >= 1.6, provider ~> 5.x, matching terraform/versions.tf
```

## 2. Configure AWS credentials

Terraformer uses the standard AWS SDK credential chain — same as the AWS CLI
and Terraform's `aws` provider:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=...       # if using temporary/STS credentials
export AWS_REGION=ap-southeast-1
```

Or, if you already use a named profile (recommended):
```bash
export AWS_PROFILE=my-sso-profile
aws sts get-caller-identity   # sanity check before importing anything
```

Read-only IAM permissions are sufficient for import (`sagemaker:Describe*`,
`sagemaker:List*`, `ecr:Describe*`, `iam:Get*`, `iam:List*`).

## 3. Run the import

Import all three resource types this project cares about in one pass, into
a scratch directory kept separate from `terraform/` until reviewed:

```bash
mkdir -p terraformer-import && cd terraformer-import

terraformer import aws \
  --resources=sagemaker,ecr,iam \
  --regions=ap-southeast-1 \
  --profile=my-sso-profile
```

To avoid pulling in every IAM role/ECR repo in the account (`iam` and `ecr`
resources are not namespaced by project), scope the import with a filter on
resource name/ARN, e.g.:

```bash
terraformer import aws \
  --resources=sagemaker,ecr,iam \
  --regions=ap-southeast-1 \
  --filter="Name=aws_sagemaker_endpoint;Resource=jurong-port-instructir-endpoint-jl" \
  --filter="Name=aws_ecr_repository;Resource=instructir-sagemaker-jl"
```

This produces:
```
terraformer-import/
  generated/
    aws/
      sagemaker/
        aws_sagemaker_model.tf
        aws_sagemaker_endpoint_configuration.tf
        aws_sagemaker_endpoint.tf
        terraform.tfstate
      ecr/
        aws_ecr_repository.tf
        terraform.tfstate
      iam/
        aws_iam_role.tf
        aws_iam_role_policy.tf / aws_iam_role_policy_attachment.tf
        terraform.tfstate
```

## 4. Clean up and structure the generated output

Terraformer's output is intentionally verbose and uses generated resource
names (e.g. `aws_sagemaker_model.tfer--jurong_port_instructir`) — it is a
starting point, not final HCL. Before merging into `terraform/`:

1. **Rename resource labels** to match this repo's naming convention
   (`aws_sagemaker_model.instructir`, not `tfer--...`) so future `terraform
   plan` diffs stay readable.
2. **Replace hardcoded values with variable references.** Terraformer emits
   literal strings (account IDs, ARNs, image URIs) — swap in
   `var.instructir_image_tag`, `aws_ecr_repository.instructir.repository_url`,
   `data.aws_caller_identity.current.account_id`, etc., matching the pattern
   already used in `terraform/sagemaker.tf` and `terraform/iam.tf`.
3. **Drop attributes that are read-only/computed** (`arn`, `creation_time`,
   `*_id` outputs) — Terraformer sometimes emits them as if they were inputs.
4. **Reconcile IAM policy documents** against `terraform/iam.tf`'s
   already-scoped `data.aws_iam_policy_document` blocks instead of keeping
   Terraformer's broader, auto-generated policy JSON — re-tighten to
   least-privilege after import, don't just adopt whatever the live role had.
5. **Delete duplicate/unrelated resources** the broad `iam`/`ecr` import
   pulled in that aren't part of this project.

## 5. Merge into the existing state instead of running two states

Do **not** just `terraform init` inside `terraformer-import/` and apply from
there long-term — that creates a second, disconnected state file for
resources that conceptually belong in `terraform/`. Reconcile into the one
state:

```bash
cd terraform
terraform init

# Register each cleaned-up resource against the address it now has in
# terraform/sagemaker.tf, terraform/ecr.tf, terraform/iam.tf:
terraform import aws_sagemaker_model.instructir jurong-port-instructir-jl
terraform import aws_sagemaker_endpoint_configuration.instructir jurong-port-instructir-config-jl
terraform import aws_sagemaker_endpoint.instructir jurong-port-instructir-endpoint-jl
terraform import aws_ecr_repository.instructir instructir-sagemaker-jl
terraform import aws_iam_role.sagemaker_execution jurong-port-sagemaker-execution-role-jl
```

After every `terraform import`, run `terraform plan` — it must show **no
changes** (or only cosmetic ones you intend, like a tag) before you consider
the resource reconciled. A non-empty diff means the HCL you copied from
Terraformer's output doesn't yet match the real resource's configuration —
fix the HCL, not the live resource.

Finally, delete the scratch `terraformer-import/` directory (and its
throwaway state) once every resource has been re-imported into
`terraform/`'s state — it should not be committed to version control.

## 6. When to reach for Terraformer again

- Someone created or modified a SageMaker endpoint/config directly via the
  console or CLI and it needs to be brought back under this repo's IaC.
- Onboarding a pre-existing AWS account/region that has InstructIR-style
  resources this project didn't originally provision.
- Auditing drift: re-run the import into a scratch dir and diff its output
  against `terraform/*.tf` to spot manual changes, then reconcile via
  `terraform plan`/`apply` (preferred) rather than re-importing wholesale.
