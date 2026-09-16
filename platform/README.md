# Jurong Port — SageMaker BYOC serving scaffold

A reusable, model-agnostic Bring-Your-Own-Container (BYOC) scaffold for
real-time SageMaker inference endpoints, built around a small shared FastAPI
serving framework. **No model is currently deployed** — InstructIR and
SwinIR, the two models previously served here, were both removed: their
underlying S3 model artifacts (in the shared `gp-jurong-port-s3-jl` bucket)
were deleted, permanently breaking both, so their code, Terraform resources,
and live AWS endpoints were torn down rather than left as dead references.

## Repository layout

```
common/                    shared serving framework (model-agnostic)
  server.py                FastAPI app: GET /ping, POST /invocations (port 8080)
  schemas.py                Pydantic request/response models
  utils.py                  device mgmt, pre/post-processing, OOM guard, base64 codec
  serve                     executable SageMaker calls via `docker run <image> serve`
models/                    empty -- add a new model here (see below)
terraform/                  IaC: IAM role scaffolding for a future model's SageMaker resources
requirements.txt
TERRAFORMER_GUIDE.md         Terraformer install/import/merge walkthrough
```

### How the shared server + model handler fit together

A model's Dockerfile copies `models/<name>/inference.py` to the fixed path
`/opt/program/model_inference.py`; `common/server.py` does
`import model_inference` (selecting `<name>` via the `MODEL_TYPE` environment
variable) and calls the module-level `handler` singleton it exposes. That
singleton must implement:

```python
class ModelHandler:
    name: str
    is_ready: bool
    def load(self) -> None: ...
    def predict(self, image: PIL.Image.Image, *, prompt: str | None) -> PIL.Image.Image: ...
```

`load()` runs once from a FastAPI `lifespan` handler at container startup.
`predict()` runs inside a threadpool worker per request (`run_in_threadpool`)
so the event loop keeps answering `/ping` while a GPU forward pass is in flight.

### Adding a model

1. Add `models/<name>/inference.py` exposing the `ModelHandler` contract above,
   and `models/<name>/Dockerfile.<name>` copying it to
   `/opt/program/model_inference.py` and setting `MODEL_TYPE=<name>` —
   `common/server.py` needs no changes.
2. Build and push the image to a new ECR repository for `<name>`.
3. Add matching `aws_ecr_repository` / `aws_sagemaker_model` /
   `aws_sagemaker_endpoint_configuration` / `aws_sagemaker_endpoint` (and,
   if idle scale-to-zero is wanted, `aws_appautoscaling_target`/`policy` +
   `aws_cloudwatch_metric_alarm`) resources in `terraform/` — there's no
   existing model's resources left to copy as a template, since both prior
   models were fully decommissioned; start from the AWS provider docs for
   these resource types.
4. Extend `terraform/iam.tf`'s `sagemaker_execution` role with an ECR-pull
   statement scoped to the new repository, and an async-inference S3
   read/write statement scoped to `<name>/async-{input,output,failure}/`
   if using Asynchronous Inference.

---

## 1. Local development (no Docker, no GPU required for the contract tests)

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

MODEL_TYPE=<name> uvicorn common.server:app --host 0.0.0.0 --port 8080 --reload

# In another shell:
curl -fsS http://localhost:8080/ping   # 503 until model.load() succeeds
```

There are no contract tests checked in right now (the previous ones tested
InstructIR's server contract specifically and were removed with it) — a new
model should add its own `tests/test_server.py` covering `/ping` and
`/invocations` content-type/payload-size handling.

---

## 2. Deploying via Terraform

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars: model_artifact_bucket_name, s3_bucket_name, etc.

terraform init
terraform plan
terraform apply
```

This currently provisions only the shared scaffolding:
- The `sagemaker_execution` IAM role, scoped to S3-read on the model
  artifact bucket + CloudWatch Logs (`iam.tf`)
- Nothing model-specific — add resources per "Adding a model" above once a
  new model exists.

---

## 3. Terraformer

See [TERRAFORMER_GUIDE.md](./TERRAFORMER_GUIDE.md) for installing
Terraformer and reverse-engineering HCL from any SageMaker/ECR/IAM resources
that were created outside this Terraform codebase (e.g. via the console or
`aws sagemaker create-endpoint`), and how to merge the generated code back in
without clobbering the hand-written configuration above.
