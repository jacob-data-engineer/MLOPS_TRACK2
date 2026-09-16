"""Deploys/updates one YOLO11 size (s/m/l/x) as a SageMaker Inference
Component on the one shared, always-on real-time endpoint -- lets the
review-app UI switch which size handles a request instantly (just a
different InferenceComponentName on invoke_endpoint), since all four
stay warm simultaneously. See config.py's multi_size_* fields and
infra/sagemaker.tf's per-size Model Package Groups.

Deliberately boto3, not Terraform: Terraform's AWS provider has no
aws_sagemaker_inference_component resource at all, and even the shared
endpoint/config can't be expressed in Terraform either -- its
aws_sagemaker_endpoint_configuration resource requires model_name on
every production variant, even though AWS's own CreateEndpointConfig API
makes it optional (that's exactly what makes a variant "inference
component enabled"). Confirmed by checking the provider source directly:
nothing matches "inference_component", and a plain HCL attempt failed
validation on the missing model_name. So the endpoint/config bootstrap
lives here too, not split half-and-half.

Two ways to source a size's weights:
  --model-package-arn <arn>   Model Registry package (Approved only,
                               checked below) -- create_model can
                               reference it directly via
                               PrimaryContainer.ModelPackageName, no
                               manual image/artifact lookup needed.
  --weights-s3-uri <s3://...> A raw S3 folder (the "browse S3, pick a
                               folder" UI path) -- if it holds a bare
                               model.pt instead of an already-packaged
                               model.tar.gz, this repackages it first
                               (same rename-into-tarball step as this
                               project's very first SageMaker deploy,
                               documented in summary.md's "Starting
                               point"), since the serving container's
                               inference.py hardcodes that filename.
                               Bypasses the Model Registry entirely, so
                               there's no ModelApprovalStatus gate and
                               no attached metrics for this route --
                               that's a deliberate tradeoff for
                               flexibility, not an oversight.

Usage:
    uv run python deploy_inference_component.py s --model-package-arn <arn>
    uv run python deploy_inference_component.py m --weights-s3-uri s3://track2-s3/weights/m/2026-09-07/
    uv run python deploy_inference_component.py --bootstrap-endpoint-only
"""

import argparse
import os
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from urllib.parse import urlparse

import boto3
from botocore.exceptions import ClientError

from config import CONFIG, SIZES

sm = boto3.client("sagemaker", region_name=CONFIG.region)
s3 = boto3.client("s3", region_name=CONFIG.region)

# Conservative starting point for four concurrent CPU-only components on
# one ml.m5.2xlarge (8 vCPU / 32 GiB) -- sums to 6 cores / ~15GB, leaving
# real headroom, not load-tested against actual traffic yet.
COMPUTE_REQUIREMENTS = {
    "s": {"NumberOfCpuCoresRequired": 1, "MinMemoryRequiredInMb": 2048},
    "m": {"NumberOfCpuCoresRequired": 1, "MinMemoryRequiredInMb": 3072},
    "l": {"NumberOfCpuCoresRequired": 2, "MinMemoryRequiredInMb": 4096},
    "x": {"NumberOfCpuCoresRequired": 2, "MinMemoryRequiredInMb": 6144},
}


def _timestamp_suffix() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected an s3:// URI, got: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def ensure_multi_size_endpoint() -> None:
    """Idempotent: creates the shared IC-enabled endpoint (+its config) if
    it doesn't already exist. No model on the variant -- that's what makes
    it host Inference Components instead of one fixed model."""
    existing = sm.list_endpoints(NameContains=CONFIG.multi_size_endpoint_name)["Endpoints"]
    if any(e["EndpointName"] == CONFIG.multi_size_endpoint_name for e in existing):
        print(f"Endpoint {CONFIG.multi_size_endpoint_name!r} already exists, leaving it as-is.")
        return

    sm.create_endpoint_config(
        EndpointConfigName=CONFIG.multi_size_endpoint_config_name,
        # Required here specifically for IC-enabled configs (a variant
        # with no ModelName) -- confirmed live: CreateEndpointConfig
        # rejected the first attempt without it ("...include a required
        # specification for endpoints with Inference Components enabled
        # ... retry with the ExecutionRoleArn specification"), even
        # though neither the API reference nor the Terraform provider
        # schema surfaced this requirement ahead of time.
        ExecutionRoleArn=CONFIG.sagemaker_execution_role_arn,
        ProductionVariants=[
            {
                "VariantName": CONFIG.multi_size_variant_name,
                "InstanceType": CONFIG.multi_size_instance_type,
                "InitialInstanceCount": 1,
                # No ModelName -- see module docstring.
            }
        ],
    )
    sm.create_endpoint(
        EndpointName=CONFIG.multi_size_endpoint_name,
        EndpointConfigName=CONFIG.multi_size_endpoint_config_name,
    )
    print(
        f"Creating endpoint {CONFIG.multi_size_endpoint_name!r} "
        f"(config: {CONFIG.multi_size_endpoint_config_name!r}) -- this starts billing "
        f"a {CONFIG.multi_size_instance_type} instance immediately, continuously "
        "(real-time endpoints have no scale-to-zero)."
    )


def _repackage_raw_checkpoint_if_needed(weights_s3_uri: str) -> str:
    """weights_s3_uri points at a folder containing either an
    already-packaged model.tar.gz (used as-is) or a bare model.pt (tarred
    up under the exact filename inference.py's model_fn expects, then
    uploaded next to the original weights). Returns the S3 URI of a
    model.tar.gz ready to hand to create_model."""
    bucket, prefix = _parse_s3_uri(weights_s3_uri.rstrip("/") + "/")

    listing = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/")
    keys = {obj["Key"].rsplit("/", 1)[-1] for obj in listing.get("Contents", [])}

    if "model.tar.gz" in keys:
        return f"s3://{bucket}/{prefix}model.tar.gz"

    if "model.pt" not in keys:
        raise FileNotFoundError(
            f"s3://{bucket}/{prefix} contains neither model.tar.gz nor model.pt "
            f"(found: {sorted(keys) or '<empty>'}) -- nothing to deploy."
        )

    print(f"s3://{bucket}/{prefix} has a raw model.pt, not model.tar.gz -- repackaging.")
    with tempfile.TemporaryDirectory() as tmp:
        checkpoint_path = os.path.join(tmp, "model.pt")
        s3.download_file(bucket, f"{prefix}model.pt", checkpoint_path)

        # Bundled alongside model.pt, not just baked into the serving
        # image -- confirmed live that Dockerfile.serve's COPY + env var
        # alone isn't reliably honored by the inference toolkit, which
        # falls back to its default (TorchScript-only) handler instead.
        # See train.py's own copy of this comment for the same fix on
        # the registry-registration path.
        code_dir = os.path.join(tmp, "code")
        os.makedirs(code_dir, exist_ok=True)
        inference_py_src = os.path.join(os.path.dirname(__file__), "..", "training-container", "inference.py")
        shutil.copy(inference_py_src, os.path.join(code_dir, "inference.py"))

        tarball_path = os.path.join(tmp, "model.tar.gz")
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(checkpoint_path, arcname="model.pt")
            tar.add(os.path.join(code_dir, "inference.py"), arcname="code/inference.py")

        dest_key = f"{prefix}model.tar.gz"
        s3.upload_file(tarball_path, bucket, dest_key)

    print(f"Repackaged -> s3://{bucket}/{dest_key}")
    return f"s3://{bucket}/{dest_key}"


def _inference_component_exists(name: str) -> bool:
    try:
        sm.describe_inference_component(InferenceComponentName=name)
        return True
    except sm.exceptions.ResourceNotFound:
        return False
    except ClientError as exc:
        # Confirmed live: a not-found InferenceComponentName raises
        # ValidationException ("Could not find inference component ..."),
        # not ResourceNotFound -- inconsistent with most other Describe*
        # calls in this same API, but that's what the service actually
        # returns.
        if exc.response.get("Error", {}).get("Code") == "ValidationException" and "Could not find" in str(exc):
            return False
        raise


def deploy(size: str, *, model_package_arn: str | None = None, weights_s3_uri: str | None = None) -> None:
    if size not in SIZES:
        raise ValueError(f"size must be one of {SIZES}, got {size!r}")
    if bool(model_package_arn) == bool(weights_s3_uri):
        raise ValueError("Pass exactly one of model_package_arn or weights_s3_uri, not both/neither.")

    ensure_multi_size_endpoint()

    suffix = _timestamp_suffix()
    model_name = f"{CONFIG.project}-{size}-model-{suffix}"

    if model_package_arn:
        desc = sm.describe_model_package(ModelPackageName=model_package_arn)
        if desc["ModelApprovalStatus"] != "Approved":
            raise RuntimeError(
                f"Refusing to deploy {model_package_arn}: ModelApprovalStatus is "
                f"{desc['ModelApprovalStatus']!r}, not Approved."
            )
        # PrimaryContainer={"ModelPackageName": ...} (the simpler path
        # deploy_endpoint.py uses for the single-model endpoint) is
        # rejected here -- confirmed live: "Model package cannot mix with
        # SageMaker or bring-your-own images, and only one model package
        # should be used for endpoints with Inference Components
        # enabled." An IC-enabled endpoint allows at most one
        # ModelPackageName-backed component ever, which breaks having
        # four sizes. Resolving the package's own Image/ModelDataUrl and
        # passing those directly sidesteps the restriction entirely (and
        # matches the raw-S3-folder path below, which never hit this).
        container = desc["InferenceSpecification"]["Containers"][0]
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={"Image": container["Image"], "ModelDataUrl": container["ModelDataUrl"]},
            ExecutionRoleArn=CONFIG.sagemaker_execution_role_arn,
        )
        source_desc = model_package_arn
    else:
        artifact_s3_uri = _repackage_raw_checkpoint_if_needed(weights_s3_uri)
        sm.create_model(
            ModelName=model_name,
            PrimaryContainer={"Image": CONFIG.serving_image_uri, "ModelDataUrl": artifact_s3_uri},
            ExecutionRoleArn=CONFIG.sagemaker_execution_role_arn,
        )
        source_desc = artifact_s3_uri

    inference_component_name = f"{CONFIG.project}-{size}"
    specification = {
        "ModelName": model_name,
        "ComputeResourceRequirements": COMPUTE_REQUIREMENTS[size],
        "StartupParameters": {
            "ModelDataDownloadTimeoutInSeconds": 600,
            "ContainerStartupHealthCheckTimeoutInSeconds": 600,
        },
    }

    if _inference_component_exists(inference_component_name):
        sm.update_inference_component(
            InferenceComponentName=inference_component_name,
            Specification=specification,
        )
        print(f"Updating {inference_component_name!r} -> {source_desc} (model: {model_name})")
    else:
        sm.create_inference_component(
            EndpointName=CONFIG.multi_size_endpoint_name,
            VariantName=CONFIG.multi_size_variant_name,
            InferenceComponentName=inference_component_name,
            Specification=specification,
            RuntimeConfig={"CopyCount": 1},
        )
        print(f"Creating {inference_component_name!r} -> {source_desc} (model: {model_name})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("size", nargs="?", choices=SIZES, help="Which YOLO11 size to deploy (s/m/l/x)")
    parser.add_argument("--model-package-arn", help="Deploy this exact Approved Model Registry package")
    parser.add_argument("--weights-s3-uri", help="Deploy from this raw S3 folder (model.pt or model.tar.gz)")
    parser.add_argument(
        "--bootstrap-endpoint-only",
        action="store_true",
        help="Only create the shared endpoint/config if missing; deploy no size.",
    )
    args = parser.parse_args()

    if args.bootstrap_endpoint_only:
        ensure_multi_size_endpoint()
    elif args.size:
        deploy(args.size, model_package_arn=args.model_package_arn, weights_s3_uri=args.weights_s3_uri)
    else:
        parser.error("Pass a size (s/m/l/x) or --bootstrap-endpoint-only")
