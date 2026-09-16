"""Request the deployed full-frame workflow; never rebuild the legacy crop pipeline."""
import json
import os
import time
import uuid
import boto3
from botocore.config import Config

TERMINAL = {"Succeeded", "Failed", "Stopped"}
POLL_SECONDS = 30

# The environment variables the app sets on the build, and the request
# keys the retraining Lambda reads them as. Anything left unset is simply
# not sent, so the run uses the pipeline's own default for it.
HYPERPARAMETER_ENV = {
    "TRAIN_EPOCHS": "epochs",
    "TRAIN_IMG_SIZE": "img_size",
    "TRAIN_BATCH": "batch",
    "TRAIN_SEED": "seed",
    # Folders the operator picked. Unset, the coordinator derives the
    # dataset from its pinned snapshot, which is what the weekly schedule
    # does and what every run did before these were forwarded.
    "TRAIN_DATA_S3_URI": "dataset_uri",
    "EVAL_DATA_S3_URI": "eval_dataset_uri",
}

def main():
    region = "ap-southeast-1"
    identity = boto3.client("sts").get_caller_identity()["Arn"]
    client = boto3.client("lambda", region_name=region, config=Config(read_timeout=910))
    request = {
        "manual": True,
        "request_id": os.environ.get("CODEBUILD_BUILD_ID") or str(uuid.uuid4()),
        "initiator": identity,
        "reason": os.environ.get("TRAIN_NOTES") or "CodeBuild full-frame replay retraining request",
    }
    for env_name, key in HYPERPARAMETER_ENV.items():
        value = os.environ.get(env_name)
        if value:
            request[key] = value
    response = client.invoke(FunctionName="track2-yolo-weekly-retrain", Payload=json.dumps(request).encode())
    result = json.loads(response["Payload"].read())
    if response.get("FunctionError"):
        raise RuntimeError(result)
    print(json.dumps(result, indent=2))
    if result.get("status") != "started":
        return
    # boto3 has no "pipeline_execution_complete" waiter -- asking for one
    # raised ValueError here, which failed the build about a second after
    # it had successfully started the execution. Every run therefore
    # reported FAILED while its training went on to succeed. Polling
    # describe_pipeline_execution is what the waiter would have done.
    sm = boto3.client("sagemaker", region_name=region)
    arn = result["pipeline_execution_arn"]
    while True:
        status = sm.describe_pipeline_execution(PipelineExecutionArn=arn)["PipelineExecutionStatus"]
        if status in TERMINAL:
            break
        time.sleep(POLL_SECONDS)
    print("Pipeline execution {}: {}".format(arn.rsplit("/", 1)[-1], status))
    # The build's own status has to follow the training's: reporting
    # SUCCEEDED for a run that failed is how a bad model reaches the
    # registry unnoticed.
    if status != "Succeeded":
        raise SystemExit("Pipeline execution {}".format(status))

if __name__ == "__main__":
    main()
