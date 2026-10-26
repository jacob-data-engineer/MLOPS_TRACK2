# Extends AWS's own SageMaker PyTorch training image (resolved correctly
# per-region by build_and_push.sh via the SageMaker SDK's image_uris,
# rather than a hand-guessed DLC registry account ID) with Ultralytics
# YOLO. Script-mode entrypoint (train.py) follows the SageMaker training
# toolkit contract already baked into the base image.

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

# 8.4, not 8.3, because 8.3 has NO yolo26 config at all -- the cfg/models
# tree simply does not contain it, so a YOLO26 run dies at model
# construction rather than at anything that reads like a version problem.
# Bumped 2026-09-30 with YOLO26. YOLO11 runs move to 8.4 with it, which is
# the real risk in this change and is why a YOLO11 run is re-verified
# after the rebuild rather than assumed.
RUN pip install --no-cache-dir ultralytics==8.4.* boto3 "sagemaker>=2.220,<3"

COPY train.py /opt/ml/code/train.py
# train.py copies this into the model artifact's own code/ dir at the end
# of every run -- see its own comment for why that (not Dockerfile.serve's
# COPY+ENV, confirmed unreliable) is what actually gets the serving
# container to use it instead of its default TorchScript-only handler.
COPY inference.py /opt/ml/code/inference.py

ENV SAGEMAKER_PROGRAM=train.py
