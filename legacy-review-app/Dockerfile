# Tiny — stdlib http.server + boto3, no framework needed for this size.
# Built directly on the target EC2 instance (see
# ../scripts/deploy_review_app.sh), same as the Grafana demo's pattern —
# no ECR push needed for something this small.

FROM python:3.12-slim

RUN pip install --no-cache-dir boto3

COPY review_app.py /app/review_app.py
WORKDIR /app

EXPOSE 8090

CMD ["python", "review_app.py"]
