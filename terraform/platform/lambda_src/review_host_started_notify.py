"""Emails the review-app URL whenever the review host finishes starting.

The host is stopped nightly and all weekend (see ec2_schedule.tf), and it
has no Elastic IP -- a stopped instance releases its public IPv4 address
outright, so AWS hands it a brand new one on every single start. The URL
people bookmarked yesterday is dead by this morning, and nothing told
them the new one. That is what this fixes.

Publishes to the same reviewer-notifications topic every other
human-facing notification already uses (training status, drift review,
weights approved), so whoever is already subscribed gets this too with no
separate opt-in to manage.

Triggered by EC2's own "EC2 Instance State-change Notification" event for
state=running, not by the scheduler that does the starting -- so a manual
start, or a start from anywhere else, is announced just the same.
"""

import json
import os
import time

import boto3

REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
NOTIFY_TOPIC_ARN = os.environ["NOTIFY_TOPIC_ARN"]
PORTS = json.loads(os.environ.get("REVIEW_APP_PORTS_JSON", "[8090]"))

sns = boto3.client("sns", region_name=REGION)
ec2 = boto3.client("ec2", region_name=REGION)

# "running" fires the moment the instance is assigned to hardware, which
# can very slightly precede the public IP showing up on a describe. A few
# short retries covers that without making the whole notification depend
# on winning a race.
IP_LOOKUP_ATTEMPTS = 6
IP_LOOKUP_DELAY_SECONDS = 5


def _public_ip(instance_id: str) -> str | None:
    for attempt in range(IP_LOOKUP_ATTEMPTS):
        reservations = ec2.describe_instances(InstanceIds=[instance_id])["Reservations"]
        for reservation in reservations:
            for instance in reservation["Instances"]:
                ip = instance.get("PublicIpAddress")
                if ip:
                    return ip
        if attempt < IP_LOOKUP_ATTEMPTS - 1:
            time.sleep(IP_LOOKUP_DELAY_SECONDS)
    return None


def handler(event, context):
    instance_id = event["detail"]["instance-id"]
    ip = _public_ip(instance_id)

    if not ip:
        # Worth an email anyway: "it started but has no public address" is
        # a real problem someone needs to know about, and staying silent
        # here would look identical to the notification never firing.
        sns.publish(
            TopicArn=NOTIFY_TOPIC_ARN,
            Subject="Review app started (no public IP yet)",
            Message=(
                f"The review host ({instance_id}) started but has no public IPv4 "
                f"address after {IP_LOOKUP_ATTEMPTS * IP_LOOKUP_DELAY_SECONDS}s.\n\n"
                "Check the instance in the EC2 console -- it may still be "
                "finishing, or it may have started into a subnet that no longer "
                "auto-assigns public addresses.\n"
            ),
        )
        print(f"{instance_id} started with no public IP")
        return {"instance": instance_id, "ip": None}

    urls = "\n".join(f"  http://{ip}:{port}" for port in PORTS)
    sns.publish(
        TopicArn=NOTIFY_TOPIC_ARN,
        Subject="Review app is up -- new URL",
        Message=(
            f"The review host is running and reachable at:\n\n{urls}\n\n"
            f"Instance: {instance_id}\n"
            f"Public IP: {ip}\n\n"
            "This address changes every time the host starts (it is stopped "
            "nightly at 22:00 and all weekend, and a stopped instance loses its "
            "public IP), so use the link above rather than an older bookmark.\n"
        ),
    )
    print(f"Published review-app URL for {instance_id}: {ip} ports={PORTS}")
    return {"instance": instance_id, "ip": ip, "ports": PORTS}
