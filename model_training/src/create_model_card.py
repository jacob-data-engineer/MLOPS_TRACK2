"""Creates a SageMaker Model Card for an already-registered model package, so
its metrics actually render in the new Studio UI's "Performance metrics"
panel.

Root cause (confirmed via github.com/aws/sagemaker-python-sdk/issues/4727,
closed 2024-07-12): that panel does not read ModelMetrics.ModelQuality.
Statistics directly off the model package, no matter how correctly the
attached evaluation.json is structured -- it reads from a Model Card
resource instead. RegisterModel (pipeline_definition.py) never created one,
which is why every version's tiles showed "-" regardless of evaluate.py's
JSON shape.

No manual re-parsing of evaluation.json needed: ModelPackage.
discover_evaluation_details() (sagemaker.model_card) already reads
ModelMetrics.ModelQuality.Statistics.S3Uri straight off the model package
and parses it with the same schema our evaluate.py already emits
(binary_classification_metrics -> confusion_matrix/precision/recall/
accuracy/f1) -- confirmed by reading the installed SDK's own
evaluation_metric_parsers.py, not guessed. This script just wires that
auto-discovery up and creates the Model Card resource itself, which is the
one step that was actually missing.

    uv run python create_model_card.py <model-package-arn>
"""

import sys

import boto3
from sagemaker.model_card import ModelCard, ModelPackage
from sagemaker.session import Session

from config import CONFIG


def create_model_card_for(model_package_arn: str):
    sagemaker_session = Session(boto_session=boto3.Session(region_name=CONFIG.region))

    model_package_details = ModelPackage.from_model_package_arn(
        model_package_arn=model_package_arn,
        sagemaker_session=sagemaker_session,
    )

    # name must be unique per model card resource, not per model package --
    # reuse the package's own group+version so it's traceable back. Parsed
    # straight out of the ARN (.../model-package/<group>/<version>) rather
    # than read off CONFIG, so this works for whichever of the four
    # per-size groups the given ARN actually belongs to.
    group_name, version_suffix = model_package_arn.rsplit("/", 2)[-2:]
    card_name = f"{group_name}-v{version_suffix}-card"

    card = ModelCard(
        name=card_name,
        sagemaker_session=sagemaker_session,
        model_package_details=model_package_details,  # triggers auto-discovery of evaluation_details from ModelMetrics
    )
    card.create()
    print(f"Created Model Card '{card_name}' for {model_package_arn}")
    print(f"Evaluation jobs auto-discovered: {[j.name for j in (card.evaluation_details or [])]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    create_model_card_for(sys.argv[1])
