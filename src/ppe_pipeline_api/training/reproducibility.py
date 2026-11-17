"""Pin the pipeline definition and reuse its exact inputs for repeat runs."""
import json
import os
import re

import boto3


def evaluation_inputs(s3, bucket, run_id, uris):
    """Keep ProcessingJob environment values within the 256-character limit."""
    if len(uris) <= 256:
        return uris
    key = f'custom/runs/{run_id}/evaluation-inputs.json'
    uri = f's3://{bucket}/{key}'
    if len(uri) > 256:
        raise ValueError('Evaluation manifest location exceeds the processing environment limit.')
    s3.put_object(Bucket=bucket, Key=key,
                  Body=json.dumps({'uris': [u.strip() for u in uris.split(',') if u.strip()]}).encode(),
                  ContentType='application/json')
    return uri


def validate_processing_environment(definition, parameters):
    doc = json.loads(definition)
    values = {p['Name']: p.get('DefaultValue', '') for p in doc.get('Parameters', [])}
    values.update(parameters)
    for step in doc.get('Steps', []):
        if step.get('Type') != 'Processing':
            continue
        for key, value in step.get('Arguments', {}).get('Environment', {}).items():
            if isinstance(value, dict) and str(value.get('Get', '')).startswith('Parameters.'):
                value = values.get(value['Get'].split('.', 1)[1], '')
            if isinstance(value, str) and len(value) > 256:
                raise ValueError(f"{step['Name']} environment {key} has {len(value)} characters; AWS allows 256. "
                                 'Queue a new run using the updated evaluation pipeline instead of repeating this older recipe.')


def freeze_definition(definition):
    """Replace mutable container tags and static processing-code inputs."""
    from .. import data
    doc = json.loads(definition)
    dependencies = []

    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            for key, value in list(node.items()):
                if key in ('TrainingImage', 'ImageUri', 'Image') and isinstance(value, str):
                    match = re.fullmatch(r'(\d+)\.dkr\.ecr\.([^.]+)\.amazonaws\.com/([^@:]+):([^:]+)', value)
                    if match:
                        account, region, repository, tag = match.groups()
                        result = boto3.client('ecr', region_name=region).describe_images(
                            registryId=account, repositoryName=repository, imageIds=[{'imageTag': tag}])
                        digest = result['imageDetails'][0]['imageDigest']
                        node[key] = value.rsplit(':', 1)[0] + '@' + digest
                        dependencies.append({'source': value, 'preserved': node[key]})
                if key == 'S3Input' and isinstance(value, dict):
                    uri = value.get('S3Uri')
                    if isinstance(uri, str) and uri.startswith('s3://') and '/_training_inputs/' not in uri:
                        artifact = data.preserve_training_base(uri)
                        value['S3Uri'] = artifact['uri']
                        dependencies.append(artifact)
                visit(value)
    visit(doc)
    return json.dumps(doc), dependencies


def pin_pipeline(sm, name, saved, current):
    saved = saved or {}
    name = saved.get('name') or name
    version = saved.get('version_id')
    if version is None:
        versions = sm.list_pipeline_versions(PipelineName=name, SortOrder='Descending',
                                             MaxResults=1).get('PipelineVersionSummaries') or []
        if not versions:
            raise ValueError('The training pipeline has no version to preserve.')
        version = versions[0]['PipelineVersionId']
    version = int(version)
    definition = sm.describe_pipeline(PipelineName=name, PipelineVersionId=version)['PipelineDefinition']
    dependencies = saved.get('dependencies') or []
    if saved.get('definition') and json.loads(saved['definition']) != json.loads(definition):
        raise ValueError('The saved pipeline definition no longer matches its version.')
    if not saved:
        frozen, dependencies = freeze_definition(definition)
        if json.loads(frozen) != json.loads(definition):
            updated = sm.update_pipeline(PipelineName=name, PipelineDefinition=frozen)
            version = int(updated['PipelineVersionId'])
            definition = frozen
    parameters = dict(current)
    if saved.get('recorded_parameters'):
        parameters = dict(saved['recorded_parameters'])
        # Only identifiers and result destinations change on a repeated run.
        for key in ('RunId', 'DatasetVersion', 'RunMetadataUri', 'EvaluationOutputUri'):
            if key in current:
                parameters[key] = current[key]
    declared = {p['Name']: p for p in json.loads(definition).get('Parameters', [])}
    unknown = set(parameters) - set(declared)
    missing = {k for k, p in declared.items() if 'DefaultValue' not in p and k not in parameters}
    if unknown or missing:
        raise ValueError(f'Pipeline inputs do not match version {version}: unknown {sorted(unknown)}, missing {sorted(missing)}.')
    validate_processing_environment(definition, parameters)
    return {'name': name, 'version_id': version, 'definition': definition,
            'app_build': saved.get('app_build') or os.environ.get('APP_BUILD_TAG') or os.environ.get('BUILD_TAG', ''),
            'dependencies': dependencies,
            'recorded_parameters': parameters}, parameters
