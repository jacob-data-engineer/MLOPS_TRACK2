"""Track2 training control -- dataset/evaluation/weights S3 browsing, GPU
status, training progress, the ad-hoc training queue, and the per-run
training audit. Backs `/track2/models` (labelled "Track2 Training" in nav
to avoid colliding with the existing PPE `/models` page).

This is the largest single piece ported from the old review app's Models
tab -- see `track2_shared.py` for the underlying business logic; this
router is mostly just parameter parsing + response shaping."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel

from .. import dataset_yaml as dy
from .. import hyperparams as hp
from .. import track2_shared as t2
from ..auth import require_session

router = APIRouter()


@router.get("/scratch-check")
def scratch_check(detector_slug: str = "", architecture: str = "",
                  session=Depends(require_session)) -> dict:
    """Whether this Class may be trained from scratch, and why not.

    Its own endpoint because the page cannot work it out: the registries
    it loads are the four PRODUCTION groups, and a Class registers into
    `yolo11m-helmet-detector`, which is not among them. The card looked
    that group up, found nothing, and offered scratch for exactly the
    Classes the rule exists to protect -- while the server refused the
    submission.

    Returning the server's own sentence rather than a boolean is the
    point: the reason shown on the card is then literally the reason the
    run would be refused with, and the two cannot drift.
    """
    return {"refusal": t2.scratch_refusal(detector_slug, architecture)}


@router.get("/live")
def get_models_live(session=Depends(require_session)) -> dict:
    """The three things on the Models page that go stale while you look.

    The page's main payload lists registries, S3 folders and the whole
    dataset browser -- far too heavy to poll, so it is fetched once on
    load and the page then sits on it. That was fine until it carried
    live state too: on 2026-10-06 the GPU card read "Available" in green
    while a training job was downloading its dataset, and the queue panel
    showed "Succeeded -- Epoch 0 / 50" for a run that had failed, because
    both were rendered from a snapshot taken before either happened. The
    one person who checks that card is the one about to press Start.

    So the three that move get their own endpoint, cheap enough to poll:
    a training-jobs list, a pipeline-execution list and an in-memory
    queue read.
    """
    training_progress = t2.get_training_progress(t2.progress_pipeline_name())
    return {
        "training_progress": training_progress,
        "custom_run_id": t2.custom_run_in_progress(),
        "pending_message": t2.pending_submission_overlay(training_progress),
        "gpu_jobs": t2.get_gpu_resource_usage(),
        "cpu_jobs": t2.get_cpu_resource_usage(),
        "queue": t2.queue_snapshot(),
    }


@router.get("")
def get_models(
    # None (the param genuinely omitted) means "give me the nice landing
    # default"; "" (the param explicitly present but empty) means "the
    # bucket root" -- these must stay distinguishable. Collapsing both to
    # "" via `browse_prefix or DATASETS_S3_PREFIX` made navigating to the
    # bucket root silently snap back to the default subfolder instead,
    # since an empty string is falsy in that expression regardless of
    # which of the two cases it actually was.
    browse_prefix: str | None = None,
    weights_browse_prefix: str = "",
    # Same None-vs-"" distinction as browse_prefix above: None means "give
    # me the landing default", "" means the operator navigated to the
    # bucket root and must stay there.
    eval_browse_prefix: str | None = None,
    session=Depends(require_session),
) -> dict:
    dataset_prefix = t2.DATASETS_S3_PREFIX if browse_prefix is None else browse_prefix
    eval_prefix = t2.DATASETS_S3_PREFIX if eval_browse_prefix is None else eval_browse_prefix
    registries = t2.all_model_registries()
    # Report whichever pipeline ran MOST RECENTLY -- not "the custom one
    # while it is executing". That older test was true only during a
    # custom run, so the moment one ended the panel fell back to the
    # production pipeline: a custom run that had just FAILED showed as
    # "Succeeded -- Epoch 0 / 50", a different pipeline's result from
    # four days earlier.
    custom_run_id = t2.custom_run_in_progress()
    training_progress = t2.get_training_progress(t2.progress_pipeline_name())
    pending_message = t2.pending_submission_overlay(training_progress)
    return {
        "registries": t2.serialize_registries_brief(),
        "latest_approved": t2.latest_approved_lookup(registries),
        "generation_sizes": t2.GENERATION_SIZES,
        "generation_labels": t2.GENERATION_LABELS,
        "size_labels": t2.SIZE_LABELS,
        "default_dataset": t2.DEFAULT_TRAIN_DATASET,
        "training_progress": training_progress,
        # The open pipeline's currently-running id, or None. Also decides
        # which pipeline training_progress above reports on, so the two
        # cannot describe different runs.
        "custom_run_id": custom_run_id,
        "pending_message": pending_message,
        "gpu_jobs": t2.get_gpu_resource_usage(),
        "queue": t2.queue_snapshot(),
        # Files as well as folders, the same shape `weights_browser`
        # uses: the browser can then offer the actual data.yaml rather
        # than a folder tick box that says nothing about which file it is
        # about to read.
        "dataset_browser": dict(
            zip(("folders", "yaml_files"), t2.list_s3_folders_and_yaml_files(dataset_prefix)),
            prefix=dataset_prefix,
        ),
        # Files as well as folders, from one list_objects_v2 round trip.
        # Without the files the browser could only select a folder, so the
        # only reachable checkpoints were ones literally named model.pt --
        # a convention only the upload path follows.
        "weights_browser": dict(
            zip(("folders", "weight_files"), t2.list_s3_folders_and_weight_files(weights_browse_prefix)),
            prefix=weights_browse_prefix,
        ),
        "eval_browser": {
            "prefix": eval_prefix,
            "folders": t2.list_s3_folders(eval_prefix),
        },
    }


@router.get("/audit")
def training_audit(limit: int = 50, session=Depends(require_session)) -> dict:
    """Full record of every submitted run: the exact folders and file names
    on both sides, every parameter, the notes, which model version came out
    of it, and when it started and ended."""
    return {"audits": t2.scan_training_audits(limit=min(max(limit, 1), 200))}


@router.get("/audit/{audit_id}/export.csv")
def export_training_audit(audit_id: str, session=Depends(require_session)) -> Response:
    """The whole run as a CSV, including EVERY file name it used.

    Read back from the S3 manifest rather than the audit row, whose file
    list is a bounded sample -- an export that quietly stopped at 1,000
    names would be worse than none.
    """
    audit = next(
        (a for a in t2.scan_training_audits(limit=500) if a.get("id") == audit_id), None,
    )
    if not audit:
        raise HTTPException(status_code=404, detail="Run not found.")
    started = str(audit.get("submitted_at") or "")[:10]
    return Response(
        content=t2.audit_as_csv(audit),
        media_type="text/csv",
        headers={
            "Content-Disposition":
                f'attachment; filename="training-run-{started}-{audit_id}.csv"',
        },
    )


@router.get("/audit/{audit_id}/weights-url")
def training_audit_weights_url(audit_id: str, session=Depends(require_session)) -> dict:
    """A short-lived download link for the .pt this run produced.

    Presigned rather than streamed: a checkpoint is 19MB+ and the API
    container shares one small box with everything else, so the bytes go
    browser-to-S3 and never through here.
    """
    audit = next(
        (a for a in t2.scan_training_audits(limit=500) if a.get("id") == audit_id), None,
    )
    if not audit:
        raise HTTPException(status_code=404, detail="Run not found.")
    uri = str(audit.get("weights_uri") or "")
    if not uri:
        raise HTTPException(
            status_code=404,
            detail=(
                "No weights published for this run yet. They are copied out "
                "shortly after the model registers -- try again in a minute, or "
                "check the run actually produced a model."
            ),
        )
    try:
        return {"url": t2.presign_weights_download(uri), "uri": uri}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not prepare the download: {exc}")


@router.get("/audit/files/used-by")
def runs_using_file(name: str, session=Depends(require_session)) -> dict:
    """Which training runs used a given image -- reverse lineage.

    The reason the file names are per-row rather than a blob on the audit
    record: a list can be read back, but only an indexed row can be asked
    this. `name` is the path as recorded, relative to the dataset folder,
    e.g. images/train/live-....jpg
    """
    audit_ids = t2.runs_using_file(name)
    audits = {a["id"]: a for a in t2.scan_training_audits(limit=500)}
    return {
        "name": name,
        "runs": [
            {
                "audit_id": audit_id,
                "submitted_at": (audits.get(audit_id) or {}).get("submitted_at", ""),
                "status": (audits.get(audit_id) or {}).get("status", ""),
                "model": (audits.get(audit_id) or {}).get("model"),
            }
            for audit_id in audit_ids
        ],
    }


@router.post("/audit/{audit_id}/delete")
def delete_training_audit(audit_id: str, session=Depends(require_session)) -> dict:
    """Removes a run from the history AND deletes the model version it
    produced -- the two are one record, so they go together."""
    try:
        return t2.delete_training_audit(audit_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Run not found (already deleted?).")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


class TrainingParams(BaseModel):
    generation: str
    size: str
    epochs: int = 50
    img_size: int = 640
    batch: int = 16
    seed: int = 42
    dataset_folder: str
    # Optional: unset, the pipeline evaluates on the training dataset, which
    # is what it has always done.
    eval_dataset_folder: str = ""
    base_model_folder: str = ""
    notes: str = ""
    # Custom-class runs only: fine-tune from a registered package instead
    # of a browsed/uploaded folder. Exactly one of the two is set.
    base_model_package_arn: str = ""
    # Which YOLO architecture the base checkpoint is. Declared by whoever
    # uploaded it, verified against the file itself at upload time.
    architecture: str = ""
    # Train on the dataset's own classes, through the custom pipeline.
    # Carried on queued entries too, so a queued custom run still runs as
    # one when its turn comes.
    custom_classes: bool = False
    # An exact .pt/.tar.gz key anywhere in the bucket, rather than a folder
    # that happens to contain model.pt.
    base_model_key: str = ""
    # Start from random weights. Resolved to the pre-built seed for
    # `architecture`; see make_scratch_checkpoint.py for why scratch is a
    # checkpoint rather than a flag.
    base_model_scratch: bool = False
    # Which detector this run trains. Decides both the taxonomy the dataset
    # is checked against and the per-size registry group the version lands
    # in. Empty means the shared open groups, as before detectors existed.
    detector_slug: str = ""
    # Early stopping: epochs without improvement before the run stops
    # itself. 0 leaves it to Ultralytics.
    patience: int = 0
    #: Everything else Ultralytics takes -- optimiser, learning rate,
    #: schedule, loss weights, augmentation -- as plain {key: value}.
    #:
    #: OVERRIDES ONLY. A key absent means "whatever Ultralytics does by
    #: default", so a run that sets none of this trains exactly as it did
    #: before the field existed. The trainer validates every key against
    #: the Ultralytics it actually has installed and REFUSES one it does
    #: not know, rather than dropping it in silence.
    #:
    #: Typed and checked against hyperparams.py by every route below
    #: before anything is submitted -- a 400 naming the key, as a plain
    #: string the card can show (a Pydantic validator would answer 422
    #: with a list, which the card renders as "[object Object]").
    hyperparams: dict = {}


def _typed_hyperparams(payload: "TrainingParams") -> None:
    """Replace the request's hyperparameters with their typed form, or
    refuse with a 400 naming the key and value. Legacy strings ("0.01",
    "false", "10") become the numbers and booleans Ultralytics requires."""
    try:
        payload.hyperparams = hp.normalize_hyperparams(payload.hyperparams)
    except hp.HyperparamError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/train/dataset")
def start_training(payload: TrainingParams, session=Depends(require_session)) -> dict:
    _typed_hyperparams(payload)
    already_running = t2.training_already_running_reason(t2.get_training_progress())
    if already_running:
        raise HTTPException(status_code=409, detail=already_running)
    try:
        build_id = t2.start_training_run(
            payload.generation, payload.size, payload.epochs, payload.dataset_folder,
            payload.base_model_folder, payload.notes[:2048], payload.img_size, payload.batch, payload.seed,
            payload.eval_dataset_folder, source="manual",
            base_model_key=payload.base_model_key,
            base_model_scratch=payload.base_model_scratch,
            base_model_package_arn=payload.base_model_package_arn,
            architecture=payload.architecture,
            hyperparams=payload.hyperparams,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Training trigger failed: {exc}")
    return {"message": f"Training started (YOLO{payload.generation}-{payload.size}, {payload.epochs} epochs, {payload.dataset_folder}): {build_id}", "build_id": build_id}


@router.post("/train/custom")
def start_custom_training(payload: TrainingParams, session=Depends(require_session)) -> dict:
    """Trains on the chosen dataset's own classes, through the separate
    custom pipeline and registry group -- see
    model_training/custom_training/README.md for why it is separate rather
    than a mode of the production one."""
    _typed_hyperparams(payload)
    # The same one-GPU guard /train/dataset has. Without it a custom run
    # could be launched on top of an executing one and die with
    # SAGEMAKER_RESOURCE_LIMIT, which is the whole reason the guard exists.
    already_running = t2.training_already_running_reason(t2.get_training_progress())
    if already_running:
        raise HTTPException(status_code=409, detail=already_running)
    try:
        # Source selection is validated by the one dispatcher inside
        # start_custom_training_run -- there are four ways to choose a base
        # model now, and repeating that check here is how the two drift.
        result = t2.start_custom_training_run(
            dataset_folder=payload.dataset_folder,
            eval_dataset_folder=payload.eval_dataset_folder,
            base_model_folder=payload.base_model_folder,
            base_model_key=payload.base_model_key,
            base_model_scratch=payload.base_model_scratch,
            base_model_package_arn=payload.base_model_package_arn,
            architecture=payload.architecture,
            detector_slug=payload.detector_slug,
            patience=payload.patience,
            hyperparams=payload.hyperparams,
            epochs=payload.epochs, img_size=payload.img_size,
            batch=payload.batch, seed=payload.seed,
            notes=payload.notes[:2048],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Custom training failed to start: {exc}")
    return {
        "message": (
            f"Custom training started ({result['run_id']}) -- classes come from the dataset, "
            f"and the model registers to {t2.CUSTOM_REGISTRY_GROUP}."
        ),
        **result,
    }


class BaseModelUploadRequest(BaseModel):
    filename: str
    architecture: str


class BaseModelVerifyRequest(BaseModel):
    key: str
    architecture: str


class DetectorRequest(BaseModel):
    slug: str
    label: str = ""
    # {"0": "person", "1": "helmet"} -- indices as strings because that is
    # how JSON carries them and how data.yaml is read back.
    classes: dict[str, str]


@router.get("/detectors")
def get_detectors(session=Depends(require_session)) -> dict:
    """Every detector and the registry groups its models land in."""
    detectors = t2.list_detectors()
    return {
        "detectors": [
            {
                "slug": d["slug"],
                "label": d["label"],
                "builtin": d["builtin"],
                "classes": {str(k): v for k, v in d["classes"].items()},
                # Flat, generation 11 -- what every existing caller reads.
                "groups": {size: t2.detector_group(d, size) for size in t2.REGISTRY_SIZES},
                # Nested {generation: {size: group}}, which is what the
                # merged card needs to show where a YOLO26 run would land.
                "groups_by_generation": {
                    generation: {
                        size: group
                        for gen, size, group in t2.detector_groups(d)
                        if gen == generation
                    }
                    for generation in t2.detector_generations(d)
                },
                "generations": list(t2.detector_generations(d)),
            }
            for d in detectors
        ],
        "sizes": list(t2.REGISTRY_SIZES),
        # What a new model may be created as. The card builds its
        # generation/size picker from this rather than a hardcoded list,
        # so adding a generation server-side is all it takes.
        "generations": [
            {
                "generation": generation,
                "label": t2.GENERATION_LABELS.get(generation, f"YOLO{generation}"),
                "sizes": list(sizes),
                "size_labels": {size: t2.SIZE_LABELS.get(size, size) for size in sizes},
            }
            for generation, sizes in t2.GENERATION_SIZES.items()
        ],
        "default_generation": t2.DEFAULT_GENERATION,
    }


@router.post("/detectors")
def save_detector(payload: DetectorRequest, session=Depends(require_session)) -> dict:
    try:
        detector = t2.upsert_detector(payload.slug, payload.label, payload.classes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not save the detector: {exc}")
    return {"slug": detector["slug"], "saved": True}


@router.post("/detectors/{slug}/delete")
def remove_detector(slug: str, force: bool = False, session=Depends(require_session)) -> dict:
    """Removes a detector. `force` takes its registered versions with it --
    off by default, because that is not recoverable."""
    try:
        return t2.delete_detector(slug, force=force)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not delete the detector: {exc}")


class RenameDetectorRequest(BaseModel):
    #: The new identifier. Becomes part of the registry group name, so it
    #: has the same charset rules the slug always had.
    new_slug: str
    #: The display name. Optional -- renaming the identifier without
    #: retitling it is a reasonable thing to want.
    label: str = ""


@router.post("/detectors/{slug}/rename")
def rename_detector(slug: str, payload: RenameDetectorRequest,
                    session=Depends(require_session)) -> dict:
    """Gives a Model Name a new identifier and points the workflow at it.

    The identifier derives the registry group name, and a model package
    group CANNOT be renamed -- SageMaker offers create, delete, describe
    and list, and no update. So this does the two things that are
    possible: future runs register into the new groups, and the old
    identifier is remembered so the versions already registered stay
    findable instead of vanishing from the card.

    Then it retargets the live configuration, because a rename that left
    five settings pointing at an identifier that no longer resolves is
    not a rename.
    """
    try:
        detector = t2.rename_detector(slug, payload.new_slug, payload.label)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(status_code=400,
                            detail=f"Could not rename the Model Name: {exc}")

    # After the rename, and never conditional on it: the detector exists
    # under its new name whatever happens here, and a half-retargeted
    # configuration is still better than one pointing entirely at a name
    # that is gone.
    cascaded = t2.retarget_detector_references(slug, detector["slug"])
    return {
        "detector": {
            "slug": detector["slug"],
            "label": detector["label"],
            "classes": detector["classes"],
            "previous_slugs": detector.get("previous_slugs") or [],
        },
        #: What now follows the new name, for the card to say plainly.
        "cascaded": cascaded,
        #: The groups future runs register into.
        "groups": [g for _, _, g in t2.detector_groups(detector)],
        #: Kept readable, not moved. Named so nobody goes looking for
        #: versions that are exactly where they always were.
        "kept_groups": [
            g for _, _, g in t2.detector_history_groups(detector)
            if g not in {x for _, _, x in t2.detector_groups(detector)}
        ],
    }


class DatasetSource(BaseModel):
    images: str = ""
    labels: str = ""


class DatasetLayoutRequest(BaseModel):
    # The original one-folder-per-side fields. Still accepted, because the
    # single-source panel sends them.
    train_images: str = ""
    train_labels: str = ""
    eval_images: str = ""
    eval_labels: str = ""
    # Several folders per side, merged into one split. When present these
    # take precedence -- a caller sending both meant the list.
    train_sources: list[DatasetSource] = []
    eval_sources: list[DatasetSource] = []


def _splits_of(payload: DatasetLayoutRequest) -> dict:
    def sources(listed: list[DatasetSource], images: str, labels: str) -> list:
        if listed:
            return [(s.images, s.labels) for s in listed if s.images]
        return [(images, labels)] if images else []

    return {
        "train": sources(payload.train_sources, payload.train_images, payload.train_labels),
        "val": sources(payload.eval_sources, payload.eval_images, payload.eval_labels),
    }


@router.post("/dataset-layout")
def dataset_layout(payload: DatasetLayoutRequest, session=Depends(require_session)) -> dict:
    """Pairs the chosen image and label folders and reports what it found.

    Read-only: nothing is written. This is the preview shown before the
    dataset is assembled, so a wrong label folder is caught here rather
    than as a run that trains happily on unannotated images.
    """
    try:
        return t2.inspect_dataset_layout(splits=_splits_of(payload))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read those folders: {exc}")


class AssembleDatasetRequest(DatasetLayoutRequest):
    # {"0": "helmet", ...} -- index order is the model's output order.
    classes: dict[str, str]
    destination: str = ""


class DatasetYamlInPlaceRequest(DatasetLayoutRequest):
    """The same folders `dataset-assemble` takes, named instead of copied."""
    # {"0": "helmet", ...} -- index order is the model's output order.
    classes: dict[str, str]
    #: Replace a data.yaml this card did not write. Off unless ticked, so
    #: one written by hand cannot be destroyed by a build nobody read.
    overwrite: bool = False
    #: The training container drops boxes whose class the yaml does not
    #: name. With it off, a write naming fewer classes than the labels use
    #: is refused; the preview reports the mismatch either way.
    drop_undeclared: bool = False


class DiscoverRequest(BaseModel):
    root: str
    images_subpath: str = "images"
    labels_subpath: str = "labels"


@router.post("/dataset-discover")
def dataset_discover(payload: DiscoverRequest, session=Depends(require_session)) -> dict:
    """Every week folder under a root, expanded into image/label pairs.

    Read-only. Show it one week's shape and it finds the rest, so a week
    added next month needs no new configuration.
    """
    try:
        return t2.discover_dataset_sources(
            root=payload.root,
            images_subpath=payload.images_subpath,
            labels_subpath=payload.labels_subpath,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read those folders: {exc}")


@router.post("/dataset-assemble")
def dataset_assemble(payload: AssembleDatasetRequest, session=Depends(require_session)) -> dict:
    """Copies the chosen folders into the layout Ultralytics requires."""
    try:
        return t2.assemble_dataset(
            splits=_splits_of(payload),
            classes=payload.classes,
            destination=payload.destination,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not assemble the dataset: {exc}")


@router.post("/dataset-yaml-preview")
def dataset_yaml_preview(payload: DatasetYamlInPlaceRequest,
                         session=Depends(require_session)) -> dict:
    """The yaml these folders would produce. Writes nothing.

    Shown before it is written because a generated data.yaml IS the
    training decision -- which folders, in which split, under which class
    names -- and it should be read once before a GPU run depends on it.
    """
    try:
        return dy.build_in_place(
            splits=_splits_of(payload),
            classes=payload.classes,
            overwrite=payload.overwrite,
            drop_undeclared=payload.drop_undeclared,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:                                       # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not build the yaml: {exc}")


@router.post("/dataset-yaml-write")
def dataset_yaml_write(payload: DatasetYamlInPlaceRequest,
                       session=Depends(require_session)) -> dict:
    """Writes that yaml to S3. Copies no images."""
    try:
        return dy.write_in_place(
            splits=_splits_of(payload),
            classes=payload.classes,
            overwrite=payload.overwrite,
            drop_undeclared=payload.drop_undeclared,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:                                       # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Could not write the yaml: {exc}")


class YamlTextRequest(BaseModel):
    #: The file's contents. A data.yaml is a few hundred bytes, so it
    #: comes through the request rather than being uploaded to S3 first:
    #: reading class names is not a reason to leave a file in the bucket.
    text: str = ""
    #: What it was called on disk, for the message only.
    filename: str = ""


@router.post("/dataset-classes/from-text")
def dataset_classes_from_text(payload: YamlTextRequest,
                              session=Depends(require_session)) -> dict:
    """Class names out of a data.yaml the operator has on their machine.

    Parsed HERE rather than in the browser. The alternative was a YAML
    reader in TypeScript, and this codebase already carries two
    hand-rolled readers of this exact key -- a third is the one that
    drifts, and the failure mode is a silently shifted class index.
    """
    if not payload.text.strip():
        raise HTTPException(status_code=422, detail="That file is empty.")
    # A data.yaml is a few hundred bytes. Anything of this size is not
    # one, and parsing it would be somebody else's problem arriving here.
    if len(payload.text) > 1_000_000:
        raise HTTPException(status_code=422,
                            detail="That file is too large to be a data.yaml.")
    try:
        classes = t2.classes_from_yaml_text(payload.text)
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(
            status_code=400,
            detail=f"Could not read {payload.filename or 'that file'} as YAML: {exc}")
    return {
        "folder": "",
        "key": payload.filename,
        "classes": {str(k): v for k, v in classes.items()},
        "matches_production": classes == t2.PRODUCTION_CLASSES,
        "production_classes": {str(k): v for k, v in t2.PRODUCTION_CLASSES.items()},
    }


@router.get("/dataset-classes")
def dataset_classes(folder: str = "", key: str = "",
                    session=Depends(require_session)) -> dict:
    """What taxonomy a dataset declares, and therefore which pipeline it
    can run through.

    The UI uses this to preselect the pipeline instead of making the
    operator work out from a rejection that their dataset is not
    person/helmet. Deliberately advisory: it does NOT route the run. The
    production path carries meaning beyond its taxonomy -- the replay
    lineage, the production pointer, the registry group the production
    pointer is chosen from -- so a person/helmet dataset can legitimately
    want either pipeline, and that stays the operator's choice.
    """
    # `key` names the exact yaml; `folder` keeps the old meaning, which
    # is `<folder>/data.yaml`. Both, because the browser now picks a file
    # while the training form still has only a folder in hand.
    source = key or (f"{folder.strip('/')}/data.yaml" if folder else "")
    if not source:
        raise HTTPException(status_code=422, detail="Pass a folder or a key.")
    try:
        classes = t2.classes_from_yaml_key(source)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not read '{source}': {exc}")
    return {
        "folder": source.rsplit("/", 1)[0] if "/" in source else "",
        "key": source,
        "classes": {str(k): v for k, v in classes.items()},
        "matches_production": classes == t2.PRODUCTION_CLASSES,
        "production_classes": {str(k): v for k, v in t2.PRODUCTION_CLASSES.items()},
    }


@router.get("/base-models/architectures")
def base_model_architectures(session=Depends(require_session)) -> dict:
    """The architectures an upload may declare itself as."""
    return {"architectures": list(t2.KNOWN_ARCHITECTURES)}


@router.post("/base-models/upload")
def base_model_upload(
    architecture: str = Form(...),
    file: UploadFile = File(...),
    session=Depends(require_session),
) -> dict:
    """Uploads a checkpoint through this app, then reports what it actually is.

    Same-origin, so the bucket's CORS policy never applies -- see
    upload_base_model_fileobj for why the presigned route below cannot be
    used from this UI. Verification is the same either way: the file is
    read back and its real architecture compared with the declared one.
    """
    try:
        stored = t2.upload_base_model_fileobj(file.file, file.filename or "", architecture)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Upload failed: {exc}")
    try:
        return t2.inspect_checkpoint(stored["key"], architecture)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Uploaded, but could not verify it: {exc}")


@router.post("/base-models/upload-url")
def base_model_upload_url(payload: BaseModelUploadRequest, session=Depends(require_session)) -> dict:
    """Hands back a presigned PUT so the browser uploads straight to S3.

    The architecture is required here rather than after the fact: it
    decides where the file lands, so a checkpoint is filed under what it
    claims to be from the moment it exists.
    """
    try:
        return t2.presign_base_model_upload(payload.filename, payload.architecture)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not prepare the upload: {exc}")


@router.post("/base-models/verify")
def base_model_verify(payload: BaseModelVerifyRequest, session=Depends(require_session)) -> dict:
    """Reads an uploaded checkpoint and reports what it actually is.

    Called after the browser's PUT completes. A mismatch does not delete
    the upload or block the run -- it is surfaced, the way the training
    audit surfaces a hyperparameter that did not reach the job.
    """
    try:
        t2.assert_inspectable_checkpoint(payload.key)
        return t2.inspect_checkpoint(payload.key, payload.architecture)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not verify the checkpoint: {exc}")


@router.get("/base-models/library")
def yolo_weights_library(session=Depends(require_session)) -> dict:
    """The official YOLO releases mirrored into this bucket.

    The third way to import, beside an upload and a browse: the usual
    starting point for a brand new Class is the stock COCO checkpoint,
    and before this that meant finding it on the internet and uploading
    it by hand every time.
    """
    try:
        return {"weights": t2.list_yolo_weights_library(),
                "prefix": t2.YOLO_WEIGHTS_LIBRARY_PREFIX}
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(status_code=400,
                            detail=f"Could not read the weights library: {exc}")


class BaseModelRegisterRequest(BaseModel):
    key: str
    detector_slug: str
    architecture: str
    notes: str = ""
    #: File it under a Class whose class names it does NOT detect. The
    #: deliberate exception, not the default: training rebuilds the head
    #: for whatever the dataset declares, so a COCO checkpoint really can
    #: become the first version of a three-class Class -- but until that
    #: run happens the weights and the registry disagree, so the version
    #: is marked a seed and registered Pending rather than Approved.
    allow_class_mismatch: bool = False
    #: Register only these of the checkpoint's classes, by their ORIGINAL
    #: index in the file.
    #:
    #: Given a strict subset, the head is cut down to them before anything
    #: is registered -- so the version really does detect exactly what the
    #: Model Name declares, and is deployable rather than a seed waiting
    #: on a training run. Empty, or the whole set, registers the file as
    #: it is.
    keep_classes: list[int] = []


@router.post("/base-models/register")
def base_model_register(payload: BaseModelRegisterRequest, session=Depends(require_session)) -> dict:
    """Registers a checkpoint as a version in a detector's registry.

    Either a fresh upload or a `.pt` already sitting in the bucket -- the
    same two sources the training form offers as starting weights.
    """
    try:
        t2.assert_inspectable_checkpoint(payload.key)
        key = payload.key
        notes = payload.notes
        sliced = None
        if payload.keep_classes:
            whole = t2.inspect_checkpoint(key, payload.architecture)["classes"] or {}
            seen: set = set()
            wanted = [int(i) for i in payload.keep_classes
                      if not (int(i) in seen or seen.add(int(i)))]
            # Worth cutting unless it is the identity. A SUBSET obviously
            # needs it; so does a REORDER -- asking for [7, 2] means class
            # 0 is whatever 7 was, and a file registered unchanged would
            # be wrong about every label while erroring nowhere. Asking
            # for exactly what is already there, in order, needs nothing:
            # a byte-copy under a new key is a second artefact saying the
            # same thing.
            if wanted and wanted != list(range(len(whole))):
                sliced = t2.slice_checkpoint_classes(
                    key, wanted, architecture=payload.architecture)
                key = sliced["key"]
                kept = ", ".join(sliced["classes"].values())
                notes = (f"Head cut from {len(whole)} classes to {len(wanted)} "
                         f"({kept}); predictions for those are unchanged. "
                         f"{notes}").strip()
        registered = t2.register_uploaded_base_model(
            key=key,
            detector_slug=payload.detector_slug,
            architecture=payload.architecture,
            notes=notes,
            # A sliced checkpoint matches the Model Name by construction,
            # so it needs no exception -- and must not be handed one, or a
            # genuine mismatch would be waved through with it.
            allow_class_mismatch=payload.allow_class_mismatch and sliced is None,
        )
        if sliced:
            registered["sliced"] = {"key": sliced["key"],
                                    "classes": sliced["classes"]}
        return registered
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not register the checkpoint: {exc}")


@router.get("/base-models/packages")
def base_model_packages(session=Depends(require_session)) -> dict:
    """Registered packages usable as a base for a custom-class run.

    Unapproved packages are included and labelled, not hidden -- a new
    taxonomy's first run often has nothing approved to start from.
    """
    return {"packages": t2.list_base_model_packages()}


class StopTrainingRequest(BaseModel):
    #: Also stop GPU training jobs this app did not start.
    #:
    #: OFF by default, and it has to stay off: those jobs belong to
    #: somebody else, and "Stop Training" meaning "stop everything in the
    #: account" is not what the button says. The card asks before sending
    #: it, naming each job.
    include_foreign: bool = False


@router.post("/train/stop")
def stop_training(payload: StopTrainingRequest = StopTrainingRequest(),
                  session=Depends(require_session)) -> dict:
    """Stops the currently Executing pipeline execution, if any. A stopped
    execution's steps end in a Stopped/Stopping state -- SageMaker does not
    support resuming one, a fresh run has to be started instead.

    With `include_foreign`, also stops GPU training jobs this app did not
    start. The single-GPU guard refuses to run while one of those holds
    the slot, and until now nothing here could clear it -- the slot could
    be held for hours by something the app could see and not touch.
    """
    progress = t2.get_training_progress()
    execution_arn = progress["execution_arn"] if progress and progress["status"] == "Executing" else None
    if execution_arn is None:
        # get_training_progress only lists the PRODUCTION pipeline, so a
        # custom run was unstoppable: nothing could stop it and the
        # single-GPU guard refused every other run until it finished on
        # its own, up to the step's 24h MaxRuntimeInSeconds.
        execution_arn = t2.custom_run_execution_arn()

    foreign = [j for j in t2.get_gpu_resource_usage() if not j.get("own_pipeline")]
    if execution_arn is None and not (payload.include_foreign and foreign):
        if foreign:
            # NOT a bare "nothing is executing". Something IS on the GPU,
            # the app can name it, and saying "nothing" while the guard
            # refuses every run is the card contradicting itself.
            raise HTTPException(
                status_code=409,
                detail="No run started here is executing. "
                       + ", ".join(j["name"] for j in foreign)
                       + " is on the GPU and was not started here -- use "
                         "\"Stop it too\" to end it.",
            )
        raise HTTPException(status_code=409, detail="No training run is currently executing.")

    stopped: list = []
    problems: list = []
    if execution_arn is not None:
        try:
            t2.sm.stop_pipeline_execution(PipelineExecutionArn=execution_arn)
            stopped.append(f"run {execution_arn.rsplit('/', 1)[-1]}")
        except Exception as exc:                                   # noqa: BLE001
            # Reported, not raised: when the operator asked for the
            # foreign jobs too, failing here would leave those running
            # with nothing said about them.
            problems.append(f"{execution_arn.rsplit('/', 1)[-1]}: {exc}")

    if payload.include_foreign:
        out = t2.stop_foreign_gpu_jobs()
        stopped += out["stopped"]
        problems += out["problems"]

    if not stopped:
        raise HTTPException(status_code=400,
                            detail="Stop failed: " + ("; ".join(problems) or "nothing was stopped."))
    message = "Stopping " + ", ".join(stopped) + "..."
    if problems:
        message += " Not stopped: " + "; ".join(problems)
    return {"message": message, "stopped": stopped, "problems": problems}


def _stamp_snapshot(entry: dict) -> None:
    """Record the frozen definition the live slot currently holds.

    Only when it is the SAME ROOT the entry trains on. A snapshot
    restored into `retraining` says nothing about a run pointed at some
    other folder, and stamping it there would be a label that is simply
    untrue -- worse than none, because it reads as provenance.

    Never raises: this is annotation. An entry that queues without its
    snapshot id is a smaller problem than one that does not queue.
    """
    try:
        from .. import pull_control as pc

        chosen = pc.read_config().get("default_dataset") or {}
        if str(chosen.get("set_by") or "") != "snapshot":
            return
        folder = str(chosen.get("folder") or "").strip("/")
        if not folder or folder != str(entry.get("dataset_folder") or "").strip("/"):
            return
        snapshot_id = str(chosen.get("snapshot_id") or "")
        if not snapshot_id:
            return
        _write_snapshot_fields(
            entry, snapshot_id,
            created_at=str(chosen.get("snapshot_created_at") or ""),
            files=chosen.get("files"),
            pinned=bool(chosen.get("pinned")),
            note=str(chosen.get("snapshot_note") or ""),
        )
    except Exception as exc:                                   # noqa: BLE001
        print(f"Could not stamp the snapshot onto a queue entry: {exc}")


def _write_snapshot_fields(entry: dict, snapshot_id: str, *, created_at: str,
                           files, pinned: bool, note: str) -> None:
    entry["snapshot_id"] = snapshot_id
    entry["snapshot_created_at"] = created_at
    entry["snapshot_files"] = files
    entry["snapshot_pinned"] = bool(pinned)
    entry["snapshot_note"] = note
    # Into the notes as well, because that is the field that reaches the
    # registry. A model whose package says only "1 epoch" cannot be
    # traced back to the images it saw.
    existing = str(entry.get("notes") or "")
    if snapshot_id not in existing:
        suffix = f"frozen snapshot {snapshot_id}"
        entry["notes"] = (f"{existing} -- {suffix}" if existing.strip() else suffix)[:2048]


def _freeze_for_entry(entry: dict) -> None:
    """Freeze the dataset this entry will train on, if nothing already did.

    Every run should be answerable to "which images was this trained
    on". The CHAIN freezes a snapshot as one of its steps, so its runs
    always were; a queue-add from the training card never did, and the
    history for those runs had an empty snapshot block and a folder name
    that had moved on since. Asked of a finished run, "which snapshot?"
    had no answer at all.

    A snapshot is a list of names and two small objects -- no image is
    copied -- so freezing one at queue time costs a listing and answers
    the question permanently.

    Never fatal: a run that queues without a frozen record is worse than
    one with it, and far better than one that will not queue.
    """
    folder = str(entry.get("dataset_folder") or "").strip("/")
    if not folder or entry.get("snapshot_id"):
        return
    try:
        from .. import dataset_snapshots as ds
        from .. import pull_control as pc

        config = pc.read_config()
        # Ticked off on the Retention card: no automatic freeze.
        if not config.get("freeze_on_run", True):
            return
        row = ds.create(
            root=folder,
            source_bucket=str(config.get("source_bucket") or ""),
            images_subpath=str(config.get("images_subpath") or "images"),
            labels_subpath=str(config.get("labels_subpath") or "labels"),
            note=f"queued run {entry.get('id', '')}".strip(),
            # The SECOND download, when the held-out set has a prefix of
            # its own. Without it the freeze walked the training root
            # only, so the images the model is SCORED on -- the numbers
            # every registered version is approved on -- had no record.
            # Ignored when it sits under `root`, which is the ordinary
            # case.
            eval_root=str(entry.get("eval_dataset_folder") or "").strip("/"),
        )
        _write_snapshot_fields(
            entry, str(row.get("snapshot_id") or ""),
            created_at=str(row.get("created_at") or ""),
            files=row.get("file_count"),
            # Whether a RUN would read a pinned list. A live week-folder
            # yaml is not pinned today -- but the manifest is written
            # either way, so restoring this snapshot later pins it.
            pinned=str(row.get("names_from") or "") == "list",
            note="frozen when the run was queued",
        )
    except Exception as exc:                                   # noqa: BLE001
        print(f"Could not freeze a snapshot for queue entry {entry.get('id')}: {exc}")


@router.post("/queue/add")
def add_to_queue(payload: TrainingParams, session=Depends(require_session)) -> dict:
    # Before anything else: an entry can sit for hours, and a value that
    # cannot be sent should be refused now, not when its turn comes.
    _typed_hyperparams(payload)
    try:
        epochs, img_size, batch, seed = t2.validate_training_params(
            payload.epochs, payload.dataset_folder, payload.img_size, payload.batch, payload.seed,
            payload.eval_dataset_folder, custom_classes=payload.custom_classes, detector_slug=payload.detector_slug,
        )
        if payload.custom_classes:
            # The lineage rule, at add time for the same reason as the
            # base-model check below: a queued entry can sit for hours,
            # and refusing it then means the queue skips it and moves on
            # with nothing on screen saying why.
            if payload.base_model_scratch:
                refusal = t2.scratch_refusal(payload.detector_slug, payload.architecture)
                if refusal:
                    raise ValueError(refusal)
            # Resolved at add time, not at run time: a queued entry can sit
            # for hours, and discovering then that its base model is
            # missing means the queue skips it and moves on. The return
            # value is discarded -- this is a validation call.
            t2.resolve_base_model_source(
                scratch=payload.base_model_scratch,
                scratch_architecture=payload.architecture,
                base_model_key=payload.base_model_key,
                base_model_folder=payload.base_model_folder,
                base_model_package_arn=payload.base_model_package_arn,
            )
        entry = {
            "id": uuid.uuid4().hex[:12],
            "generation": payload.generation,
            "size": payload.size,
            "epochs": epochs,
            "img_size": img_size,
            "batch": batch,
            "seed": seed,
            "dataset_folder": payload.dataset_folder,
            "eval_dataset_folder": payload.eval_dataset_folder,
            "base_model_folder": payload.base_model_folder,
            # Which pipeline this entry runs through when its turn comes.
            # Without it the runner sent every entry to the production
            # pipeline, silently ignoring the operator's choice.
            "custom_classes": payload.custom_classes,
            "base_model_package_arn": payload.base_model_package_arn,
            "base_model_key": payload.base_model_key,
            "base_model_scratch": payload.base_model_scratch,
            "architecture": payload.architecture,
            "detector_slug": payload.detector_slug,
            "patience": payload.patience,
            # Already typed by TrainingParams. Without it the entry ran on
            # Ultralytics' defaults while the card showed the operator's
            # numbers -- the queue replays `entry["hyperparams"]`.
            "hyperparams": payload.hyperparams,
            "notes": payload.notes[:2048],
            "added_at": datetime.now(timezone.utc).isoformat(),
        }
        # WHICH frozen definition this entry trains on, captured at add
        # time. An entry can sit in the queue for hours and the live slot
        # can be restored to something else in the meantime, so recording
        # it when the run starts would name whatever happened to be
        # current then. The chain's own entries carry this already; a
        # queue-add from the training card did not, so a run on a
        # deliberately chosen snapshot was indistinguishable afterwards
        # from one on the live corpus.
        _stamp_snapshot(entry)
        # And when no snapshot was chosen, freeze one now -- so "which
        # images was this trained on" has an answer for every run, not
        # only the ones the chain queued.
        _freeze_for_entry(entry)
        # A DATASET VERSION, resolved now -- the folder's data.yaml can be
        # rebuilt before this entry's turn comes, and the run must train on
        # what was queued. Refused here, at the button, if it cannot be
        # made: the alternative is the overwritable slot. The open
        # pipeline only; the production pipeline takes no yaml URI.
        if entry.get("custom_classes"):
            from .. import dataset_versions as dsv
            from .. import pull_control as pc

            version = dsv.for_run(home=pc.destination_bucket(), root=entry["dataset_folder"],
                                  eval_root=entry.get("eval_dataset_folder", ""),
                                  trigger="manual queue")
            entry.update(dsv.run_fields(version))
            dsv.mark(version["version_id"], state="queued", entry_id=entry["id"],
                     entry_template={k: v for k, v in entry.items()
                                     if k not in ("id", "added_at")})
        t2.queue_add(entry)
    except Exception as exc:
        # Logged as well as returned: uvicorn records the 400 but not the
        # reason, so a rejected queue-add left nothing on the server saying
        # which guard fired -- and the operator's screenshot of "failed"
        # rarely includes the message.
        print(
            f"queue/add rejected (dataset={payload.dataset_folder!r} "
            f"eval={payload.eval_dataset_folder!r} custom_classes={payload.custom_classes} "
            f"package={payload.base_model_package_arn!r} key={payload.base_model_key!r}): {exc}"
        )
        raise HTTPException(status_code=400, detail=f"Add to queue failed: {exc}")
    return {"message": f"Added to queue (YOLO{payload.generation}-{payload.size}, {epochs} epochs, {payload.dataset_folder})", "entry": entry}


@router.post("/queue/remove/{entry_id}")
def remove_from_queue(entry_id: str, session=Depends(require_session)) -> dict:
    if not t2.queue_remove(entry_id):
        raise HTTPException(status_code=404, detail="Queue entry not found (already started or removed)")
    return {"removed": True}


@router.post("/queue/start")
def start_queue(session=Depends(require_session)) -> dict:
    if not t2.queue_start():
        raise HTTPException(status_code=400, detail="Queue is empty -- add at least one run first.")
    return {"message": "Queue started"}


@router.post("/queue/stop")
def stop_queue(session=Depends(require_session)) -> dict:
    t2.queue_stop()
    return {"message": "Queue stopped -- any run already in progress will finish; remaining entries stay queued."}

