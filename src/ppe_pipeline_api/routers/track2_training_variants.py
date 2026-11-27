"""Authenticated saved-variant and training batch API. Creation explicitly starts paid background work."""
from uuid import UUID
from typing import Literal
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field, ConfigDict
from ..auth import require_session
from ..chain import training_variants as e

router=APIRouter()
class Variant(BaseModel):
    model_config=ConfigDict(extra='forbid')
    name:str=Field(min_length=1,max_length=100)
    overrides:dict=Field(default_factory=dict)
class Evaluation(BaseModel):
    model_config=ConfigDict(extra='forbid')
    imgsz:int=Field(default=640,ge=32,le=4096)
    confidence:float=Field(default=.25,gt=0,lt=1)
    ap_conf:float=Field(default=.001,gt=0,lt=1)
    nms_iou:float=Field(default=.7,gt=0,lt=1)
    max_det:int=Field(default=300,ge=1,le=3000)
class VariantBatch(BaseModel):
    model_config=ConfigDict(extra='forbid')
    id:UUID
    name:str=Field(min_length=1,max_length=150)
    detector_slug:str
    architecture:str
    baseline_arn:str
    dataset_version_id:str=""
    data_mode:Literal["saved","prepared"]="saved"
    shared:dict=Field(default_factory=dict)
    evaluation:Evaluation=Field(default_factory=Evaluation)
    variants:list[Variant]=Field(min_length=1,max_length=20)
class Retry(BaseModel):
    index:int|None=Field(default=None,ge=0)

class SavedVariant(BaseModel):
    model_config=ConfigDict(extra='forbid')
    id:UUID
    name:str=Field(min_length=1,max_length=100)
    settings:dict
    selected:bool=True
class Draft(BaseModel):
    batch_name:str=Field(default="",max_length=150)
    model_config=ConfigDict(extra='forbid')
    detector_slug:str
    architecture:str
    baseline_arn:str
    dataset_version_id:str=""
    data_mode:Literal["saved","prepared"]="saved"
    revision:int=Field(default=0,ge=0)
    variants:list[SavedVariant]=Field(max_length=20)
def call(fn):
    try:return fn()
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
@router.get('/drafts')
def drafts(session=Depends(require_session)):
    return {'drafts':e.all_records(e.DRAFT_KIND)}
@router.put('/drafts')
def save_draft(payload:Draft,session=Depends(require_session)):
    return call(lambda:e.save_draft(payload.model_dump(mode='json')))
@router.get('')
def listing(session=Depends(require_session)):
    from ..chain.preparation_jobs import pending_summaries
    return {'runs':e.list_summaries(), 'preparations':pending_summaries()}
@router.post('')
def create(payload:VariantBatch,session=Depends(require_session)):
    return call(lambda:e.create(payload.model_dump(mode='json')))
@router.post('/{ident}/retry')
def retry(ident:str,payload:Retry,session=Depends(require_session)):
    result=call(lambda:e.retry(ident,payload.index))
    if result and result.get('started') is False:
        raise HTTPException(409,'Variant worker is busy; retry shortly.')
    return result
@router.get('/{ident}/report')
def report(ident:str,session=Depends(require_session)):
    return call(lambda:e.public(e.read(ident)))


@router.delete('/{ident}')
def remove_pending(ident:str,session=Depends(require_session)):
    result=call(lambda:e.remove_pending(ident))
    if result.get('started') is False:
        raise HTTPException(409,'This experiment is starting or running and cannot be removed. Refresh its status.')
    return result
