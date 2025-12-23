from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from bson import ObjectId
from pydantic_core import core_schema


class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type, handler
    ):
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.str_schema(),
            serialization=core_schema.plain_serializer_function_ser_schema(
                lambda x: str(x)
            ),
        )

    @classmethod
    def validate(cls, v):
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str):
            if not ObjectId.is_valid(v):
                raise ValueError("Invalid objectid")
            return ObjectId(v)
        raise ValueError("Invalid objectid")


class ResumeUpload(BaseModel):
    user_id: Optional[str] = None
    filename: str


class JobDescription(BaseModel):
    text: str
    user_id: Optional[str] = None


class GapItem(BaseModel):
    requirement: str
    status: str  # "missing" or "partial"
    evidence: Optional[str] = None


class FitScoreResponse(BaseModel):
    fit_score: float = Field(..., ge=0, le=100)
    gaps: List[GapItem]
    strengths: List[str]
    evidence: List[Dict[str, Any]]
    tailored_bullets: List[str]
    cover_letter_snippet: Optional[str] = None


class AnalysisResult(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    user_id: Optional[str] = None
    resume_text: str
    job_description: str
    fit_score: float
    gaps: List[GapItem]
    strengths: List[str]
    evidence: List[Dict[str, Any]]
    tailored_bullets: List[str]
    cover_letter_snippet: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True


