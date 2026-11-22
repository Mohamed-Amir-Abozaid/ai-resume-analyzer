from typing import List, Literal, Optional

from pydantic import BaseModel, EmailStr, Field

class RegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class JobRequest(BaseModel):
    title: str
    company: str
    location: str
    description: str
    required_skills: str
    salary: str
    job_type: str


# ============================================================
# Career & Interview Coach (HAI)
# ============================================================

class CoachTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., max_length=2000)


class CoachChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    job_description: str = Field("", max_length=4000)
    interview_question: str = Field("", max_length=500)
    focus: Literal["clarity", "relevance", "evidence"] = "clarity"
    plain_language: bool = False
    language: Literal["auto", "en", "ar"] = "auto"
    history: List[CoachTurn] = []


class CoachChatResponse(BaseModel):
    response_id: str
    type: str
    answer: str
    explanation: str
    evidence: List[str]
    criterion: str
    focus: str
    level: Optional[str] = None
    limitations: str
    next_action: str
    warnings: List[str]
    plain_language: bool
    language: str
    ai_generated: bool
    evidence_verified: Optional[bool] = None


class CoachRatingRequest(BaseModel):
    response_id: str = Field(..., min_length=8, max_length=64)
    rating: Literal["up", "down"]
    reason: str = Field("", max_length=500)
    focus: str = Field("", max_length=20)


class CoachChallengeRequest(BaseModel):
    response_id: str = Field(..., min_length=8, max_length=64)
    category: Literal["unsupported", "incorrect", "unfair", "other"]
    reason: str = Field(..., min_length=5, max_length=800)
    focus: str = Field("", max_length=20)
