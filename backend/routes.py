"""
Career & Interview Coach routes (HAI).

GET  /coach/greeting   Inform: AI disclosure, can / cannot do, onboarding samples
GET  /coach/rubric     Explain: rubric criteria and level descriptions
POST /coach/chat       Feedback with explanation, evidence, criterion, focus, limits
POST /coach/feedback   Enable feedback: thumbs up / down (anonymous)
POST /coach/challenge  Enable feedback: challenge weak or unsupported feedback
"""

import logging
import re

from fastapi import APIRouter, HTTPException

import coach
from database import get_db_connection
from models import (
    CoachChallengeRequest,
    CoachChatRequest,
    CoachChatResponse,
    CoachRatingRequest,
)

logger = logging.getLogger("coach")

router = APIRouter(prefix="/coach", tags=["Career & Interview Coach"])

_RESPONSE_ID = re.compile(r"[0-9a-f]{32}")
MAX_TOTAL_INPUT = 9000


def _check_response_id(response_id: str):
    if not _RESPONSE_ID.fullmatch(response_id):
        raise HTTPException(status_code=422, detail="Invalid response id.")


@router.get("/greeting")
def greeting(language: str = "en"):
    return coach.get_greeting(language)


@router.get("/rubric")
def rubric(language: str = "en"):
    return coach.get_rubric(language)


@router.post("/chat", response_model=CoachChatResponse)
def chat(request: CoachChatRequest):
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Please type a message first.")

    history = [
        {"role": t.role, "content": t.content.strip()}
        for t in request.history[-coach.MAX_HISTORY_TURNS:]
        if t.content.strip()
    ]

    total = len(message) + len(request.job_description) + len(request.interview_question) + sum(len(t["content"]) for t in history)
    if total > MAX_TOTAL_INPUT:
        raise HTTPException(status_code=422, detail="The text is too long. Please shorten it and try again.")

    try:
        return coach.run_coach(
            message=message,
            job_description=request.job_description.strip(),
            interview_question=request.interview_question.strip(),
            history=history,
            focus=request.focus,
            plain_language=request.plain_language,
            language=request.language,
        )
    except coach.CoachError as error:
        raise HTTPException(status_code=error.status_code, detail=error.message)


@router.post("/feedback")
def rate_response(payload: CoachRatingRequest):
    _check_response_id(payload.response_id)
    reason = coach.redact_pii(payload.reason.strip())[:500]

    connection = get_db_connection()
    try:
        connection.execute(
            "INSERT INTO coach_feedback (response_id, rating, reason, focus) VALUES (?, ?, ?, ?)",
            (payload.response_id, payload.rating, reason or None, payload.focus or None),
        )
        connection.commit()
    except Exception:
        logger.exception("Could not save rating")
        raise HTTPException(status_code=500, detail="Could not save your rating. Please try again.")
    finally:
        connection.close()

    return {"status": "saved", "message": "Thanks. Your rating was saved anonymously."}


@router.post("/challenge")
def challenge_response(payload: CoachChallengeRequest):
    _check_response_id(payload.response_id)
    reason = coach.redact_pii(payload.reason.strip())[:800]
    if len(reason) < 5:
        raise HTTPException(status_code=422, detail="Please explain what looks wrong in at least a few words.")

    connection = get_db_connection()
    try:
        cursor = connection.execute(
            "INSERT INTO coach_challenges (response_id, category, reason, focus) VALUES (?, ?, ?, ?)",
            (payload.response_id, payload.category, reason, payload.focus or None),
        )
        connection.commit()
        challenge_id = cursor.lastrowid
    except Exception:
        logger.exception("Could not save challenge")
        raise HTTPException(status_code=500, detail="Could not save your challenge. Please try again.")
    finally:
        connection.close()

    return {
        "status": "received",
        "challenge_id": challenge_id,
        "message": "Thanks. Your challenge was saved for the team to review. It does not change this answer automatically.",
    }
