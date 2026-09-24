from pathlib import Path
import json

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from database import get_db_connection
from models import RegisterRequest, LoginRequest, JobRequest

from services import (
    extract_resume_text,
    analyze_resume,
    analyze_job,
    analyze_job_match,
    generate_rag_career_advice
)


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="AI Resume Analyzer",
    description="An AI-powered resume analyzer that provides insights and suggestions for improving resumes.",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# ============================================================
# Upload Directory
# ============================================================

UPLOAD_DIR = Path("../uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


# ============================================================
# Programming Languages
# ============================================================

PROGRAMMING_LANGUAGES = {
    "python",
    "sql",
    "java",
    "c",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "go",
    "rust",
    "php",
    "ruby",
    "swift",
    "kotlin",
    "r",
    "matlab",
    "scala",
    "dart",
    "bash",
    "shell"
}


# ============================================================
# Helper Functions
# ============================================================

def normalize_json_list(value):
    """
    Safely convert database JSON fields into a Python list.
    """

    if not value:
        return []

    if isinstance(value, list):
        items = value

    elif isinstance(value, tuple):
        items = list(value)

    elif isinstance(value, str):

        try:
            parsed = json.loads(value)

            # Handle double encoded JSON
            if isinstance(parsed, str):

                try:
                    parsed = json.loads(parsed)
                except (json.JSONDecodeError, TypeError):
                    pass

            if isinstance(parsed, list):
                items = parsed

            else:
                items = [parsed]

        except (json.JSONDecodeError, TypeError):
            items = [value]

    else:
        items = [value]

    result = []

    for item in items:

        if not isinstance(item, str):
            continue

        # Handle comma-separated strings
        parts = item.split(",")

        for part in parts:

            skill = part.strip()

            if skill and skill not in result:
                result.append(skill)

    return result


def classify_programming_languages(skills):
    """
    Separate programming languages from technical skills.
    """

    programming_languages = []
    technical_skills = []

    for skill in normalize_json_list(skills):

        normalized = skill.strip().lower()

        if normalized in PROGRAMMING_LANGUAGES:

            if skill not in programming_languages:
                programming_languages.append(skill)

        else:

            if skill not in technical_skills:
                technical_skills.append(skill)

    return programming_languages, technical_skills


def parse_job_skills(required_skills_text):
    """
    Supports:

    {
        "required": [],
        "preferred": []
    }

    and the extended format:

    {
        "required": [],
        "preferred": [],
        "required_programming_languages": [],
        "required_technical_skills": [],
        "preferred_programming_languages": [],
        "preferred_technical_skills": []
    }

    Also supports old comma-separated strings.
    """

    if not required_skills_text:
        return [], []

    if isinstance(required_skills_text, dict):

        required = required_skills_text.get(
            "required",
            []
        )

        preferred = required_skills_text.get(
            "preferred",
            []
        )

        return (
            normalize_json_list(required),
            normalize_json_list(preferred)
        )

    try:

        data = json.loads(required_skills_text)

        # Handle double JSON encoding
        if isinstance(data, str):

            try:
                data = json.loads(data)
            except (json.JSONDecodeError, TypeError):
                pass

        if isinstance(data, dict):

            required = data.get(
                "required",
                []
            )

            preferred = data.get(
                "preferred",
                []
            )

            return (
                normalize_json_list(required),
                normalize_json_list(preferred)
            )

    except (json.JSONDecodeError, TypeError):
        pass

    # Old comma-separated format
    if isinstance(required_skills_text, str):

        return (
            normalize_json_list(
                required_skills_text
            ),
            []
        )

    return [], []


def parse_job_skill_categories(required_skills_text):
    """
    Return all job skill categories.

    Works with the new structured JSON and
    also supports old job records.
    """

    if not required_skills_text:
        return {
            "required_programming_languages": [],
            "required_technical_skills": [],
            "preferred_programming_languages": [],
            "preferred_technical_skills": []
        }

    data = required_skills_text

    if isinstance(data, str):

        try:
            data = json.loads(data)

            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except (json.JSONDecodeError, TypeError):
                    pass

        except (json.JSONDecodeError, TypeError):
            data = None

    if isinstance(data, dict):

        # New structured format
        if any(
            key in data
            for key in [
                "required_programming_languages",
                "required_technical_skills",
                "preferred_programming_languages",
                "preferred_technical_skills"
            ]
        ):

            return {
                "required_programming_languages":
                    normalize_json_list(
                        data.get(
                            "required_programming_languages",
                            []
                        )
                    ),

                "required_technical_skills":
                    normalize_json_list(
                        data.get(
                            "required_technical_skills",
                            []
                        )
                    ),

                "preferred_programming_languages":
                    normalize_json_list(
                        data.get(
                            "preferred_programming_languages",
                            []
                        )
                    ),

                "preferred_technical_skills":
                    normalize_json_list(
                        data.get(
                            "preferred_technical_skills",
                            []
                        )
                    )
            }

        # Old format
        required = normalize_json_list(
            data.get("required", [])
        )

        preferred = normalize_json_list(
            data.get("preferred", [])
        )

        required_languages, required_technical = (
            classify_programming_languages(required)
        )

        preferred_languages, preferred_technical = (
            classify_programming_languages(preferred)
        )

        return {
            "required_programming_languages":
                required_languages,

            "required_technical_skills":
                required_technical,

            "preferred_programming_languages":
                preferred_languages,

            "preferred_technical_skills":
                preferred_technical
        }

    # Old comma-separated format
    skills = normalize_json_list(
        required_skills_text
    )

    languages, technical = (
        classify_programming_languages(skills)
    )

    return {
        "required_programming_languages": languages,
        "required_technical_skills": technical,
        "preferred_programming_languages": [],
        "preferred_technical_skills": []
    }


# ============================================================
# Home
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Welcome to the AI Resume Analyzer API!"
    }


# ============================================================
# Authentication
# ============================================================

@app.post("/register")
def register(user: RegisterRequest):

    connection = get_db_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO users (
                full_name,
                email,
                password
            )
            VALUES (?, ?, ?)
            """,
            (
                user.full_name,
                user.email,
                user.password
            )
        )

        connection.commit()

        return {
            "message": "User registered successfully!"
        }

    except Exception:

        connection.rollback()

        raise HTTPException(
            status_code=400,
            detail="Could not register user. The email may already be registered."
        )

    finally:

        connection.close()


@app.post("/login")
def login(user: LoginRequest):

    connection = get_db_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                full_name,
                email
            FROM users
            WHERE email = ?
            AND password = ?
            """,
            (
                user.email,
                user.password
            )
        )

        user_data = cursor.fetchone()

        if user_data:

            return {
                "message": "User logged in successfully!",
                "user": {
                    "id": user_data["id"],
                    "full_name": user_data["full_name"],
                    "email": user_data["email"]
                }
            }

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )

    except HTTPException:

        raise

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="Error occurred while logging in."
        )

    finally:

        connection.close()


# ============================================================
# Resume Upload + AI Analysis
# ============================================================

@app.post("/resume/upload")
async def upload_resume(
    user_id: int = Form(...),
    file: UploadFile = File(...)
):

    allowed_extensions = [
        ".pdf",
        ".docx"
    ]

    if not file.filename:

        return {
            "error": "File name is missing"
        }

    file_extension = Path(
        file.filename
    ).suffix.lower()

    if file_extension not in allowed_extensions:

        return {
            "error": "Only PDF and DOCX files are allowed"
        }

    file_path = UPLOAD_DIR / file.filename

    file_content = await file.read()

    with open(file_path, "wb") as output_file:
        output_file.write(file_content)

    # Extract resume text
    resume_text = extract_resume_text(
        file_path
    )

    # AI Resume Analysis
    analysis = analyze_resume(
        resume_text
    )

    # Make sure lists exist
    programming_languages = normalize_json_list(
        analysis.get(
            "programming_languages",
            []
        )
    )

    technical_skills = normalize_json_list(
        analysis.get(
            "technical_skills",
            []
        )
    )

    connection = get_db_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO resumes (
                user_id,
                file_name,
                file_path,
                extracted_text,
                summary,
                programming_languages,
                technical_skills,
                soft_skills,
                education,
                experience,
                strengths,
                weaknesses
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                user_id,
                file.filename,
                str(file_path),
                resume_text,

                analysis.get(
                    "summary",
                    ""
                ),

                json.dumps(
                    programming_languages
                ),

                json.dumps(
                    technical_skills
                ),

                json.dumps(
                    analysis.get(
                        "soft_skills",
                        []
                    )
                ),

                json.dumps(
                    analysis.get(
                        "education",
                        []
                    )
                ),

                json.dumps(
                    analysis.get(
                        "experience",
                        []
                    )
                ),

                json.dumps(
                    analysis.get(
                        "strengths",
                        []
                    )
                ),

                json.dumps(
                    analysis.get(
                        "weaknesses",
                        []
                    )
                )
            )
        )

        connection.commit()

        resume_id = cursor.lastrowid

        return {
            "message": "Resume uploaded successfully",
            "resume_id": resume_id,
            "file_name": file.filename,
            "text_preview": resume_text[:200],
            "analysis": analysis
        }

    finally:

        connection.close()


# ============================================================
# Get Resume
# ============================================================

@app.get("/resume/{resume_id}")
def get_resume(resume_id: int):

    connection = get_db_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM resumes
            WHERE id = ?
            """,
            (resume_id,)
        )

        resume = cursor.fetchone()

        if resume is None:

            return {
                "error": "Resume not found"
            }

        resume_data = dict(resume)

        resume_data["programming_languages"] = (
            normalize_json_list(
                resume_data.get(
                    "programming_languages"
                )
            )
        )

        resume_data["technical_skills"] = (
            normalize_json_list(
                resume_data.get(
                    "technical_skills"
                )
            )
        )

        resume_data["soft_skills"] = (
            normalize_json_list(
                resume_data.get(
                    "soft_skills"
                )
            )
        )

        resume_data["education"] = (
            normalize_json_list(
                resume_data.get(
                    "education"
                )
            )
        )

        resume_data["experience"] = (
            normalize_json_list(
                resume_data.get(
                    "experience"
                )
            )
        )

        resume_data["strengths"] = (
            normalize_json_list(
                resume_data.get(
                    "strengths"
                )
            )
        )

        resume_data["weaknesses"] = (
            normalize_json_list(
                resume_data.get(
                    "weaknesses"
                )
            )
        )

        return resume_data

    finally:

        connection.close()


# ============================================================
# Create Job
# ============================================================

@app.post("/jobs")
def create_job(job: JobRequest):

    # AI analyzes the job
    try:
        job_analysis = analyze_job(
            job_description=job.description,
            job_title=job.title
        )
    except Exception as error:
        print(f"Job AI analysis failed: {error}")

        manual_required_skills = [
            skill.strip()
            for skill in job.required_skills.split(",")
            if skill.strip()
        ]

        job_analysis = {
            "required_skills": manual_required_skills,
            "preferred_skills": [],
            "required_programming_languages": [],
            "required_technical_skills": manual_required_skills,
            "preferred_programming_languages": [],
            "preferred_technical_skills": [],
            "experience_years": 0,
            "education": [],
            "seniority": "",
            "job_keywords": []
        }

    required_skills = normalize_json_list(
        job_analysis.get(
            "required_skills",
            []
        )
    )

    preferred_skills = normalize_json_list(
        job_analysis.get(
            "preferred_skills",
            []
        )
    )

    required_programming_languages = (
        normalize_json_list(
            job_analysis.get(
                "required_programming_languages",
                []
            )
        )
    )

    required_technical_skills = (
        normalize_json_list(
            job_analysis.get(
                "required_technical_skills",
                []
            )
        )
    )

    preferred_programming_languages = (
        normalize_json_list(
            job_analysis.get(
                "preferred_programming_languages",
                []
            )
        )
    )

    preferred_technical_skills = (
        normalize_json_list(
            job_analysis.get(
                "preferred_technical_skills",
                []
            )
        )
    )

    # Store complete AI analysis
    skills_data = json.dumps(
        {
            "required": required_skills,
            "preferred": preferred_skills,

            "required_programming_languages":
                required_programming_languages,

            "required_technical_skills":
                required_technical_skills,

            "preferred_programming_languages":
                preferred_programming_languages,

            "preferred_technical_skills":
                preferred_technical_skills
        }
    )

    connection = get_db_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO jobs (
                title,
                company,
                location,
                description,
                required_skills,
                salary,
                job_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.title,
                job.company,
                job.location,
                job.description,
                skills_data,
                job.salary,
                job.job_type
            )
        )

        connection.commit()

        return {
            "message": "Job added successfully",
            "job_id": cursor.lastrowid,
            "analysis": job_analysis
        }

    finally:

        connection.close()


# ============================================================
# Get All Jobs
# ============================================================

@app.get("/jobs")
def get_jobs():

    connection = get_db_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM jobs
            ORDER BY created_at DESC
            """
        )

        jobs = cursor.fetchall()

        result = []

        for job in jobs:

            job_data = dict(job)

            categories = parse_job_skill_categories(
                job_data.get(
                    "required_skills"
                )
            )

            job_data["required_skills"] = (
                normalize_json_list(
                    parse_job_skills(
                        job_data.get(
                            "required_skills"
                        )
                    )[0]
                )
            )

            job_data["preferred_skills"] = (
                normalize_json_list(
                    parse_job_skills(
                        job_data.get(
                            "required_skills"
                        )
                    )[1]
                )
            )

            job_data.update(categories)

            result.append(job_data)

        return {
            "jobs": result
        }

    finally:

        connection.close()


# ============================================================
# Search Jobs
#
# IMPORTANT:
# This endpoint MUST come before /jobs/{job_id}
# ============================================================

@app.get("/jobs/search")
def search_jobs(
    title: str = "",
    location: str = "",
    skill: str = ""
):

    connection = get_db_connection()

    try:

        cursor = connection.cursor()

        query = """
            SELECT *
            FROM jobs
            WHERE 1 = 1
        """

        parameters = []

        if title:

            query += """
                AND title LIKE ?
            """

            parameters.append(
                f"%{title}%"
            )

        if location:

            query += """
                AND location LIKE ?
            """

            parameters.append(
                f"%{location}%"
            )

        if skill:

            query += """
                AND required_skills LIKE ?
            """

            parameters.append(
                f"%{skill}%"
            )

        query += """
            ORDER BY created_at DESC
        """

        cursor.execute(
            query,
            parameters
        )

        jobs = cursor.fetchall()

        return {
            "jobs": [
                dict(job)
                for job in jobs
            ]
        }

    finally:

        connection.close()


# ============================================================
# Match Resume With Job
# ============================================================

@app.post("/jobs/{job_id}/match")
def match_resume_with_job(
    job_id: int,
    resume_id: int
):

    connection = get_db_connection()

    try:

        cursor = connection.cursor()

        # ----------------------------------------------------
        # Get resume
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                programming_languages,
                technical_skills
            FROM resumes
            WHERE id = ?
            """,
            (resume_id,)
        )

        resume = cursor.fetchone()

        if resume is None:

            return {
                "error": "Resume not found"
            }

        # ----------------------------------------------------
        # Get job
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                title,
                required_skills
            FROM jobs
            WHERE id = ?
            """,
            (job_id,)
        )

        job = cursor.fetchone()

        if job is None:

            return {
                "error": "Job not found"
            }

        # ----------------------------------------------------
        # Parse resume skills
        # ----------------------------------------------------

        resume_programming_languages = (
            normalize_json_list(
                resume["programming_languages"]
            )
        )

        resume_technical_skills = (
            normalize_json_list(
                resume["technical_skills"]
            )
        )

        # ----------------------------------------------------
        # Parse job skills
        # ----------------------------------------------------

        job_categories = parse_job_skill_categories(
            job["required_skills"]
        )

        required_programming_languages = (
            job_categories[
                "required_programming_languages"
            ]
        )

        required_technical_skills = (
            job_categories[
                "required_technical_skills"
            ]
        )

        preferred_programming_languages = (
            job_categories[
                "preferred_programming_languages"
            ]
        )

        preferred_technical_skills = (
            job_categories[
                "preferred_technical_skills"
            ]
        )

        # ----------------------------------------------------
        # Debug
        # ----------------------------------------------------

        print("\n========== MATCH DEBUG ==========")

        print("\nRESUME PROGRAMMING LANGUAGES:")
        print(
            resume_programming_languages
        )

        print("\nRESUME TECHNICAL SKILLS:")
        print(
            resume_technical_skills
        )

        print("\nREQUIRED PROGRAMMING LANGUAGES:")
        print(
            required_programming_languages
        )

        print("\nREQUIRED TECHNICAL SKILLS:")
        print(
            required_technical_skills
        )

        print("\nPREFERRED PROGRAMMING LANGUAGES:")
        print(
            preferred_programming_languages
        )

        print("\nPREFERRED TECHNICAL SKILLS:")
        print(
            preferred_technical_skills
        )

        print("\n=================================\n")

        # ----------------------------------------------------
        # AI Job Matching
        #
        # LLM decides suitability and recommends improvements:
        # - matched skills
        # - missing skills
        # - improvement plan
        # - explanation
        # ----------------------------------------------------

        match_result = analyze_job_match(
            resume_programming_languages=
                resume_programming_languages,

            resume_technical_skills=
                resume_technical_skills,

            required_programming_languages=
                required_programming_languages,

            required_technical_skills=
                required_technical_skills,

            preferred_programming_languages=
                preferred_programming_languages,

            preferred_technical_skills=
                preferred_technical_skills
        )

        # ----------------------------------------------------
        # Read LLM result
        # ----------------------------------------------------

        is_suitable = bool(
            match_result.get(
                "is_suitable",
                False
            )
        )

        score = match_result.get(
            "match_score",
            0
        )

        suitability = match_result.get(
            "suitability",
            "Suitable" if is_suitable else "Needs development"
        )

        matched = match_result.get(
            "matched_skills",
            []
        )

        missing = match_result.get(
            "missing_skills",
            []
        )

        matched_preferred = match_result.get(
            "preferred_matched_skills",
            []
        )

        improvement_plan = match_result.get(
            "improvement_plan",
            []
        )

        explanation = match_result.get(
            "explanation",
            ""
        )

        # ----------------------------------------------------
        # Save match
        # ----------------------------------------------------

        cursor.execute(
            """
            INSERT INTO job_matches (
                resume_id,
                job_id,
                match_score,
                explanation,
                matched_skills,
                missing_skills,
                preferred_matched_skills
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)

            ON CONFLICT(resume_id, job_id)
            DO UPDATE SET
                match_score =
                    excluded.match_score,

                explanation =
                    excluded.explanation,

                matched_skills =
                    excluded.matched_skills,

                missing_skills =
                    excluded.missing_skills,

                preferred_matched_skills =
                    excluded.preferred_matched_skills
            """,
            (
                resume_id,
                job_id,
                0,
                explanation,
                json.dumps(matched),
                json.dumps(missing),
                json.dumps(matched_preferred)
            )
        )

        connection.commit()

        return {
            "resume_id": resume_id,
            "job_id": job_id,
            "job_title": job["title"],
            "match_score": score,
            "is_suitable": is_suitable,
            "suitability": suitability,
            "matched_skills": matched,
            "missing_skills": missing,
            "preferred_matched_skills":
                matched_preferred,
            "improvement_plan": improvement_plan,
            "explanation": explanation
        }

    finally:

        connection.close()


# ============================================================
# Get Resume Matches
# ============================================================

@app.get("/resume/{resume_id}/matches")
def get_resume_matches(resume_id: int):

    connection = get_db_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                job_matches.id,
                job_matches.resume_id,
                job_matches.job_id,
                job_matches.match_score,
                job_matches.explanation,
                job_matches.matched_skills,
                job_matches.missing_skills,
                job_matches.preferred_matched_skills,

                jobs.title,
                jobs.company,
                jobs.location,
                jobs.required_skills

            FROM job_matches

            JOIN jobs
                ON job_matches.job_id = jobs.id

            WHERE job_matches.resume_id = ?

            ORDER BY job_matches.match_score DESC
            """,
            (resume_id,)
        )

        matches = cursor.fetchall()

        result = []

        for match in matches:

            match_data = dict(match)

            match_data["matched_skills"] = (
                normalize_json_list(
                    match_data[
                        "matched_skills"
                    ]
                )
            )

            match_data["missing_skills"] = (
                normalize_json_list(
                    match_data[
                        "missing_skills"
                    ]
                )
            )

            match_data[
                "preferred_matched_skills"
            ] = normalize_json_list(
                match_data.get(
                    "preferred_matched_skills"
                )
            )

            result.append(match_data)

        return {
            "resume_id": resume_id,
            "matches": result
        }

    finally:

        connection.close()


# ============================================================
# Career Advisor
# ============================================================

@app.post("/resume/{resume_id}/career-advice")
def get_career_advice(resume_id: int):

    connection = get_db_connection()

    try:

        cursor = connection.cursor()

        # ----------------------------------------------------
        # Get resume information
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT
                extracted_text,
                programming_languages,
                technical_skills,
                soft_skills,
                strengths,
                weaknesses
            FROM resumes
            WHERE id = ?
            """,
            (resume_id,)
        )

        resume = cursor.fetchone()

        if resume is None:

            return {
                "error": "Resume not found"
            }

        programming_languages = (
            normalize_json_list(
                resume[
                    "programming_languages"
                ]
            )
        )

        technical_skills = (
            normalize_json_list(
                resume[
                    "technical_skills"
                ]
            )
        )

        soft_skills = (
            normalize_json_list(
                resume[
                    "soft_skills"
                ]
            )
        )

        strengths = (
            normalize_json_list(
                resume[
                    "strengths"
                ]
            )
        )

        weaknesses = (
            normalize_json_list(
                resume[
                    "weaknesses"
                ]
            )
        )

        # ----------------------------------------------------
        # Collect missing skills from matches
        # ----------------------------------------------------

        cursor.execute(
            """
            SELECT missing_skills
            FROM job_matches
            WHERE resume_id = ?
            """,
            (resume_id,)
        )

        matches = cursor.fetchall()

        missing_skills = []

        for match in matches:

            skills = normalize_json_list(
                match["missing_skills"]
            )

            missing_skills.extend(
                skills
            )

        missing_skills = sorted(
            set(missing_skills)
        )

        # ----------------------------------------------------
        # Include programming languages
        # in career advice context
        # ----------------------------------------------------

        all_technical_skills = (
            programming_languages
            + technical_skills
        )

        # ----------------------------------------------------
        # Generate RAG Career Advice
        # ----------------------------------------------------

        advice = generate_rag_career_advice(
            resume_text=
                resume["extracted_text"] or "",

            technical_skills=
                all_technical_skills,

            soft_skills=
                soft_skills,

            strengths=
                strengths,

            weaknesses=
                weaknesses,

            missing_skills=
                missing_skills
        )

        if isinstance(advice, dict):
            advice_text = advice.get(
                "career_advice",
                ""
            )

            skills_to_learn = advice.get(
                "skills_to_learn",
                []
            )

            resume_improvements = advice.get(
                "resume_improvements",
                []
            )

            certifications = advice.get(
                "certifications",
                []
            )

            learning_roadmap = advice.get(
                "learning_roadmap",
                []
            )
        else:
            advice_text = str(advice)
            skills_to_learn = []
            resume_improvements = []
            certifications = []
            learning_roadmap = []

        return {
            "resume_id": resume_id,
            "programming_languages":
                programming_languages,
            "technical_skills":
                technical_skills,
            "career_advice": advice_text,
            "skills_to_learn": skills_to_learn,
            "resume_improvements": resume_improvements,
            "certifications": certifications,
            "learning_roadmap": learning_roadmap,
            "missing_skills":
                missing_skills
        }

    finally:

        connection.close()


# ============================================================
# Get Single Job
# ============================================================

@app.get("/jobs/{job_id}")
def get_job(job_id: int):

    connection = get_db_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT *
            FROM jobs
            WHERE id = ?
            """,
            (job_id,)
        )

        job = cursor.fetchone()

        if job is None:

            return {
                "error": "Job not found"
            }

        job_data = dict(job)

        required_skills, preferred_skills = (
            parse_job_skills(
                job_data.get(
                    "required_skills"
                )
            )
        )

        categories = (
            parse_job_skill_categories(
                job_data.get(
                    "required_skills"
                )
            )
        )

        job_data["required_skills"] = (
            required_skills
        )

        job_data["preferred_skills"] = (
            preferred_skills
        )

        job_data.update(
            categories
        )

        return job_data

    finally:

        connection.close()


# ============================================================
# Update Job
# ============================================================

@app.put("/jobs/{job_id}")
def update_job(
    job_id: int,
    job: JobRequest
):

    # Analyze updated job
    try:
        job_analysis = analyze_job(
            job_description=job.description,
            job_title=job.title
        )
    except Exception as error:
        print(f"Job AI analysis failed during update: {error}")

        manual_required_skills = [
            skill.strip()
            for skill in job.required_skills.split(",")
            if skill.strip()
        ]

        job_analysis = {
            "required_skills": manual_required_skills,
            "preferred_skills": [],
            "required_programming_languages": [],
            "required_technical_skills": manual_required_skills,
            "preferred_programming_languages": [],
            "preferred_technical_skills": []
        }

    required_skills = normalize_json_list(
        job_analysis.get(
            "required_skills",
            []
        )
    )

    preferred_skills = normalize_json_list(
        job_analysis.get(
            "preferred_skills",
            []
        )
    )

    required_programming_languages = (
        normalize_json_list(
            job_analysis.get(
                "required_programming_languages",
                []
            )
        )
    )

    required_technical_skills = (
        normalize_json_list(
            job_analysis.get(
                "required_technical_skills",
                []
            )
        )
    )

    preferred_programming_languages = (
        normalize_json_list(
            job_analysis.get(
                "preferred_programming_languages",
                []
            )
        )
    )

    preferred_technical_skills = (
        normalize_json_list(
            job_analysis.get(
                "preferred_technical_skills",
                []
            )
        )
    )

    skills_data = json.dumps(
        {
            "required": required_skills,
            "preferred": preferred_skills,

            "required_programming_languages":
                required_programming_languages,

            "required_technical_skills":
                required_technical_skills,

            "preferred_programming_languages":
                preferred_programming_languages,

            "preferred_technical_skills":
                preferred_technical_skills
        }
    )

    connection = get_db_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            UPDATE jobs
            SET
                title = ?,
                company = ?,
                location = ?,
                description = ?,
                required_skills = ?,
                salary = ?,
                job_type = ?

            WHERE id = ?
            """,
            (
                job.title,
                job.company,
                job.location,
                job.description,
                skills_data,
                job.salary,
                job.job_type,
                job_id
            )
        )

        connection.commit()

        if cursor.rowcount == 0:

            return {
                "error": "Job not found"
            }

        return {
            "message": "Job updated successfully",
            "job_id": job_id,
            "analysis": job_analysis
        }

    finally:

        connection.close()


# ============================================================
# Delete Job
# ============================================================

@app.delete("/jobs/{job_id}")
def delete_job(job_id: int):

    connection = get_db_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            DELETE FROM job_matches
            WHERE job_id = ?
            """,
            (job_id,)
        )

        cursor.execute(
            """
            DELETE FROM jobs
            WHERE id = ?
            """,
            (job_id,)
        )

        connection.commit()

        if cursor.rowcount == 0:

            return {
                "error": "Job not found"
            }

        return {
            "message": "Job deleted successfully",
            "job_id": job_id
        }

    finally:

        connection.close()


# ============================================================
# User Resumes
# ============================================================

@app.get("/resumes")
def get_user_resumes(user_id: int):

    connection = get_db_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                user_id,
                file_name,
                summary,
                created_at
            FROM resumes
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            """,
            (user_id,)
        )

        return {
            "resumes": [
                dict(resume)
                for resume in cursor.fetchall()
            ]
        }

    finally:

        connection.close()


@app.delete("/resume/{resume_id}")
def delete_resume(
    resume_id: int,
    user_id: int
):

    connection = get_db_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT file_path
            FROM resumes
            WHERE id = ?
            AND user_id = ?
            """,
            (resume_id, user_id)
        )

        resume = cursor.fetchone()

        if resume is None:
            raise HTTPException(
                status_code=404,
                detail="Resume not found for this account."
            )

        cursor.execute(
            """
            DELETE FROM job_matches
            WHERE resume_id = ?
            """,
            (resume_id,)
        )

        cursor.execute(
            """
            DELETE FROM resumes
            WHERE id = ?
            AND user_id = ?
            """,
            (resume_id, user_id)
        )

        connection.commit()

        file_path = resume["file_path"]

        if file_path:
            stored_file = Path(file_path)

            if stored_file.exists():
                stored_file.unlink()

        return {
            "message": "Resume deleted successfully.",
            "resume_id": resume_id
        }

    finally:

        connection.close()