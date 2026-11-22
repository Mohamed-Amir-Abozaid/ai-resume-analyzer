from pathlib import Path
import json

from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form
)

from fastapi.middleware.cors import CORSMiddleware

from .database import (
    get_db_connection,
    init_coach_tables,
    add_missing_job_columns
)
from .models import RegisterRequest, LoginRequest, JobRequest
from .routes import router as coach_router

import bcrypt

from .services import (
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
# Career & Interview Coach
# ============================================================

init_coach_tables()
add_missing_job_columns()
app.include_router(coach_router)


# ============================================================
# Directories
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR.parent / "uploads"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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
    Safely convert JSON values into Python lists.
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

        parts = item.split(",")

        for part in parts:

            skill = part.strip()

            if skill and skill not in result:
                result.append(skill)

    return result


# ============================================================
# Classify Programming Languages
# ============================================================

def classify_programming_languages(skills):

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


# ============================================================
# Parse Job Skills
# ============================================================

def parse_job_skills(required_skills_text):

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

    if isinstance(required_skills_text, str):

        return (
            normalize_json_list(required_skills_text),
            []
        )

    return [], []


# ============================================================
# Parse Job Skill Categories
# ============================================================

def parse_job_skill_categories(required_skills_text):

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
# Authentication - Register
# ============================================================

@app.post("/register")
def register(user: RegisterRequest):

    connection = get_db_connection()

    hashed_password = bcrypt.hashpw(
        user.password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO users (
                name,
                email,
                password
            )
            VALUES (?, ?, ?)
            """,
            (
                user.full_name,
                user.email,
                hashed_password
            )
        )

        connection.commit()

        return {
            "message": "User registered successfully!"
        }

    except Exception as error:

        connection.rollback()

        print(f"Registration error: {error}")

        raise HTTPException(
            status_code=400,
            detail="Could not register user. The email may already be registered."
        )

    finally:

        connection.close()


# ============================================================
# Authentication - Login
# ============================================================

@app.post("/login")
def login(user: LoginRequest):

    connection = get_db_connection()

    try:

        cursor = connection.cursor()

        cursor.execute(
            """
            SELECT
                id,
                name,
                email,
                password
            FROM users
            WHERE email = ?
            """,
            (user.email,)
        )

        user_data = cursor.fetchone()

        if not user_data:

            raise HTTPException(
                status_code=401,
                detail="Invalid email or password."
            )

        password_is_valid = bcrypt.checkpw(
            user.password.encode("utf-8"),
            user_data["password"].encode("utf-8")
        )

        if not password_is_valid:

            raise HTTPException(
                status_code=401,
                detail="Invalid email or password."
            )

        return {
            "message": "User logged in successfully!",

            "user": {
                "id": user_data["id"],
                "full_name": user_data["name"],
                "email": user_data["email"]
            }
        }

    except HTTPException:
        raise

    except Exception as error:

        print(f"Login error: {error}")

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

    allowed_extensions = [".pdf", ".docx"]

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="File name is missing"
        )

    file_extension = Path(file.filename).suffix.lower()

    if file_extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are allowed"
        )

    file_content = await file.read()

    if not file_content:

        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty"
        )

    # Use a safe filename
    safe_filename = Path(file.filename).name

    file_path = UPLOAD_DIR / safe_filename

    with open(file_path, "wb") as output_file:
        output_file.write(file_content)

    try:

        # Extract resume text
        resume_text = extract_resume_text(file_path)

        if not resume_text or not resume_text.strip():

            raise HTTPException(
                status_code=400,
                detail="Could not extract text from resume"
            )

        # AI Resume Analysis
        analysis = analyze_resume(resume_text)

        connection = get_db_connection()

        try:

            cursor = connection.cursor()

            cursor.execute(
                """
                INSERT INTO resumes (
                    user_id,
                    filename,
                    resume_text,
                    summary
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    user_id,
                    safe_filename,
                    resume_text,
                    analysis.get("summary", "")
                )
            )

            connection.commit()

            resume_id = cursor.lastrowid

            return {
                "message": "Resume uploaded successfully",

                "resume_id": resume_id,

                "filename": safe_filename,

                "text_preview": resume_text[:200],

                "analysis": analysis
            }

        finally:

            connection.close()

    except Exception:

        if file_path.exists():
            file_path.unlink()

        raise


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

            raise HTTPException(
                status_code=404,
                detail="Resume not found"
            )

        return dict(resume)

    finally:

        connection.close()


# ============================================================
# Create Job
# ============================================================
@app.post("/jobs")
def create_job(job: JobRequest):

    try:
        job_analysis = analyze_job(
            job_description=job.description,
            job_title=job.title
        )

    except Exception as error:
        print(f"Job AI analysis failed: {error}")

        job_analysis = {
            "required_skills": [
                skill.strip()
                for skill in job.required_skills.split(",")
                if skill.strip()
            ],
            "preferred_skills": []
        }

    skills_data = json.dumps({
        "required": normalize_json_list(
            job_analysis.get("required_skills", [])
        ),
        "preferred": normalize_json_list(
            job_analysis.get("preferred_skills", [])
        )
    })

    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            INSERT INTO jobs (
                title,
                company,
                description,
                required_skills,
                location,
                salary,
                job_type
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job.title,
                job.company,
                job.description,
                skills_data,
                job.location,
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
            raise HTTPException(
                status_code=404,
                detail="Job not found"
            )

        job_data = dict(job)

        required, preferred = parse_job_skills(
            job_data.get("required_skills")
        )

        job_data["required_skills"] = required
        job_data["preferred_skills"] = preferred

        return job_data

    finally:
        connection.close()

# ============================================================
# Search Jobs
# ============================================================

@app.get("/jobs/search")
def search_jobs(
    title: str = "",
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

        if skill:

            query += """
                AND required_skills LIKE ?
            """

            parameters.append(
                f"%{skill}%"
            )

        query += """
            ORDER BY id DESC
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

        # Get resume
        cursor.execute(
            """
            SELECT
                id,
                resume_text
            FROM resumes
            WHERE id = ?
            """,
            (resume_id,)
        )

        resume = cursor.fetchone()

        if resume is None:

            raise HTTPException(
                status_code=404,
                detail="Resume not found"
            )

        if not resume["resume_text"]:

            raise HTTPException(
                status_code=400,
                detail="Resume text is empty"
            )

        # Get job
        cursor.execute(
            """
            SELECT
                id,
                title,
                description,
                required_skills
            FROM jobs
            WHERE id = ?
            """,
            (job_id,)
        )

        job = cursor.fetchone()

        if job is None:

            raise HTTPException(
                status_code=404,
                detail="Job not found"
            )

        # Analyze resume
        resume_analysis = analyze_resume(
            resume["resume_text"]
        )

        resume_programming_languages = normalize_json_list(
            resume_analysis.get(
                "programming_languages",
                []
            )
        )

        resume_technical_skills = normalize_json_list(
            resume_analysis.get(
                "technical_skills",
                []
            )
        )

        # Parse job skills
        job_categories = parse_job_skill_categories(
            job["required_skills"]
        )

        required_programming_languages = job_categories[
            "required_programming_languages"
        ]

        required_technical_skills = job_categories[
            "required_technical_skills"
        ]

        preferred_programming_languages = job_categories[
            "preferred_programming_languages"
        ]

        preferred_technical_skills = job_categories[
            "preferred_technical_skills"
        ]

        # Debug
        print("\n========== MATCH DEBUG ==========")

        print("RESUME PROGRAMMING LANGUAGES:")
        print(resume_programming_languages)

        print("RESUME TECHNICAL SKILLS:")
        print(resume_technical_skills)

        print("REQUIRED PROGRAMMING LANGUAGES:")
        print(required_programming_languages)

        print("REQUIRED TECHNICAL SKILLS:")
        print(required_technical_skills)

        print("PREFERRED PROGRAMMING LANGUAGES:")
        print(preferred_programming_languages)

        print("PREFERRED TECHNICAL SKILLS:")
        print(preferred_technical_skills)

        print("=================================\n")

        # AI Job Matching
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

        # Read AI result
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

        # Check existing match
        cursor.execute(
            """
            SELECT id
            FROM job_matches
            WHERE resume_id = ?
            AND job_id = ?
            """,
            (
                resume_id,
                job_id
            )
        )

        existing_match = cursor.fetchone()

        if existing_match:

            cursor.execute(
                """
                UPDATE job_matches
                SET
                    match_score = ?,
                    explanation = ?
                WHERE id = ?
                """,
                (
                    score,
                    explanation,
                    existing_match["id"]
                )
            )

        else:

            cursor.execute(
                """
                INSERT INTO job_matches (
                    resume_id,
                    job_id,
                    match_score,
                    explanation
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    resume_id,
                    job_id,
                    score,
                    explanation
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

            "improvement_plan":
                improvement_plan,

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

                jobs.title,
                jobs.company,
                jobs.description,
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

            required_skills, preferred_skills = (
                parse_job_skills(
                    match_data.get("required_skills")
                )
            )

            match_data["required_skills"] = required_skills

            match_data["preferred_skills"] = preferred_skills

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

        # Get resume
        cursor.execute(
            """
            SELECT resume_text
            FROM resumes
            WHERE id = ?
            """,
            (resume_id,)
        )

        resume = cursor.fetchone()

        if resume is None:

            raise HTTPException(
                status_code=404,
                detail="Resume not found"
            )

        resume_text = resume["resume_text"] or ""

        if not resume_text.strip():

            raise HTTPException(
                status_code=400,
                detail="Resume text is empty"
            )

        # Analyze resume
        analysis = analyze_resume(resume_text)

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

        soft_skills = normalize_json_list(
            analysis.get(
                "soft_skills",
                []
            )
        )

        strengths = normalize_json_list(
            analysis.get(
                "strengths",
                []
            )
        )

        weaknesses = normalize_json_list(
            analysis.get(
                "weaknesses",
                []
            )
        )

        # Generate Career Advice
        advice = generate_rag_career_advice(

            resume_text=resume_text,

            technical_skills=(
                programming_languages + technical_skills
            ),

            soft_skills=soft_skills,

            strengths=strengths,

            weaknesses=weaknesses,

            missing_skills=[]
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

            "soft_skills":
                soft_skills,

            "strengths":
                strengths,

            "weaknesses":
                weaknesses,

            "career_advice":
                advice_text,

            "skills_to_learn":
                skills_to_learn,

            "resume_improvements":
                resume_improvements,

            "certifications":
                certifications,

            "learning_roadmap":
                learning_roadmap,

            "missing_skills": []
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

        cursor.execute("SELECT * FROM jobs ORDER BY id DESC")

        jobs = cursor.fetchall()

        result = []

        for job in jobs:
            job_data = dict(job)

            required, preferred = parse_job_skills(
                job_data.get("required_skills")
            )

            categories = parse_job_skill_categories(
                job_data.get("required_skills")
            )

            job_data["required_skills"] = required
            job_data["preferred_skills"] = preferred

            job_data.update(categories)

            result.append(job_data)

        return {"jobs": result}

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

            raise HTTPException(
                status_code=404,
                detail="Job not found"
            )

        job_data = dict(job)

        required_skills, preferred_skills = (
            parse_job_skills(
                job_data.get("required_skills")
            )
        )

        categories = parse_job_skill_categories(
            job_data.get("required_skills")
        )

        job_data["required_skills"] = required_skills

        job_data["preferred_skills"] = preferred_skills

        job_data.update(categories)

        return job_data

    finally:

        connection.close()


# ============================================================
# Update Job
# ============================================================
@app.put("/jobs/{job_id}")
def update_job(job_id: int, job: JobRequest):

    try:
        job_analysis = analyze_job(
            job_description=job.description,
            job_title=job.title
        )

    except Exception as error:
        print(f"Job AI analysis failed: {error}")

        job_analysis = {
            "required_skills": [
                skill.strip()
                for skill in job.required_skills.split(",")
                if skill.strip()
            ],
            "preferred_skills": []
        }

    skills_data = json.dumps({
        "required": normalize_json_list(
            job_analysis.get("required_skills", [])
        ),
        "preferred": normalize_json_list(
            job_analysis.get("preferred_skills", [])
        )
    })

    connection = get_db_connection()

    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            UPDATE jobs
            SET
                title = ?,
                company = ?,
                description = ?,
                required_skills = ?,
                location = ?,
                salary = ?,
                job_type = ?
            WHERE id = ?
            """,
            (
                job.title,
                job.company,
                job.description,
                skills_data,
                job.location,
                job.salary,
                job.job_type,
                job_id
            )
        )

        if cursor.rowcount == 0:
            raise HTTPException(
                status_code=404,
                detail="Job not found"
            )

        connection.commit()

        return {
            "message": "Job updated successfully",
            "job_id": job_id
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
            SELECT id
            FROM jobs
            WHERE id = ?
            """,
            (job_id,)
        )

        if cursor.fetchone() is None:

            raise HTTPException(
                status_code=404,
                detail="Job not found"
            )

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

        return {
            "message": "Job deleted successfully",
            "job_id": job_id
        }

    finally:

        connection.close()


# ============================================================
# Get User Resumes
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
                filename,
                summary
            FROM resumes
            WHERE user_id = ?
            ORDER BY id DESC
            """,
            (user_id,)
        )

        resumes = cursor.fetchall()

        return {
            "resumes": [
                dict(resume)
                for resume in resumes
            ]
        }

    finally:

        connection.close()


# ============================================================
# Delete Resume
# ============================================================

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
            SELECT filename
            FROM resumes
            WHERE id = ?
            AND user_id = ?
            """,
            (
                resume_id,
                user_id
            )
        )

        resume = cursor.fetchone()

        if resume is None:

            raise HTTPException(
                status_code=404,
                detail="Resume not found"
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
            (
                resume_id,
                user_id
            )
        )

        connection.commit()

        # Delete uploaded file safely
        stored_file = (
            UPLOAD_DIR / Path(resume["filename"]).name
        ).resolve()

        if stored_file.parent == UPLOAD_DIR.resolve():

            if stored_file.exists():
                stored_file.unlink()

        return {
            "message": "Resume deleted successfully",
            "resume_id": resume_id
        }

    finally:

        connection.close()