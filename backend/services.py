from pathlib import Path
import json
import os
import re

from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document

from google import genai
from huggingface_hub import InferenceClient

from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

load_dotenv()

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "gemini"
).lower()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.0-flash"
)

HF_TOKEN = os.getenv(
    "HF_TOKEN"
)

HF_MODEL = os.getenv(
    "HF_MODEL",
    "Qwen/Qwen3-0.6B"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "BAAI/bge-base-en-v1.5"
)


# ============================================================
# LLM Clients
# ============================================================

gemini_client = None
hf_client = None


if LLM_PROVIDER == "gemini":

    if not GEMINI_API_KEY:
        raise ValueError(
            "GEMINI_API_KEY is missing."
        )

    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )

else:

    if not HF_TOKEN:
        raise ValueError(
            "HF_TOKEN is missing."
        )

    hf_client = InferenceClient(
        api_key=HF_TOKEN
    )


# ============================================================
# LLM Generation
# ============================================================

def generate_text(prompt: str) -> str:

    if LLM_PROVIDER == "gemini":

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt
        )

        return response.text.strip()

    response = hf_client.chat.completions.create(
        model=HF_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=1500
    )

    return response.choices[0].message.content.strip()


# ============================================================
# JSON Parser
# ============================================================

def parse_json(response: str):

    response = response.strip()

    if response.startswith("```"):

        response = response.replace(
            "```json",
            ""
        )

        response = response.replace(
            "```",
            ""
        )

        response = response.strip()

    try:

        return json.loads(response)

    except json.JSONDecodeError:

        start = response.find("{")
        end = response.rfind("}")

        if start == -1 or end == -1:
            raise ValueError(
                "LLM did not return valid JSON."
            )

        try:

            return json.loads(
                response[start:end + 1]
            )

        except json.JSONDecodeError:

            raise ValueError(
                "LLM returned invalid JSON."
            )


# ============================================================
# Resume Text Extraction
# ============================================================

def extract_resume_text(
    file_path: Path
) -> str:

    extension = file_path.suffix.lower()

    if extension == ".pdf":

        reader = PdfReader(
            str(file_path)
        )

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        resume_text = "\n".join(
            pages
        )

    elif extension == ".docx":

        document = Document(
            str(file_path)
        )

        paragraphs = [
            paragraph.text
            for paragraph in document.paragraphs
            if paragraph.text.strip()
        ]

        resume_text = "\n".join(
            paragraphs
        )

    else:

        raise ValueError(
            "Only PDF and DOCX files are supported."
        )

    if not resume_text.strip():

        raise ValueError(
            "No readable text was found in the resume."
        )

    return resume_text.strip()


# ============================================================
# Resume Analysis
# ============================================================

def analyze_resume(
    resume_text: str
):

    prompt = f"""
You are an expert Resume Analyzer.

Analyze the resume carefully.

Do NOT invent information.

Your main goal is to extract the candidate's skills
in a clean and structured way for job matching.

==================================================
SKILL EXTRACTION RULES
==================================================

1. PROGRAMMING LANGUAGES

Put actual programming languages in:

"programming_languages"

Examples:

Python
SQL
Java
C++
C#
JavaScript
TypeScript
R
Go
PHP

IMPORTANT:

If Python is explicitly mentioned anywhere in the resume,
you MUST return:

"Python"

inside "programming_languages".

If SQL is explicitly mentioned, return:

"SQL"

inside "programming_languages".

Do NOT hide programming languages inside another skill.

--------------------------------------------------

2. TECHNICAL SKILLS

Put technologies, frameworks, libraries, tools,
methods, and technical concepts in:

"technical_skills"

Examples:

Machine Learning
Deep Learning
TensorFlow
PyTorch
Keras
Scikit-Learn
Pandas
NumPy
Docker
FastAPI
Git
GitHub
Computer Vision
NLP
Transformers
MLflow
Feature Engineering

Each skill must be a separate item.

Bad:

"Machine Learning: Classification, Regression, Random Forest"

Good:

[
    "Machine Learning",
    "Classification",
    "Regression",
    "Random Forest"
]

Bad:

"Frameworks & Libraries: TensorFlow, PyTorch, Keras"

Good:

[
    "TensorFlow",
    "PyTorch",
    "Keras"
]

--------------------------------------------------

3. DO NOT MIX CATEGORIES

Programming languages must go into:

"programming_languages"

Frameworks, libraries, tools, technologies,
and technical concepts must go into:

"technical_skills"

For example:

Python -> programming_languages

SQL -> programming_languages

TensorFlow -> technical_skills

PyTorch -> technical_skills

FastAPI -> technical_skills

Docker -> technical_skills

Machine Learning -> technical_skills

--------------------------------------------------

4. DO NOT INFER SKILLS

Only include skills that are explicitly mentioned
or clearly demonstrated in the resume.

Do NOT assume:

TensorFlow -> Python

Django -> Python

React -> JavaScript

Do not infer a programming language from a framework.

--------------------------------------------------

5. INDIVIDUAL SKILLS

Every skill must be a separate string.

Do not create long category strings.

--------------------------------------------------

6. PRESERVE THE ACTUAL SKILL NAME

Use standard readable names.

Examples:

"machine-learning" -> "Machine Learning"

"fast api" -> "FastAPI"

"deep-learning" -> "Deep Learning"

==================================================
OUTPUT FORMAT
==================================================

Return ONLY valid JSON.

Return exactly this structure:

{{
    "summary": "",
    "programming_languages": [],
    "technical_skills": [],
    "soft_skills": [],
    "education": [],
    "experience": [],
    "strengths": [],
    "weaknesses": []
}}

Resume:

{resume_text}
"""

    response = generate_text(
        prompt
    )

    print("\n" + "=" * 70)
    print("RESUME ANALYSIS")
    print("=" * 70)
    print(response)

    result = parse_json(
        response
    )

    return result


# ============================================================
# Job Analysis
# ============================================================

def analyze_job(
    job_description: str,
    job_title: str = ""
):

    prompt = f"""
You are an expert Job Description Analyzer.

Analyze the following job description.

Extract the skills required for the position.

Required skills are skills explicitly required
by the employer.

Preferred skills are skills described as preferred,
nice-to-have, bonus, or similar.

Do NOT invent skills.

Return ONLY valid JSON.

Return exactly:

{{
    "required_skills": [],
    "preferred_skills": [],
    "experience_years": 0,
    "education": [],
    "seniority": "",
    "job_keywords": []
}}

Job Title:

{job_title}

Job Description:

{job_description}
"""

    response = generate_text(
        prompt
    )

    return parse_json(
        response
    )


# ============================================================
# AI Job Matching
# ============================================================

def normalize_skill_name(skill):

    normalized = str(skill).lower().strip()
    normalized = normalized.replace("&", " and ")
    normalized = normalized.replace("fast api", "fastapi")
    normalized = normalized.replace("machine-learning", "machine learning")
    normalized = normalized.replace("deep-learning", "deep learning")
    normalized = normalized.replace("machine_learning", "machine learning")
    normalized = normalized.replace("deep_learning", "deep learning")

    aliases = {
        "ml": "machine learning",
        "dl": "deep learning",
        "nlp": "natural language processing",
        "apis": "api"
    }

    normalized = aliases.get(normalized, normalized)
    normalized = re.sub(r"[^a-z0-9+#]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    return normalized


def skill_matches(required_skill, candidate_skill):

    required = normalize_skill_name(required_skill)
    candidate = normalize_skill_name(candidate_skill)

    if not required or not candidate:
        return False

    if required == candidate:
        return True

    if len(required) < 3 or len(candidate) < 3:
        return False

    return required in candidate or candidate in required


def calculate_skill_match(
    resume_skills,
    required_skills,
    preferred_skills
):

    matched_required = []
    missing_required = []

    for required_skill in required_skills:
        if any(
            skill_matches(required_skill, candidate_skill)
            for candidate_skill in resume_skills
        ):
            matched_required.append(required_skill)
        else:
            missing_required.append(required_skill)

    matched_preferred = [
        preferred_skill
        for preferred_skill in preferred_skills
        if any(
            skill_matches(preferred_skill, candidate_skill)
            for candidate_skill in resume_skills
        )
    ]

    required_coverage = (
        len(matched_required) / len(required_skills) * 100
        if required_skills
        else 0
    )

    preferred_coverage = (
        len(matched_preferred) / len(preferred_skills) * 100
        if preferred_skills
        else 0
    )

    if required_skills and preferred_skills:
        score = required_coverage * 0.9 + preferred_coverage * 0.1
    elif required_skills:
        score = required_coverage
    else:
        score = preferred_coverage

    if matched_required:
        explanation = (
            f"The candidate matches {len(matched_required)} of "
            f"{len(required_skills)} required skills."
        )
    else:
        explanation = "The required job skills are not present in the candidate's skills."

    if missing_required:
        explanation += (
            " Missing skills: "
            + ", ".join(missing_required)
            + "."
        )

    if matched_preferred:
        explanation += (
            " Preferred skills matched: "
            + ", ".join(matched_preferred)
            + "."
        )

    is_suitable = (
        not missing_required
        or (
            bool(required_skills)
            and len(matched_required) / len(required_skills) >= 0.6
        )
    )

    return {
        "match_score": round(score, 2),
        "is_suitable": is_suitable,
        "suitability": "Suitable" if is_suitable else "Needs development",
        "matched_skills": matched_required,
        "missing_skills": missing_required,
        "preferred_matched_skills": matched_preferred,
        "improvement_plan": [
            f"Build practical experience with {skill}."
            for skill in missing_required
        ],
        "explanation": explanation
    }

def analyze_job_match(
    resume_programming_languages,
    resume_technical_skills,
    required_programming_languages,
    required_technical_skills,
    preferred_programming_languages=None,
    preferred_technical_skills=None
):
    """
    Compare resume skills with job requirements.

    The LLM calculates:
    - Match score
    - Matched required skills
    - Missing required skills
    - Matched preferred skills
    - Explanation

    IMPORTANT:
    The original matching logic is preserved.
    """

    preferred_programming_languages = (
        preferred_programming_languages or []
    )

    preferred_technical_skills = (
        preferred_technical_skills or []
    )

    # ========================================================
    # COMBINE SKILLS
    # ========================================================

    # Resume skills
    programming_languages = (
        resume_programming_languages or []
    )

    technical_skills = (
        resume_technical_skills or []
    )

    all_resume_skills = (
        programming_languages
        + technical_skills
    )

    # Job required skills
    required_skills = (
        required_programming_languages
        + required_technical_skills
    )

    # Job preferred skills
    preferred_skills = (
        preferred_programming_languages
        + preferred_technical_skills
    )

    fallback_result = calculate_skill_match(
        resume_skills=all_resume_skills,
        required_skills=required_skills,
        preferred_skills=preferred_skills
    )

    prompt = f"""
You are the Job Suitability Agent.

Decide whether this candidate is suitable for the job.
Do not calculate or return a numeric score.
Use only the supplied resume and job skills.

Candidate skills:
{all_resume_skills}

Required job skills:
{required_skills}

Preferred job skills:
{preferred_skills}

Return ONLY valid JSON in this exact shape:
{{
  "is_suitable": true,
  "suitability": "Suitable",
  "matched_skills": [],
  "missing_skills": [],
  "improvement_plan": [],
  "explanation": ""
}}

The suitability value must be either "Suitable" or "Needs development".
The improvement_plan must contain practical actions for missing skills.
"""

    try:
        llm_result = parse_json(generate_text(prompt))

        suitability = llm_result.get(
            "suitability",
            "Suitable" if llm_result.get("is_suitable") else "Needs development"
        )

        if suitability not in ["Suitable", "Needs development"]:
            suitability = (
                "Suitable"
                if llm_result.get("is_suitable")
                else "Needs development"
            )

        is_suitable = suitability == "Suitable"

        improvement_plan = llm_result.get(
            "improvement_plan",
            fallback_result["improvement_plan"]
        )

        if isinstance(improvement_plan, str):
            improvement_plan = [improvement_plan]

        if not isinstance(improvement_plan, list):
            improvement_plan = fallback_result["improvement_plan"]

        return {
            "match_score": fallback_result["match_score"],
            "is_suitable": is_suitable,
            "suitability": suitability,
            "matched_skills": fallback_result["matched_skills"],
            "missing_skills": fallback_result["missing_skills"],
            "preferred_matched_skills": fallback_result[
                "preferred_matched_skills"
            ],
            "improvement_plan": improvement_plan,
            "explanation": llm_result.get(
                "explanation",
                fallback_result["explanation"]
            )
        }
    except Exception as error:
        print(f"Suitability Agent failed: {error}")
        return fallback_result

    # ========================================================
    # DEBUG INPUT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("                 JOB MATCH DEBUG")
    print("=" * 70)

    print("\n[1] RESUME PROGRAMMING LANGUAGES:")
    print(programming_languages)

    print("\n[2] RESUME TECHNICAL SKILLS:")
    print(technical_skills)

    print("\n[3] ALL RESUME SKILLS:")
    print(all_resume_skills)

    print("\n[4] REQUIRED PROGRAMMING LANGUAGES:")
    print(required_programming_languages)

    print("\n[5] REQUIRED TECHNICAL SKILLS:")
    print(required_technical_skills)

    print("\n[6] ALL REQUIRED SKILLS:")
    print(required_skills)

    print("\n[7] PREFERRED PROGRAMMING LANGUAGES:")
    print(preferred_programming_languages)

    print("\n[8] PREFERRED TECHNICAL SKILLS:")
    print(preferred_technical_skills)

    print("\n[9] ALL PREFERRED SKILLS:")
    print(preferred_skills)

    print("\n[10] COUNTS:")

    print(
        f"Resume programming languages: "
        f"{len(programming_languages)}"
    )

    print(
        f"Resume technical skills: "
        f"{len(technical_skills)}"
    )

    print(
        f"All resume skills: "
        f"{len(all_resume_skills)}"
    )

    print(
        f"Required skills: "
        f"{len(required_skills)}"
    )

    print(
        f"Preferred skills: "
        f"{len(preferred_skills)}"
    )

    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""
You are an expert AI Job Matching System.

Compare the candidate's resume skills with the
job requirements.

Do NOT invent skills.

Use ONLY the information provided below.

==================================================
CANDIDATE PROGRAMMING LANGUAGES
==================================================

{programming_languages}

==================================================
CANDIDATE TECHNICAL SKILLS
==================================================

{technical_skills}

==================================================
ALL CANDIDATE SKILLS
==================================================

{all_resume_skills}

==================================================
REQUIRED JOB PROGRAMMING LANGUAGES
==================================================

{required_programming_languages}

==================================================
REQUIRED JOB TECHNICAL SKILLS
==================================================

{required_technical_skills}

==================================================
ALL REQUIRED JOB SKILLS
==================================================

{required_skills}

==================================================
PREFERRED JOB PROGRAMMING LANGUAGES
==================================================

{preferred_programming_languages}

==================================================
PREFERRED JOB TECHNICAL SKILLS
==================================================

{preferred_technical_skills}

==================================================
ALL PREFERRED JOB SKILLS
==================================================

{preferred_skills}

==================================================
MATCHING RULES
==================================================

1. MATCHED REQUIRED SKILLS

Return required job skills that are present
in the candidate's skills.

2. MISSING REQUIRED SKILLS

Return required job skills that are not present
in the candidate's skills.

3. MATCHED PREFERRED SKILLS

Return preferred skills that are present
in the candidate's skills.

4. PROGRAMMING LANGUAGES

Programming languages must be matched normally.

For example:

Candidate:
Python

Job:
Python

Result:

Python -> matched

5. SKILL NORMALIZATION

Understand obvious variations.

Examples:

ML = Machine Learning

machine-learning = Machine Learning

machine_learning = Machine Learning

DL = Deep Learning

deep-learning = Deep Learning

deep_learning = Deep Learning

Fast API = FastAPI

6. DO NOT MAKE UNRELATED MATCHES

GitHub is NOT automatically Git.

Python is NOT automatically Django.

Machine Learning is NOT automatically Deep Learning.

TensorFlow is NOT automatically Python.

PyTorch is NOT automatically Python.

7. MATCH SCORE

Calculate a score from 0 to 100.

Required skills are the main factor.

Preferred skills are secondary.

The score must reflect the actual matching result.

The score MUST be consistent with:

matched_skills

missing_skills

required_skills

If 4 out of 7 required skills match,
the required skill coverage is approximately:

4 / 7 * 100 = 57.14%

Preferred skills can improve the score slightly.

If at least one required skill is matched,
DO NOT return a score of 0.

8. EXPLANATION

Give a short professional explanation.

The explanation must agree with:
- match_score
- matched_skills
- missing_skills
- preferred_matched_skills

==================================================
IMPORTANT
==================================================

Do not invent skills.

Do not add skills that are not present.

matched_skills must contain only required job skills.

missing_skills must contain only required job skills.

preferred_matched_skills must contain only preferred job skills.

match_score must be a number from 0 to 100.

Return ONLY valid JSON.

Do not use Markdown.

Do not add ```json.

Return exactly:

{{
    "match_score": 0,
    "matched_skills": [],
    "missing_skills": [],
    "preferred_matched_skills": [],
    "explanation": ""
}}
"""

    # ========================================================
    # CALL LLM
    # ========================================================

    response = generate_text(
        prompt
    )

    print("\n[11] RAW LLM RESPONSE:")
    print(response)

    # ========================================================
    # PARSE RESPONSE
    # ========================================================

    result = parse_json(
        response
    )

    print("\n[12] PARSED RESULT:")
    print(result)

    # ========================================================
    # SCORE
    # ========================================================

    try:

        score = float(
            result.get(
                "match_score",
                0
            )
        )

    except (
        TypeError,
        ValueError
    ):

        score = 0

    score = max(
        0,
        min(100, score)
    )

    # ========================================================
    # MATCHED SKILLS
    # ========================================================

    matched_skills = result.get(
        "matched_skills",
        []
    )

    if not isinstance(
        matched_skills,
        list
    ):

        matched_skills = []

    # ========================================================
    # MISSING SKILLS
    # ========================================================

    missing_skills = result.get(
        "missing_skills",
        []
    )

    if not isinstance(
        missing_skills,
        list
    ):

        missing_skills = []

    # ========================================================
    # PREFERRED MATCHED SKILLS
    # ========================================================

    preferred_matched_skills = result.get(
        "preferred_matched_skills",
        []
    )

    if not isinstance(
        preferred_matched_skills,
        list
    ):

        preferred_matched_skills = []

    # ========================================================
    # EXPLANATION
    # ========================================================

    explanation = result.get(
        "explanation",
        ""
    )

    if not isinstance(
        explanation,
        str
    ):

        explanation = ""

    # ========================================================
    # DEBUG SCORE
    # ========================================================

    required_count = len(
        required_skills
    )

    matched_count = len(
        matched_skills
    )

    coverage = (
        matched_count
        / required_count
        * 100
        if required_count
        else 0
    )

    print("\n[13] SCORE DEBUG:")

    print(
        f"Required skills : {required_count}"
    )

    print(
        f"Matched skills  : {matched_count}"
    )

    print(
        f"Missing skills  : {len(missing_skills)}"
    )

    print(
        f"Required coverage: {coverage:.2f}%"
    )

    print(
        f"LLM match score  : {score:.2f}%"
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    final_result = {
        "match_score": round(
            score,
            2
        ),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "preferred_matched_skills": preferred_matched_skills,
        "explanation": explanation
    }

    print("\n[14] FINAL MATCH RESULT:")
    print(final_result)

    print("\n" + "=" * 70)
    print("              END JOB MATCH DEBUG")
    print("=" * 70)
    print()

    return final_result


# ============================================================
# RAG Configuration
# ============================================================

BASE_DIR = Path(
    __file__
).resolve().parent

KNOWLEDGE_BASE_DIR = (
    BASE_DIR / "knowledge_base"
)

CHROMA_DIR = (
    BASE_DIR / "chroma_db"
)


# ============================================================
# ChromaDB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)

collection = chroma_client.get_or_create_collection(
    name="career_knowledge"
)


# ============================================================
# Embedding Model
# ============================================================

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)


# ============================================================
# RAG - Build Knowledge Base
# ============================================================

def initialize_knowledge_base():

    KNOWLEDGE_BASE_DIR.mkdir(
        exist_ok=True
    )

    documents = []
    metadatas = []
    ids = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    counter = 0

    for file_path in KNOWLEDGE_BASE_DIR.glob(
        "*.txt"
    ):

        text = file_path.read_text(
            encoding="utf-8"
        )

        chunks = splitter.split_text(
            text
        )

        for chunk in chunks:

            documents.append(
                chunk
            )

            metadatas.append({
                "source": file_path.name
            })

            ids.append(
                f"{file_path.stem}_{counter}"
            )

            counter += 1

    if not documents:
        return

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True
    )

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )


# ============================================================
# RAG - Retrieve Knowledge
# ============================================================

def retrieve_knowledge(
    query: str,
    top_k: int = 5
):

    if collection.count() == 0:
        initialize_knowledge_base()

    if collection.count() == 0:
        return []

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    )[0]

    result = collection.query(
        query_embeddings=[
            query_embedding.tolist()
        ],
        n_results=top_k
    )

    return result.get(
        "documents",
        [[]]
    )[0]


# ============================================================
# RAG - Career Advice
# ============================================================

def generate_rag_career_advice(
    resume_text,
    technical_skills,
    soft_skills,
    strengths,
    weaknesses,
    missing_skills
):

    query = f"""
Career advice for a candidate with:

Technical Skills:
{technical_skills}

Soft Skills:
{soft_skills}

Strengths:
{strengths}

Weaknesses:
{weaknesses}

Missing Skills:
{missing_skills}
"""

    knowledge = retrieve_knowledge(
        query,
        top_k=5
    )

    context = "\n\n".join(
        knowledge
    )

    prompt = f"""
You are a Career Advisor.

Use the following Knowledge Base information
to provide career advice.

Do not invent certifications, resources,
or roadmaps outside the provided knowledge.

Knowledge Base:

{context}

Candidate Resume:

{resume_text}

Technical Skills:

{technical_skills}

Soft Skills:

{soft_skills}

Strengths:

{strengths}

Weaknesses:

{weaknesses}

Missing Skills:

{missing_skills}

Return ONLY valid JSON:

{{
    "skills_to_learn": [],
    "resume_improvements": [],
    "certifications": [],
    "learning_roadmap": [],
    "career_advice": ""
}}
"""

    response = generate_text(
        prompt
    )

    return parse_json(
        response
    )


# ============================================================
# Career Advice
# ============================================================

def generate_career_advice(
    resume_text,
    technical_skills,
    soft_skills,
    strengths,
    weaknesses,
    missing_skills
):

    return generate_rag_career_advice(
        resume_text=resume_text,
        technical_skills=technical_skills,
        soft_skills=soft_skills,
        strengths=strengths,
        weaknesses=weaknesses,
        missing_skills=missing_skills
    )


# ============================================================
# AI AGENT 1 - Resume Analyzer
# ============================================================

class ResumeAnalyzerAgent:

    def run(
        self,
        resume_text: str
    ):

        return analyze_resume(
            resume_text
        )


# ============================================================
# AI AGENT 2 - Job Matching
# ============================================================

class JobMatchingAgent:

    def run(
        self,
        programming_languages,
        technical_skills,
        required_skills,
        preferred_skills=None
    ):

        return analyze_job_match(
            programming_languages=programming_languages,
            technical_skills=technical_skills,
            required_skills=required_skills,
            preferred_skills=preferred_skills
        )


# ============================================================
# AI AGENT 3 - Career Advisor
# ============================================================

class CareerAdvisorAgent:

    def run(
        self,
        resume_text,
        technical_skills,
        soft_skills,
        strengths,
        weaknesses,
        missing_skills
    ):

        return generate_rag_career_advice(
            resume_text=resume_text,
            technical_skills=technical_skills,
            soft_skills=soft_skills,
            strengths=strengths,
            weaknesses=weaknesses,
            missing_skills=missing_skills
        )


# ============================================================
# Agent Instances
# ============================================================

resume_analyzer_agent = ResumeAnalyzerAgent()

job_matching_agent = JobMatchingAgent()

career_advisor_agent = CareerAdvisorAgent()
