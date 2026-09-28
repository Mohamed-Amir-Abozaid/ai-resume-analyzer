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

def analyze_job_match(
    resume_programming_languages,
    resume_technical_skills,
    required_programming_languages,
    required_technical_skills,
    preferred_programming_languages=None,
    preferred_technical_skills=None
):
    """
    Use the LLM to semantically compare candidate skills
    with job requirements and return a structured result.
    """

    preferred_programming_languages = (
        preferred_programming_languages or []
    )

    preferred_technical_skills = (
        preferred_technical_skills or []
    )

    # --------------------------------------------------------
    # Combine candidate skills
    # --------------------------------------------------------

    candidate_skills = (
        (resume_programming_languages or [])
        + (resume_technical_skills or [])
    )

    # --------------------------------------------------------
    # Combine job skills
    # --------------------------------------------------------

    required_skills = (
        (required_programming_languages or [])
        + (required_technical_skills or [])
    )

    preferred_skills = (
        preferred_programming_languages
        + preferred_technical_skills
    )

    # --------------------------------------------------------
    # LLM Prompt
    # --------------------------------------------------------

    prompt = f"""
You are an expert AI Job Matching Agent.

Compare the candidate's skills with the job requirements.

Use ONLY the information provided below.
Do NOT invent skills.

Candidate skills:
{candidate_skills}

Required job skills:
{required_skills}

Preferred job skills:
{preferred_skills}

Your tasks:

1. Identify which required skills the candidate has.

2. Identify which required skills are missing.

3. Identify which preferred skills the candidate has.

4. Understand obvious skill variations and synonyms.

Examples:
- ML = Machine Learning
- DL = Deep Learning
- NLP = Natural Language Processing
- machine-learning = Machine Learning
- machine_learning = Machine Learning
- deep-learning = Deep Learning
- deep_learning = Deep Learning
- Fast API = FastAPI

5. Do NOT make automatic assumptions.

Examples:
- Python does NOT mean Django.
- Python does NOT mean FastAPI.
- GitHub does NOT automatically mean Git.
- Machine Learning does NOT automatically mean Deep Learning.
- TensorFlow does NOT automatically mean Python.

6. Calculate match_score from 0 to 100.

Use this principle:

Required skills are the main factor.
Preferred skills are a secondary factor.

Required skill coverage:

matched required skills / total required skills * 100

Preferred skills may increase the final score,
but they must have less influence than required skills.

The score must be consistent with the matched
and missing skills.

7. suitability:

Return "Suitable" when the candidate satisfies
most of the required skills.

Otherwise return "Needs development".

8. improvement_plan:

For every missing required skill, provide
one practical improvement action.

9. explanation:

Give a short explanation that is consistent with
the returned score and matching results.

Return ONLY valid JSON.

Return exactly this structure:

{{
    "match_score": 0,
    "is_suitable": false,
    "suitability": "Needs development",
    "matched_skills": [],
    "missing_skills": [],
    "preferred_matched_skills": [],
    "improvement_plan": [],
    "explanation": ""
}}
"""

    try:
        response = generate_text(prompt)

        result = parse_json(response)

        # ----------------------------------------------------
        # Validate basic structure
        # ----------------------------------------------------

        score = result.get("match_score", 0)

        try:
            score = float(score)
        except (TypeError, ValueError):
            score = 0

        score = max(0, min(100, score))

        matched_skills = result.get("matched_skills", [])
        if not isinstance(matched_skills, list):
            matched_skills = []

        missing_skills = result.get("missing_skills", [])
        if not isinstance(missing_skills, list):
            missing_skills = []

        preferred_matched_skills = result.get(
            "preferred_matched_skills",
            []
        )

        if not isinstance(preferred_matched_skills, list):
            preferred_matched_skills = []

        improvement_plan = result.get(
            "improvement_plan",
            []
        )

        if not isinstance(improvement_plan, list):
            improvement_plan = []

        explanation = result.get("explanation", "")

        if not isinstance(explanation, str):
            explanation = ""

        suitability = result.get(
            "suitability",
            "Suitable" if result.get("is_suitable") else "Needs development"
        )

        if suitability not in [
            "Suitable",
            "Needs development"
        ]:
            suitability = "Needs development"

        is_suitable = suitability == "Suitable"

        return {
            "match_score": round(score, 2),
            "is_suitable": is_suitable,
            "suitability": suitability,
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "preferred_matched_skills": preferred_matched_skills,
            "improvement_plan": improvement_plan,
            "explanation": explanation
        }

    except Exception as error:
        print(f"Job Matching Agent failed: {error}")

        return {
            "match_score": 0,
            "is_suitable": False,
            "suitability": "Needs development",
            "matched_skills": [],
            "missing_skills": required_skills,
            "preferred_matched_skills": [],
            "improvement_plan": [
                f"Build practical experience with {skill}."
                for skill in required_skills
            ],
            "explanation": (
                "The job matching agent could not complete "
                "the analysis."
            )
        }


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
