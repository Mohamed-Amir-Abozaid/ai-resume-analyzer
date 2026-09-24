from pydantic import BaseModel,EmailStr

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