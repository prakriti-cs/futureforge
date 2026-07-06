from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
from utils.llm import extract_resume_info

def skill_gap(resume_text:str, required_skills:list[str]) -> dict:
    result = extract_resume_info(resume_text)
    resume_data=result.model_dump()

    resume_skills=resume_data.get("skills",[])
    matched_skills = [skill for skill in required_skills if skill in resume_skills]
    missing_skills = [skill for skill in required_skills if skill not in resume_skills]

    return {
        "resume_skills": resume_skills,
        "required_skills": required_skills,
        "matched_skills": matched_skills,
        "missing_skills": missing_skills
    }