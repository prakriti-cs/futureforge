import os
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from utils.llm import extract_resume_info
from skill_gap_agent import analyze_skill_gap

load_dotenv()

CHROMA_DIR = "chroma_db"


class LearningPhase(BaseModel):
    phase: str
    focus_skills: list[str]
    topics: list[str]
    deliverable: str


class LearningPlanResult(BaseModel):
    target_role: str
    current_strengths: list[str]
    priority_skills_to_learn: list[str]
    learning_plan: list[LearningPhase]
    recommended_projects: list[str]
    summary: str


def get_learning_context(target_role: str, k: int = 4) -> str:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    query = f"Learning roadmap, required skills, project suggestions, and progression plan for {target_role}"
    docs = vector_store.similarity_search(query, k=k)

    context = "\n\n".join(doc.page_content for doc in docs)
    return context


def generate_learning_plan(resume_text: str, target_role: str) -> LearningPlanResult:
   
    resume_info = extract_resume_info(resume_text)
    resume_data = resume_info.model_dump()

    
    skill_gap = analyze_skill_gap(resume_text, target_role)
    skill_gap_data = skill_gap.model_dump()

    
    learning_context = get_learning_context(target_role)

    
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    structured_llm = llm.with_structured_output(LearningPlanResult)

    prompt = f"""
You are a Learning Planner Agent.

Your task is to create a practical learning roadmap for a student / intern candidate.

Target Role:
{target_role}

Candidate Resume Data:
{resume_data}

Skill Gap Analysis:
{skill_gap_data}

Retrieved Learning Context:
{learning_context}

Instructions:
1. Use the candidate's resume data and skill gap analysis to identify their current strengths.
2. Focus especially on:
   - missing_required_skills
   - missing_preferred_skills
   - project_gaps
   - priority_skills_to_learn
3. Create a realistic learning plan divided into 3 phases:
   - Phase 1: Foundations / immediate weak areas
   - Phase 2: Core role preparation
   - Phase 3: Project-building / portfolio readiness
4. For each phase, return:
   - phase
   - focus_skills
   - topics
   - deliverable
5. Recommend 2 to 4 projects relevant to the target role and the candidate’s gaps.
6. Keep the plan practical for a college student preparing for internships.
7. Prioritize required skills before preferred skills.
8. Do not invent fake resume details. Use only reasonable inferences from the provided resume and retrieved role-learning context.

Return structured output only.
"""

    result = structured_llm.invoke(prompt)
    return result