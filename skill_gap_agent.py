import os
from dotenv import load_dotenv
from pydantic import BaseModel

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

from utils.llm import extract_resume_info

load_dotenv()

CHROMA_DIR = "chroma_db"


class SkillGapResult(BaseModel):
    target_role: str
    matched_skills: list[str]
    missing_required_skills: list[str]
    missing_preferred_skills: list[str]
    project_gaps: list[str]
    skill_match_score: int
    priority_skills_to_learn: list[str]
    summary: str


def get_role_context(target_role: str, k: int = 3) -> str:
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    vector_store = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )

    query = f"Required skills, preferred skills and expected projects for {target_role}"
    docs = vector_store.similarity_search(query, k=k)

    context = "\n\n".join(doc.page_content for doc in docs)
    return context


def analyze_skill_gap(resume_text: str, target_role: str) -> SkillGapResult:
    # Step 1: extract structured resume info
    resume_info = extract_resume_info(resume_text)
    resume_data = resume_info.model_dump()

    # Step 2: retrieve role context from ChromaDB
    retrieved_context = get_role_context(target_role)

    # Step 3: create LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    structured_llm = llm.with_structured_output(SkillGapResult)

    # Step 4: prompt
    prompt = f"""
You are a Skill Gap Analysis Agent.

Your job is to compare the candidate's resume with the target role requirements.

Target Role:
{target_role}

Candidate Resume Data:
{resume_data}

Retrieved Role Knowledge:
{retrieved_context}

Instructions:
1. Compare the candidate's skills, projects, and experience against the target role.
2. Identify:
   - matched_skills
   - missing_required_skills
   - missing_preferred_skills
   - project_gaps (important project types or portfolio evidence missing for this role)
3. Give a skill_match_score from 0 to 100 based on overall role fit.
4. Give priority_skills_to_learn: the most important skills the candidate should focus on first.
5. Write a short summary of the candidate's fit for the role.

Important rules:
- Use the retrieved role knowledge as the grounding context.
- Use only reasonable inferences from the candidate resume data.
- Do not invent fake projects or fake experience.
- Be specific and role-oriented.

Return the result in structured format.
"""

    result = structured_llm.invoke(prompt)
    return result