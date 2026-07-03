from utils.llm import ask_gemini
from utils.llm import extract_resume_info

from utils.llm import extract_resume_info

resume_text = """
Prakriti Singh
CGPA: 8.8
GitHub: github.com/prakriti
Skills: Python, SQL, LangGraph
Education: B.Tech CSE
Projects: Resume Analyzer, Job Matcher
ML Intern at XYZ for 2 months
"""

result = extract_resume_info(resume_text)

print(result.candidate_name)
print(result.skills)
print(result.no_of_internships)