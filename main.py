from utils.llm import extract_resume_info
from resume_agent import analyze_resume
from pdf_reader import text_from_pdf
from skill_gap_agent import skill_gap

#text file
with open("sample_data.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

#pdf
"""resume_text = text_from_pdf("resume.pdf")"""

result = extract_resume_info(resume_text)

#extract
print(result.candidate_name)
print(result.skills)
print(result.education)

#analysis
analysis = analyze_resume(resume_text, "AI/ML Intern")
print("Strengths:", analysis.strengths)
print("Weaknesses:", analysis.weaknesses)
print("Missing Keywords:", analysis.missing_keywords)
print("Fit Summary:", analysis.fit_summary)

#skill gap
required_skills =["Python", "Machine Learning", "SQL", "TensorFlow"]
gap_result = skill_gap(resume_text, required_skills)
print("matched_skills:")
print(gap_result["matched_skills"])
print("missing_skills:")
print( gap_result["missing_skills"])
