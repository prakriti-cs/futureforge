from utils.llm import extract_resume_info
from resume_agent import analyze_resume
from pdf_reader import text_from_pdf

#text file
"""with open("sample_data.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()
"""
#pdf
resume_text = text_from_pdf("resume.pdf")

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
