from utils.llm import extract_resume_info
from resume_agent import analyze_resume
from pdf_reader import text_from_pdf
from skill_gap_agent import analyze_skill_gap
from learning_planner_agent import generate_learning_plan


#text file
"""with open("sample_data.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()"""

#pdf
resume_text = text_from_pdf("sde_sample_resume.pdf")

extract_result = extract_resume_info(resume_text)

#extract
print(extract_result.candidate_name)
print(extract_result.skills)
print(extract_result.education)

#analysis
analysis = analyze_resume(resume_text, "AI/ML Intern")
print("Strengths:", analysis.strengths)
print("Weaknesses:", analysis.weaknesses)
print("Missing Keywords:", analysis.missing_keywords)
print("Fit Summary:", analysis.fit_summary)



#skill_gap
with open("sample_data.txt", "r", encoding="utf-8") as f:
    resume_text = f.read()

gap_result = analyze_skill_gap(resume_text, "AI/ML Intern")

print("TARGET ROLE:", gap_result.target_role)
print("MATCHED SKILLS:", gap_result.matched_skills)
print("MISSING REQUIRED SKILLS:", gap_result.missing_required_skills)
print("MISSING PREFERRED SKILLS:", gap_result.missing_preferred_skills)
print("PROJECT GAPS:", gap_result.project_gaps)
print("SKILL MATCH SCORE:", gap_result.skill_match_score)
print("PRIORITY SKILLS:", gap_result.priority_skills_to_learn)
print("SUMMARY:", gap_result.summary)



#learning_agent

learning_result = generate_learning_plan(resume_text, "AI/ML Intern")

print("TARGET ROLE:", learning_result.target_role)
print("CURRENT STRENGTHS:", learning_result.current_strengths)
print("PRIORITY SKILLS:", learning_result.priority_skills_to_learn)
print("RECOMMENDED PROJECTS:", learning_result.recommended_projects)
print("SUMMARY:", learning_result.summary)

for phase in learning_result.learning_plan:
    print("\nPHASE:", phase.phase)
    print("FOCUS SKILLS:", phase.focus_skills)
    print("TOPICS:", phase.topics)
    print("DELIVERABLE:", phase.deliverable)
