from pdf_reader import text_from_pdf
from pipeline_graph import build_career_graph

resume_text = text_from_pdf("sde_sample_resume.pdf")

graph = build_career_graph()

final_state = graph.invoke({
    "resume_text": resume_text,
    "target_role": "AI/ML Intern",
    "resume_info": None,
    "resume_analysis": None,
    "skill_gap": None,
    "learning_plan": None,
    "error": None
})

print("\n=== RESUME INFO ===")
print(final_state["resume_info"])

print("\n=== RESUME ANALYSIS ===")
print(final_state["resume_analysis"])

print("\n=== SKILL GAP ===")
print(final_state["skill_gap"])

print("\n=== LEARNING PLAN ===")
print(final_state["learning_plan"])

print("\n=== ERROR ===")
print(final_state["error"])