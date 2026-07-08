from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from utils.llm import extract_resume_info
from resume_agent import analyze_resume
from skill_gap_agent import analyze_skill_gap
from learning_planner_agent import generate_learning_plan


class CareerState(TypedDict):
    resume_text: str
    target_role: str

    resume_info: Optional[dict]
    resume_analysis: Optional[dict]
    skill_gap: Optional[dict]
    learning_plan: Optional[dict]

    error: Optional[str]


def extract_resume_node(state: CareerState):
    try:
        result = extract_resume_info(state["resume_text"])
        return {
            "resume_info": result.model_dump(),
            "error": None
        }
    except Exception as e:
        return {
            "error": f"Resume extraction failed: {str(e)}"
        }

def resume_analysis_node(state: CareerState):
    try:
        # If your analyze_resume takes resume_text + target_role:
        result = analyze_resume(state["resume_text"], state["target_role"])

        # If result is Pydantic:
        if hasattr(result, "model_dump"):
            result = result.model_dump()

        return {
            "resume_analysis": result,
            "error": None
        }
    except Exception as e:
        return {
            "error": f"Resume analysis failed: {str(e)}"
        }

def skill_gap_node(state: CareerState):
    try:
        result = analyze_skill_gap(state["resume_text"], state["target_role"])

        if hasattr(result, "model_dump"):
            result = result.model_dump()

        return {
            "skill_gap": result,
            "error": None
        }
    except Exception as e:
        return {
            "error": f"Skill gap analysis failed: {str(e)}"
        }

def learning_plan_node(state: CareerState):
    try:
        result = generate_learning_plan(state["resume_text"], state["target_role"])

        if hasattr(result, "model_dump"):
            result = result.model_dump()

        return {
            "learning_plan": result,
            "error": None
        }
    except Exception as e:
        return {
            "error": f"Learning planner failed: {str(e)}"
        }

def route_after_extraction(state: CareerState):
    if state.get("error"):
        return "end"

    resume_info = state.get("resume_info")

    if not resume_info:
        return "end"
    if(
        not resume_info.get("candidate_name")
        and not resume_info.get("skills")
        and not resume_info.get("education")
    ):
        return "end"

    return "resume_analysis"


def route_after_skill_gap(state: CareerState):
    if state.get("error"):
        return "end"

    skill_gap = state.get("skill_gap")
    if not skill_gap:
        return "end"

    missing_required = skill_gap.get("missing_required_skills", [])
    missing_preferred = skill_gap.get("missing_preferred_skills", [])

    if len(missing_required) == 0 and len(missing_preferred) == 0:
        return "end"

    return "learning_plan"

def build_career_graph():
    graph_builder = StateGraph(CareerState)

    graph_builder.add_node("extract_resume", extract_resume_node)
    graph_builder.add_node("resume_analysis", resume_analysis_node)
    graph_builder.add_node("skill_gap", skill_gap_node)
    graph_builder.add_node("learning_plan", learning_plan_node)

    graph_builder.add_edge(START, "extract_resume")

    graph_builder.add_conditional_edges(
        "extract_resume",
        route_after_extraction,
        {
            "resume_analysis": "resume_analysis",
            "end": END
        }
    )

    graph_builder.add_edge("resume_analysis", "skill_gap")

    graph_builder.add_conditional_edges(
        "skill_gap",
        route_after_skill_gap,
        {
            "learning_plan": "learning_plan",
            "end": END
        }
    )

    graph_builder.add_edge("learning_plan", END)

    graph = graph_builder.compile()
    return graph