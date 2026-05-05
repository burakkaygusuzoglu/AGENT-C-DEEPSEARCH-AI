"""
Match Score System — Role-aware skill alignment scoring for career intelligence queries.
Lightweight, rule-based, explainable. No external ML dependencies.
"""

from __future__ import annotations

import re

# ── Canonical skill vocabulary ────────────────────────────────────────────────
# Multi-word phrases (e.g. "digital twin", "data modeling") are matched with
# regex \b boundaries and must be listed before any of their constituent words.
KNOWN_SKILLS: list[str] = [
    # Languages
    "python", "javascript", "typescript", "java", "go", "rust", "c++", "c#", "kotlin", "swift",
    # Data & query
    "sql", "nosql", "postgresql", "mysql", "mongodb", "redis", "sqlite",
    # Web frameworks & APIs
    "fastapi", "flask", "django", "express", "nodejs", "react", "vue", "angular", "nextjs", "api",
    # ML / DL frameworks
    "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn", "xgboost", "lightgbm",
    # GenAI / LLM tooling
    "langchain", "langgraph", "llamaindex", "huggingface", "openai", "anthropic",
    "rag", "llm", "nlp", "transformers", "fine-tuning", "lora",
    # Computer vision
    "opencv", "yolo", "yolov5", "yolov8", "pillow", "torchvision",
    # Data engineering — multi-word phrases before single words
    "data modeling", "data modelling",
    "spark", "kafka", "airflow", "dbt", "pandas", "numpy", "polars", "etl",
    # Mechanical / CAD — multi-word phrases before single words
    "digital twin", "solidworks", "autocad", "matlab", "cad",
    "simulation", "manufacturing", "thermodynamics", "materials",
    # MLOps / DevOps / Testing
    "mlflow", "wandb", "docker", "kubernetes", "git", "github", "ci/cd",
    "terraform", "linux", "bash", "testing",
    # Cloud
    "aws", "gcp", "azure", "s3", "lambda", "bigquery",
    # Databases / backends
    "supabase", "firebase", "pinecone", "weaviate", "chromadb",
    # General shorthand concepts
    "ml", "ai", "cv",
]

# ── Role → canonical skill sets ───────────────────────────────────────────────
ROLE_SKILL_MAP: dict[str, list[str]] = {
    "ai_engineer": [
        "python", "pytorch", "tensorflow", "sql", "git", "docker", "ml", "llm", "rag",
    ],
    "ml_engineer": [
        "python", "pytorch", "tensorflow", "sklearn", "sql", "docker", "git", "mlflow", "ml",
    ],
    "data_engineer": [
        "python", "sql", "spark", "airflow", "docker", "aws", "etl", "data modeling",
    ],
    "software_engineer": [
        "python", "javascript", "git", "sql", "react", "api", "testing", "docker",
    ],
    "mechanical_engineer": [
        "cad", "solidworks", "autocad", "matlab", "python", "simulation",
        "manufacturing", "thermodynamics", "materials", "digital twin",
    ],
}

# ── Human-readable display names (also exported for agent logging) ────────────
ROLE_DISPLAY_NAMES: dict[str, str] = {
    "ai_engineer":         "AI Engineer",
    "ml_engineer":         "Machine Learning Engineer",
    "data_engineer":       "Data Engineer",
    "software_engineer":   "Software Engineer",
    "mechanical_engineer": "Mechanical Engineer",
}

DEFAULT_ROLE = "ai_engineer"

# ── Role detection patterns ───────────────────────────────────────────────────
# Listed most-specific first: "machine learning engineer" before "machine engineer"
# so the first regex match wins and we never partially match a longer phrase.
_ROLE_PATTERNS: list[tuple[str, str]] = [
    (r"\bmachine\s+learning\s+engineer\b",       "ml_engineer"),
    (r"\bml\s+engineer\b",                        "ml_engineer"),
    (r"\bartificial\s+intelligence\s+engineer\b", "ai_engineer"),
    (r"\bai\s+engineer\b",                        "ai_engineer"),
    (r"\bdata\s+engineer\b",                      "data_engineer"),
    (r"\bsoftware\s+engineer\b",                  "software_engineer"),
    (r"\bsoftware\s+developer\b",                 "software_engineer"),
    (r"\bfull.?stack\s+developer\b",              "software_engineer"),
    (r"\bmechanical\s+engineer\b",                "mechanical_engineer"),
    # "machine engineer" normalised → mechanical engineer
    (r"\bmachine\s+engineer\b",                   "mechanical_engineer"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def detect_target_role(query: str) -> str:
    """
    Detect the target career role from free text.
    Returns a canonical role key (e.g. "ai_engineer").
    Falls back to DEFAULT_ROLE when nothing is recognised.
    """
    q = query.lower()
    for pattern, role in _ROLE_PATTERNS:
        if re.search(pattern, q):
            return role
    return DEFAULT_ROLE


def get_target_skills(role: str) -> list[str]:
    """Return the canonical skill list for a role key. Falls back to AI Engineer."""
    return ROLE_SKILL_MAP.get(role, ROLE_SKILL_MAP[DEFAULT_ROLE])


def extract_skills_from_text(text: str) -> list[str]:
    """
    Keyword-based skill extraction from free text.

    Uses regex word-boundary matching. Multi-word phrases like "digital twin"
    and "data modeling" are matched as complete phrases. Aliases are normalised
    (e.g. "data modelling" → "data modeling", "sklearn" → "scikit-learn").
    Returns a sorted, lowercase, deduplicated list.
    """
    if not text:
        return []

    text_lower = text.lower()
    found: set[str] = set()

    for skill in KNOWN_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            # Normalise aliases to canonical form
            if skill == "data modelling":
                found.add("data modeling")
            elif skill == "sklearn":
                found.add("scikit-learn")
            else:
                found.add(skill)

    return sorted(found)


def compute_match_score(
    user_skills: list[str],
    target_skills: list[str],
) -> dict:
    """
    Compute a structured career match score.

    Returns a dict with:
        match_score      int   0–100  weighted composite
        skills_match     int   0–100  % of target skills the user covers
        experience_match int   0–100  breadth heuristic based on skill count
        market_demand    str   "low" | "medium" | "high"
        matched_skills   list  intersection of user and target skills
        missing_skills   list  target skills the user is missing
    """
    if not target_skills:
        return {
            "match_score":      0,
            "skills_match":     0,
            "experience_match": 50,
            "market_demand":    "medium",
            "matched_skills":   [],
            "missing_skills":   [],
        }

    user_set   = set(user_skills)
    target_set = set(target_skills)

    matched = sorted(user_set & target_set)
    missing = sorted(target_set - user_set)

    # Skills coverage
    skills_match = int(len(matched) / len(target_set) * 100)

    # Experience breadth heuristic
    n = len(user_skills)
    if n >= 6:
        experience_match = 80
    elif n >= 4:
        experience_match = 65
    else:
        experience_match = 50

    # Market demand derived from target skill set
    target_text = " ".join(target_skills).lower()
    if any(kw in target_text for kw in ("ai", "ml", "data", "llm", "nlp", "rag", "etl")):
        market_demand = "high"
        demand_score  = 100
    elif any(kw in target_text for kw in ("api", "testing", "docker", "simulation", "cad")):
        market_demand = "medium"
        demand_score  = 70
    else:
        market_demand = "medium"
        demand_score  = 70

    match_score = int(
        0.6 * skills_match +
        0.3 * experience_match +
        0.1 * demand_score
    )

    return {
        "match_score":      match_score,
        "skills_match":     skills_match,
        "experience_match": experience_match,
        "market_demand":    market_demand,
        "matched_skills":   matched,
        "missing_skills":   missing,
    }


def format_score_section(score: dict, user_skills: list[str], role_name: str) -> str:
    """
    Render a Markdown Match Score Analysis block for when user skills ARE available.
    Returns an empty string if score dict is empty or invalid.
    """
    if not score:
        return ""

    demand_display = {
        "high":   "High 🔥",
        "medium": "Medium ⚡",
        "low":    "Low ❄️",
    }.get(score.get("market_demand", "medium"), "Medium ⚡")

    ms = score.get("match_score", 0)
    if ms >= 70:
        score_label    = f"{ms}% 🟢"
        interpretation = (
            f"You are well-aligned with {role_name} role requirements. "
            "Filling the missing skills below will make your profile highly competitive."
        )
    elif ms >= 50:
        score_label    = f"{ms}% 🟡"
        interpretation = (
            f"You are moderately aligned with {role_name} requirements. "
            "You have a solid foundation, but key tools are missing — prioritise the gaps below."
        )
    else:
        score_label    = f"{ms}% 🔴"
        interpretation = (
            f"You are in early alignment with {role_name} requirements. "
            "Focus on the missing skills below before applying to this role."
        )

    missing = score.get("missing_skills", [])
    matched = score.get("matched_skills", [])

    missing_lines = (
        "\n".join(f"  - `{s}`" for s in missing)
        if missing
        else "  ✅ None — you have full coverage of the core skill set!"
    )
    matched_lines = (
        ", ".join(f"`{s}`" for s in matched)
        if matched
        else "None detected"
    )

    return (
        f"\n\n---\n\n"
        f"## Match Score Analysis\n\n"
        f"**Target Role:** {role_name}\n\n"
        f"**Overall Match Score: {score_label}**\n\n"
        f"| Dimension | Score |\n"
        f"|---|---|\n"
        f"| Skills Match | {score.get('skills_match', 0)}% |\n"
        f"| Experience Breadth | {score.get('experience_match', 0)}% |\n"
        f"| Market Demand | {demand_display} |\n\n"
        f"**Matched Skills:** {matched_lines}\n\n"
        f"**Missing Core Skills:**\n{missing_lines}\n\n"
        f"**Interpretation:** {interpretation}\n\n"
        f"---\n\n"
    )


def format_no_skills_section(role_name: str, target_skills: list[str]) -> str:
    """
    Render an informative Match Score block when no user skills were detected in the query.
    Shows the target role, its required skills, and asks the user to provide their profile.
    """
    skills_list = "\n".join(f"  - {s.title()}" for s in target_skills)

    return (
        f"\n\n---\n\n"
        f"## Match Score Analysis\n\n"
        f"**Match Score:** Not enough user skill data\n\n"
        f"**Target Role:** {role_name}\n\n"
        f"**Detected User Skills:** Not provided\n\n"
        f"**Required Skills for {role_name}:**\n{skills_list}\n\n"
        f"### Missing Information\n\n"
        f"To calculate a real score, include your current skills, tools, projects, "
        f"and experience level in your query. For example:\n\n"
        f"> *\"I have Python and MATLAB experience. "
        f"Analyze my skill gaps for {role_name} roles.\"*\n\n"
        f"### Interpretation\n\n"
        f"The agent identified your target role as **{role_name}** but needs "
        f"your current skill profile to calculate a meaningful match score.\n\n"
        f"---\n\n"
    )


if __name__ == "__main__":
    print("── Test 1: AI Engineer with skills ──")
    q1      = "I have Python, FastAPI, SQL, YOLO, OpenCV. Analyze my gaps for AI Engineer roles."
    role1   = detect_target_role(q1)
    skills1 = extract_skills_from_text(q1)
    target1 = get_target_skills(role1)
    score1  = compute_match_score(skills1, target1)
    print(f"Role:     {ROLE_DISPLAY_NAMES[role1]}")
    print(f"Skills:   {skills1}")
    print(format_score_section(score1, skills1, ROLE_DISPLAY_NAMES[role1]))

    print("── Test 2: Machine Engineer, no skills (normalised → Mechanical Engineer) ──")
    q2      = "check the recent updates, how can i be a machine engineer with the newest informations"
    role2   = detect_target_role(q2)
    skills2 = extract_skills_from_text(q2)
    target2 = get_target_skills(role2)
    print(f"Role:     {ROLE_DISPLAY_NAMES[role2]}")
    print(f"Skills:   {skills2}")
    print(format_no_skills_section(ROLE_DISPLAY_NAMES[role2], target2))

    print("── Test 3: Data Engineer with skills ──")
    q3      = "I have Python, SQL, Airflow, Spark, Docker. What gaps do I have for Data Engineer?"
    role3   = detect_target_role(q3)
    skills3 = extract_skills_from_text(q3)
    target3 = get_target_skills(role3)
    score3  = compute_match_score(skills3, target3)
    print(f"Role:     {ROLE_DISPLAY_NAMES[role3]}")
    print(f"Skills:   {skills3}")
    print(format_score_section(score3, skills3, ROLE_DISPLAY_NAMES[role3]))
