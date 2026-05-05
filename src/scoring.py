"""
Match Score System — Role-aware, category-based skill alignment scoring.
Lightweight, rule-based, explainable. No external ML dependencies.
"""

from __future__ import annotations

import re

# ─────────────────────────────────────────────────────────────────────────────
# Skill vocabulary
# Multi-word phrases listed before their single-word subsets so that regex
# word-boundary matching picks them up as complete phrases first.
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_SKILLS: list[str] = [
    # Multi-word concept phrases — must come first
    "machine learning", "deep learning", "computer vision",
    "data modeling", "data modelling", "data pipelines", "data structures",
    "data processing", "data warehouse", "data analysis",
    "additive manufacturing", "digital twin", "unit testing",
    "agentic ai",
    # Programming languages
    "python", "javascript", "typescript", "java", "go", "rust",
    "c++", "c#", "kotlin", "swift", "html", "css", "matlab",
    # Data & query
    "sql", "nosql", "postgresql", "mysql", "mongodb", "redis", "sqlite",
    # Web frameworks & APIs
    "fastapi", "flask", "django", "express", "nodejs",
    "react", "vue", "angular", "nextjs",
    # ML / DL frameworks
    "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
    "xgboost", "lightgbm",
    # GenAI / LLM tooling
    "langchain", "langgraph", "llamaindex", "huggingface", "openai", "anthropic",
    "rag", "llm", "nlp", "transformers", "fine-tuning", "lora",
    # Computer vision tools
    "opencv", "yolo", "yolov5", "yolov8", "yolov11", "pillow", "torchvision",
    # Data engineering
    "spark", "kafka", "airflow", "dbt", "pandas", "numpy", "polars", "etl",
    # Mechanical / CAD
    "solidworks", "autocad", "cad", "simulation", "manufacturing",
    "thermodynamics", "materials", "mechanics",
    # MLOps / DevOps / Quality
    "mlflow", "wandb", "docker", "kubernetes", "git", "github",
    "ci/cd", "terraform", "linux", "bash", "pytest", "selenium", "testing",
    # Cloud platforms
    "aws", "gcp", "azure", "s3", "lambda", "bigquery",
    # Backend / databases
    "supabase", "firebase", "pinecone", "weaviate", "chromadb",
    # General concepts
    "ml", "ai", "cv", "api", "oop", "algorithms", "programming",
]

# ─────────────────────────────────────────────────────────────────────────────
# Synonym / category map
# Maps raw detected tokens → canonical normalised skill name.
# Unlisted tokens are kept as-is.
# ─────────────────────────────────────────────────────────────────────────────

SKILL_SYNONYMS: dict[str, str] = {
    # Computer vision
    "opencv":           "computer vision",
    "yolo":             "computer vision",
    "yolov5":           "computer vision",
    "yolov8":           "computer vision",
    "yolov11":          "computer vision",
    "torchvision":      "computer vision",
    "pillow":           "computer vision",
    "cv":               "computer vision",
    # Backend / API
    "fastapi":          "backend api",
    "flask":            "backend api",
    "django":           "backend api",
    "express":          "backend api",
    "nodejs":           "backend api",
    "api":              "backend api",
    # SQL normalisation (keep "sql" itself as-is)
    "postgresql":       "sql",
    "mysql":            "sql",
    "sqlite":           "sql",
    # Data processing
    "pandas":           "data processing",
    "numpy":            "data processing",
    "polars":           "data processing",
    # Agentic AI grouping
    "langchain":        "agentic ai",
    "langgraph":        "agentic ai",
    "rag":              "agentic ai",
    "llm":              "agentic ai",
    "llamaindex":       "agentic ai",
    # Deployment
    "docker":           "deployment",
    "kubernetes":       "deployment",
    # Cloud
    "aws":              "cloud",
    "gcp":              "cloud",
    "azure":            "cloud",
    "s3":               "cloud",
    "lambda":           "cloud",
    "bigquery":         "cloud",
    # Frontend
    "react":            "frontend",
    "vue":              "frontend",
    "angular":          "frontend",
    "nextjs":           "frontend",
    "html":             "frontend",
    "css":              "frontend",
    "javascript":       "frontend",
    "typescript":       "frontend",
    # Testing
    "pytest":           "testing",
    "selenium":         "testing",
    "unit testing":     "testing",
    # ML short-forms
    "ml":               "machine learning",
    # Aliases
    "sklearn":          "scikit-learn",
    "data modelling":   "data modeling",
}

# ─────────────────────────────────────────────────────────────────────────────
# Role → categorised skill requirements (all values must be detectable from text)
# ─────────────────────────────────────────────────────────────────────────────

ROLE_SKILL_MAP: dict[str, dict[str, list[str]]] = {
    "ai_engineer": {
        "core":       ["python", "machine learning", "deep learning", "agentic ai"],
        "vision":     ["computer vision"],
        "tools":      ["pytorch", "tensorflow"],
        "data":       ["sql", "data processing"],
        "deployment": ["deployment", "cloud", "backend api"],
    },
    "ml_engineer": {
        "core":       ["python", "machine learning", "deep learning"],
        "tools":      ["pytorch", "tensorflow", "scikit-learn", "mlflow"],
        "data":       ["sql", "data processing"],
        "deployment": ["deployment", "cloud"],
    },
    "data_engineer": {
        "core":       ["python", "sql", "etl", "data modeling"],
        "tools":      ["airflow", "spark", "dbt", "data processing"],
        "deployment": ["deployment", "cloud"],
    },
    "software_engineer": {
        "core":       ["python", "sql", "git", "algorithms"],
        "backend":    ["backend api"],
        "frontend":   ["frontend"],
        "quality":    ["testing"],
        "deployment": ["deployment", "cloud"],
    },
    "mechanical_engineer": {
        "core":    ["cad", "thermodynamics", "materials", "manufacturing"],
        "tools":   ["solidworks", "autocad", "matlab", "simulation"],
        "modern":  ["digital twin", "python", "data analysis"],
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Role metadata
# ─────────────────────────────────────────────────────────────────────────────

ROLE_DISPLAY_NAMES: dict[str, str] = {
    "ai_engineer":         "AI Engineer",
    "ml_engineer":         "Machine Learning Engineer",
    "data_engineer":       "Data Engineer",
    "software_engineer":   "Software Engineer",
    "mechanical_engineer": "Mechanical Engineer",
}

DEFAULT_ROLE = "ai_engineer"

_ROLE_DEMAND: dict[str, tuple[str, int]] = {
    "ai_engineer":         ("high",   100),
    "ml_engineer":         ("high",   100),
    "data_engineer":       ("high",   100),
    "software_engineer":   ("high",    95),
    "mechanical_engineer": ("medium",  75),
}

PROJECT_RECOMMENDATIONS: dict[str, str] = {
    "ai_engineer": (
        "Build a RAG-powered research assistant using LangGraph, ChromaDB, and FastAPI. "
        "Add a Tavily web-search tool and deploy to Hugging Face Spaces with a Gradio interface."
    ),
    "ml_engineer": (
        "Pick a Kaggle dataset, fine-tune a pre-trained Hugging Face model with LoRA, "
        "track experiments with MLflow, serve predictions via FastAPI, and containerise with Docker."
    ),
    "data_engineer": (
        "Build an end-to-end pipeline: pull a public API → transform with Pandas/dbt → "
        "load to PostgreSQL → schedule with Airflow → expose a summary dashboard. Containerise with Docker."
    ),
    "software_engineer": (
        "Build a full-stack task manager: FastAPI backend + PostgreSQL + React frontend. "
        "Add JWT auth, write pytest unit tests (80%+ coverage), Dockerise, and deploy to a free cloud tier."
    ),
    "mechanical_engineer": (
        "Build a Python-based simulation of a mechanical system (beam deflection, "
        "heat transfer, or motor dynamics). Visualise results with Matplotlib, document in a GitHub README, "
        "and explore digital-twin concepts with a public IoT sensor dataset."
    ),
}

_QUICK_WIN_MAP: dict[str, str] = {
    "python":           "Complete CS50P or freeCodeCamp Python course (free, 1-2 weeks)",
    "sql":              "Complete Mode Analytics SQL Tutorial (free, 2-4 h) + one Kaggle SQL challenge",
    "machine learning": "Take fast.ai Lesson 1-3 (free) and train a model on a Kaggle starter dataset",
    "deep learning":    "Build a PyTorch neural network on MNIST using the official tutorials (1 weekend)",
    "computer vision":  "Complete a YOLO object-detection mini-project with Ultralytics docs (2-3 h)",
    "agentic ai":       "Build a LangChain/LangGraph agent with web search + memory in one weekend",
    "pytorch":          "Follow PyTorch Official Quickstart → train a custom image classifier (1 week)",
    "tensorflow":       "Complete TensorFlow Developer Certificate curriculum on Coursera (free audit)",
    "deployment":       "Containerise one existing project with Docker (official get-started guide, 2-3 h)",
    "cloud":            "Complete AWS Free Tier tutorial: deploy a simple app to Lambda or EC2 (1 day)",
    "backend api":      "Build a CRUD REST API with FastAPI + PostgreSQL (FastAPI docs, 1 day)",
    "frontend":         "Build a React dashboard that calls your own API (Scrimba free course, 1 week)",
    "testing":          "Add pytest tests to an existing Python project — aim for 80% coverage (2-3 h)",
    "data processing":  "Work through Pandas documentation exercises + complete one EDA Kaggle notebook",
    "etl":              "Build a simple ETL: CSV → Pandas transform → PostgreSQL load (1 day)",
    "data modeling":    "Complete dbt Learn course (free, 2-3 h) + model a star-schema dataset",
    "airflow":          "Set up Airflow locally with Docker Compose and write your first DAG (1 day)",
    "spark":            "Complete Databricks Community Edition free Spark tutorial (1-2 days)",
    "scikit-learn":     "Work through scikit-learn User Guide examples on a Kaggle competition dataset",
    "mlflow":           "Add MLflow tracking to any existing training script (MLflow quickstart, 1 h)",
    "git":              "Complete 'Learn Git Branching' (free interactive game) in 2-3 h",
    "algorithms":       "Start NeetCode 150 on LeetCode — Easy arrays/strings first (free)",
    "cad":              "Complete Onshape free CAD Fundamentals (browser-based, no install, 3 h)",
    "solidworks":       "Start MySolidWorks free online fundamentals on MathWorks (2-3 h)",
    "autocad":          "Use Autodesk AutoCAD web app free tutorial (no installation required)",
    "matlab":           "Complete MATLAB Onramp on MathWorks (free, 2 h)",
    "simulation":       "Run a simple FEM simulation in FreeCAD (open-source) or ANSYS Student (free)",
    "digital twin":     "Study Azure Digital Twins free docs + build a basic twin model (1-2 days)",
    "thermodynamics":   "Review MIT OpenCourseWare 2.005 lecture notes (free)",
    "materials":        "Audit MIT 3.091 Intro to Solid State Chemistry on OCW (free)",
    "manufacturing":    "Study CNC and additive manufacturing basics on Coursera (free audit)",
    "data analysis":    "Complete a Python data analysis project on a public engineering dataset (1 day)",
}

# ─────────────────────────────────────────────────────────────────────────────
# Role detection
# ─────────────────────────────────────────────────────────────────────────────

_ROLE_PATTERNS: list[tuple[str, str]] = [
    # Most-specific patterns first to prevent partial matches
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
    Detect the target career role from free text using ordered regex patterns.
    Returns a canonical role key. Falls back to DEFAULT_ROLE if unrecognised.
    "machine engineer" is normalised to "mechanical_engineer".
    """
    q = query.lower()
    for pattern, role in _ROLE_PATTERNS:
        if re.search(pattern, q):
            return role
    return DEFAULT_ROLE


def get_target_skills(role: str) -> dict[str, list[str]]:
    """
    Return the categorised skill dict for a role key.
    Falls back to the AI Engineer map for unknown roles.
    """
    return ROLE_SKILL_MAP.get(role, ROLE_SKILL_MAP[DEFAULT_ROLE])


def flatten_target_skills(target_skill_dict: dict[str, list[str]]) -> list[str]:
    """Flatten a categorised skill dict into a deduplicated ordered list."""
    seen: set[str] = set()
    result: list[str] = []
    for skills in target_skill_dict.values():
        for s in skills:
            if s not in seen:
                seen.add(s)
                result.append(s)
    return result


def extract_skills_from_text(text: str) -> list[str]:
    """
    Keyword-based skill extraction with synonym normalisation.

    1. Scans KNOWN_SKILLS against the input using regex word-boundary matching.
       Multi-word phrases (e.g. "digital twin") are matched as complete phrases.
    2. Maps each matched token through SKILL_SYNONYMS to a canonical form.
    3. Returns a sorted, deduplicated list of canonical skill names.

    Examples:
        "opencv" → "computer vision"
        "langchain" → "agentic ai"
        "docker" → "deployment"
        "pytorch" → "pytorch"  (no synonym, kept as-is)
    """
    if not text:
        return []

    text_lower = text.lower()
    found: set[str] = set()

    for skill in KNOWN_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            canonical = SKILL_SYNONYMS.get(skill, skill)
            found.add(canonical)

    return sorted(found)


def compute_match_score(
    user_skills: list[str],
    target_skill_dict: dict[str, list[str]],
    target_role: str,
) -> dict:
    """
    Compute a structured, role-aware career match score.

    Args:
        user_skills:       Normalised skills from extract_skills_from_text().
        target_skill_dict: Categorised skill map from get_target_skills().
        target_role:       Canonical role key (e.g. "ai_engineer").

    Returns a dict with:
        target_role          str   human-readable role name
        match_score          int   0–100  weighted composite
        skills_match         int   0–100  coverage of target flat skill list
        experience_match     int   0–100  breadth + portfolio heuristic
        market_demand        str   "low" | "medium" | "high"
        detected_skills      list  canonical skills found in user text
        missing_skills       list  target skills the user is missing
        strong_skills        list  target skills the user already has
        quick_wins           list  top-3 most learnable missing skills with steps
        project_recommendation str  role-specific capstone project idea
    """
    target_flat = flatten_target_skills(target_skill_dict)
    target_set  = set(target_flat)
    user_set    = set(user_skills)

    strong_skills  = sorted(user_set & target_set)
    missing_skills = sorted(target_set - user_set)

    # Skills coverage of the target list
    skills_match = int(len(strong_skills) / max(len(target_set), 1) * 100)

    # Experience breadth heuristic: more distinct canonical skills = broader profile
    n = len(user_skills)
    if n >= 7:
        experience_match = 85
    elif n >= 5:
        experience_match = 75
    elif n >= 3:
        experience_match = 60
    else:
        experience_match = 45

    # Market demand from role metadata
    market_demand, demand_score = _ROLE_DEMAND.get(target_role, ("medium", 70))

    # Weighted composite: skills 60%, experience 25%, market 15%
    match_score = int(
        0.60 * skills_match +
        0.25 * experience_match +
        0.15 * demand_score
    )

    # Quick wins: actionable steps for up to 3 missing skills
    quick_wins = [
        _QUICK_WIN_MAP[s]
        for s in missing_skills
        if s in _QUICK_WIN_MAP
    ][:3]

    return {
        "target_role":            ROLE_DISPLAY_NAMES.get(target_role, target_role),
        "match_score":            match_score,
        "skills_match":           skills_match,
        "experience_match":       experience_match,
        "market_demand":          market_demand,
        "detected_skills":        sorted(user_skills),
        "missing_skills":         missing_skills,
        "strong_skills":          strong_skills,
        "quick_wins":             quick_wins,
        "project_recommendation": PROJECT_RECOMMENDATIONS.get(
            target_role,
            "Build an end-to-end project combining your existing skills with one new technology. "
            "Push it to GitHub with a complete README and a live demo.",
        ),
    }


def format_score_section(score: dict) -> str:
    """
    Render a Markdown '## Match Score & Career Recommendations' block
    from a compute_match_score() result dict.
    Returns an empty string if the score dict is missing or empty.
    """
    if not score:
        return ""

    demand_display = {
        "high":   "High 🔥",
        "medium": "Medium ⚡",
        "low":    "Low ❄️",
    }.get(score.get("market_demand", "medium"), "Medium ⚡")

    ms = score.get("match_score", 0)
    role_name = score.get("target_role", "AI Engineer")

    if ms >= 70:
        score_label    = f"{ms}% 🟢"
        interpretation = (
            f"You are well-aligned with {role_name} requirements. "
            "Closing the missing skills below will make your profile highly competitive."
        )
    elif ms >= 50:
        score_label    = f"{ms}% 🟡"
        interpretation = (
            f"You are moderately aligned with {role_name} requirements. "
            "You have a solid foundation — prioritise the missing skills and quick wins below."
        )
    else:
        score_label    = f"{ms}% 🔴"
        interpretation = (
            f"You are in early alignment with {role_name} requirements. "
            "Focus on the missing skills below before applying to this role."
        )

    strong  = score.get("strong_skills", [])
    missing = score.get("missing_skills", [])
    qw      = score.get("quick_wins", [])
    project = score.get("project_recommendation", "")

    strong_lines  = ", ".join(f"`{s}`" for s in strong)  if strong  else "None detected yet"
    missing_lines = "\n".join(f"  - `{s}`" for s in missing) if missing else "  ✅ Full coverage!"
    qw_lines      = "\n".join(f"  {i+1}. {w}" for i, w in enumerate(qw)) if qw else "  No quick wins needed — great coverage!"

    return (
        f"\n\n---\n\n"
        f"## Match Score & Career Recommendations\n\n"
        f"**Target Role:** {role_name}\n\n"
        f"**Overall Match Score: {score_label}**\n\n"
        f"| Dimension | Score |\n"
        f"|---|---|\n"
        f"| Skills Match | {score.get('skills_match', 0)}% |\n"
        f"| Experience Breadth | {score.get('experience_match', 0)}% |\n"
        f"| Market Demand | {demand_display} |\n\n"
        f"**Detected Strengths:** {strong_lines}\n\n"
        f"**Missing Skills:**\n{missing_lines}\n\n"
        f"**Quick Wins (learn these first):**\n{qw_lines}\n\n"
        f"**Recommended Portfolio Project:**\n{project}\n\n"
        f"**Interpretation:** {interpretation}\n\n"
        f"---\n\n"
    )


def format_no_skills_section(
    role_name: str,
    target_skill_dict: dict[str, list[str]],
) -> str:
    """
    Render an informative block when no user skills were detected in the query.
    Shows the required skills by category and prompts the user to provide their profile.
    """
    category_lines = ""
    for category, skills in target_skill_dict.items():
        skill_str = ", ".join(skills)
        category_lines += f"  **{category.replace('_', ' ').title()}:** {skill_str}\n"

    return (
        f"\n\n---\n\n"
        f"## Match Score & Career Recommendations\n\n"
        f"**Target Role:** {role_name}\n\n"
        f"**Match Score:** Not enough user skill data\n\n"
        f"**Detected User Skills:** Not provided\n\n"
        f"**Required Skills by Category:**\n{category_lines}\n"
        f"### What to Do Next\n\n"
        f"Include your current skills, tools, and projects in your query to get a "
        f"personalised score. For example:\n\n"
        f"> *\"I have Python and MATLAB. "
        f"Analyze my skill gaps for {role_name} roles.\"*\n\n"
        f"### Interpretation\n\n"
        f"Target role **{role_name}** was detected from your query. "
        f"Provide your current skill profile to unlock a personalised match score "
        f"and targeted recommendations.\n\n"
        f"---\n\n"
    )


if __name__ == "__main__":
    sep = "─" * 60

    print(f"\n{sep}")
    print("Test 1: AI Engineer with skills")
    q1      = "I have Python, FastAPI, SQL, YOLO, OpenCV, LangChain, Docker. Gaps for AI Engineer?"
    role1   = detect_target_role(q1)
    skills1 = extract_skills_from_text(q1)
    target1 = get_target_skills(role1)
    score1  = compute_match_score(skills1, target1, role1)
    print(f"Role:    {ROLE_DISPLAY_NAMES[role1]}")
    print(f"Skills:  {skills1}")
    print(f"Score:   {score1['match_score']}%  |  strong={score1['strong_skills']}  |  missing={score1['missing_skills']}")
    print(format_score_section(score1))

    print(f"\n{sep}")
    print("Test 2: Machine engineer, no skills (normalised → Mechanical Engineer)")
    q2      = "check the recent updates, how can i be a machine engineer with the newest informations"
    role2   = detect_target_role(q2)
    skills2 = extract_skills_from_text(q2)
    target2 = get_target_skills(role2)
    print(f"Role:    {ROLE_DISPLAY_NAMES[role2]}")
    print(f"Skills:  {skills2}")
    print(format_no_skills_section(ROLE_DISPLAY_NAMES[role2], target2))

    print(f"\n{sep}")
    print("Test 3: Data Engineer with skills")
    q3      = "I know Python, SQL, Airflow, Spark, Docker, dbt. What gaps for Data Engineer?"
    role3   = detect_target_role(q3)
    skills3 = extract_skills_from_text(q3)
    target3 = get_target_skills(role3)
    score3  = compute_match_score(skills3, target3, role3)
    print(f"Role:    {ROLE_DISPLAY_NAMES[role3]}")
    print(f"Skills:  {skills3}")
    print(f"Score:   {score3['match_score']}%  |  strong={score3['strong_skills']}  |  missing={score3['missing_skills']}")
    print(format_score_section(score3))
