# Demo Queries — Career & Technology Intelligence Agent

A collection of example queries demonstrating the agent's intent routing and research capabilities.
Each query shows the expected **intent** and **route** classification.

---

## Career Planning Queries

### 1. 30-Day Internship Preparation Plan (Uses: BOTH — corpus + web)
```
I am a 3rd year computer engineering student interested in AI engineering internships in Europe.
I know Python, FastAPI, and have built a RAG system with LangChain.
Use your internal knowledge and live web search to create a 30-day preparation plan.
```
**Intent:** `career_plan` | **Route:** `both`
**Why both:** Internal corpus provides the roadmap structure and skill requirements; web search finds current open positions and company-specific requirements.

---

### 2. AI Engineer vs Data Engineer Career Comparison (Uses: BOTH)
```
Compare the skills needed for AI Engineer vs Data Engineer roles and recommend
which career path is stronger for an early-career student in 2026.
```
**Intent:** `career_plan` | **Route:** `both`
**Why both:** Corpus has role definitions and roadmaps; web brings current market data, salary trends, and demand signals.

---

### 3. Agentic AI Trend Research + Employability (Uses: BOTH)
```
Find current trends in agentic AI and explain what projects I should build
to become more employable as a junior AI engineer.
```
**Intent:** `tech_trend` | **Route:** `both`
**Why both:** Web finds the latest agentic frameworks and industry reports; corpus explains foundational concepts (LangGraph, agent patterns).

---

## Skill Gap Queries

### 4. Personalized Skill Gap Analysis (Uses: BOTH)
```
Analyze my current skills: Python, JavaScript, SQL, YOLO, OpenCV, FastAPI, Supabase.
What gaps do I have for AI Engineer roles and what should I prioritize learning next?
```
**Intent:** `skill_gap` | **Route:** `both`
**Why both:** Corpus provides AI Engineer skill requirements framework; web search enriches with current job market data showing what employers actually ask for.

---

### 5. Python/AI Internship Skill Check (Uses: RAG — corpus only)
```
Prepare me for a junior Python/AI internship interview using your internal knowledge base.
Focus on technical interview questions, ML concepts, and coding patterns.
```
**Intent:** `interview_prep` | **Route:** `rag_only`
**Why rag only:** The internal corpus has comprehensive interview prep guides, ML concept documents, and LangGraph/RAG materials — no live web needed.

---

## Interview Preparation Queries

### 6. Behavioral Interview Prep (Uses: RAG)
```
I have a behavioral interview at a tech company next week.
Help me prepare STAR method answers and explain what questions to expect.
```
**Intent:** `interview_prep` | **Route:** `rag_only`

---

### 7. Technical Interview Deep Dive (Uses: BOTH)
```
Prepare me for a technical interview at a company known for hard ML system design questions.
What ML system design patterns should I know and how do I structure my answers?
```
**Intent:** `interview_prep` | **Route:** `both`

---

## Job & Company Research

### 8. European AI Internship Market Research (Uses: WEB or BOTH)
```
What AI engineering internships are currently available in Germany, Netherlands, and Sweden?
What skills do these companies typically require and how should I apply?
```
**Intent:** `job_research` | **Route:** `both`

---

### 9. Research Lab vs Startup Internship (Uses: BOTH)
```
Should I target AI research labs or AI startups for my first internship?
Compare the experience, compensation, and career impact of each path.
```
**Intent:** `job_research` | **Route:** `both`

---

## Technology Trend Queries

### 10. LangGraph and Agentic Frameworks (Uses: BOTH)
```
What is LangGraph and how does it compare to other agentic AI frameworks in 2025?
Which framework should I learn for a career in AI engineering?
```
**Intent:** `tech_trend` | **Route:** `both`

---

### 11. RAG Architecture Deep Dive (Uses: RAG)
```
Explain Retrieval-Augmented Generation in detail. How does it work, what are the components,
and what are its limitations?
```
**Intent:** `general_research` | **Route:** `rag_only`

---

## Profile & Portfolio Queries

### 12. LinkedIn Profile Review Guidance (Uses: RAG)
```
How should I optimize my LinkedIn profile as an AI/ML student looking for internships?
What sections matter most and what keywords should I include?
```
**Intent:** `career_plan` | **Route:** `rag_only`

---

### 13. GitHub Portfolio for AI Roles (Uses: RAG)
```
What projects should I build and how should I structure my GitHub profile
to stand out when applying to AI engineering roles?
```
**Intent:** `career_plan` | **Route:** `rag_only`

---

## Off-Topic (Rejected by Agent)

### 14. Off-Topic Query — Rejected
```
Give me a pasta carbonara recipe with step-by-step instructions.
```
**Intent:** `off_topic` | **Route:** `off_topic`
**Result:** Agent politely declines and explains scope.

---

## Running the Demo

```bash
# Single query mode (recommended for demos)
python main.py "Analyze my current skills: Python, JavaScript, SQL, YOLO, OpenCV, FastAPI, Supabase. What gaps do I have for AI Engineer roles?"

# Interactive mode
python main.py

# Rebuild vector store after adding new corpus documents
python main.py --ingest
```

---

## Expected Agent Behavior

| Query Type | Intent | Route | Corpus Used | Web Used |
|---|---|---|---|---|
| Career roadmap | career_plan | both | ✓ | ✓ |
| Interview prep | interview_prep | rag_only | ✓ | ✗ |
| Job listings | job_research | both | ✓ | ✓ |
| Skill gap | skill_gap | both | ✓ | ✓ |
| AI trend | tech_trend | both | ✓ | ✓ |
| Concept question | general_research | rag_only | ✓ | ✗ |
| Breaking news | general_research | web_only | ✗ | ✓ |
| Unrelated topic | off_topic | off_topic | ✗ | ✗ |
