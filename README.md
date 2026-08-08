# OpenMentor

An open-source contribution recommender system. OpenMentor ranks GitHub 
issues by difficulty and skill-match for a given contributor, then 
generates a personalized roadmap and plain-language explanation for each 
recommendation using an LLM.

## Team & Roles

- **Member 1** — Difficulty Scoring & Skill Matching Engineer
- **Member 2** — LLM Integration, Interface & Testing Lead
- **Member 3** — Repository & Dependency Graph Engineer

## Project Structure
openmentor/
├── data/ # Shared datasets: graph metrics, issue data, ranked outputs
├── scoring/ # Difficulty scoring & skill-matching logic (Member 1)
├── llm/ # Prompt templates & roadmap generation (Member 2)
├── interface/ # Streamlit demo interface (Member 2)
├── shared/ # Agreed data schemas used across all three roles
└── tests/ # Tests

## Data Contract

See `shared/schemas.md` for the exact shape of data passed between stages:

`graph_metrics.csv` + `issues.csv` (Member 3) → `ranked_issues.json` 
(Member 1) → roadmap + explanation generation (Member 2)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:
GROQ_API_KEY=your_key_here

## Running the LLM roadmap generator (demo, uses mock data)

```bash
cd llm
python generate_roadmap.py
```