import json
import os
from dotenv import load_dotenv
from groq import Groq

from roadmap_prompt import build_roadmap_prompt

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

with open("../data/mock_ranked_issues.json") as f:
    issues = json.load(f)

contributor_skills = ["testing", "REST API"]

issue = issues[0]
messages = build_roadmap_prompt(issue, contributor_skills)

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=messages,
    temperature=0.4,
)

print(f"Issue: {issue['title']}\n")
print("Generated roadmap:\n")
print(response.choices[0].message.content)