import os
from dotenv import load_dotenv
from groq import Groq

# Load the .env file so GROQ_API_KEY becomes available via os.getenv
load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant for an open-source contribution recommender system."
        },
        {
            "role": "user",
            "content": "In one sentence, explain what a 'good first issue' typically looks like."
        }
    ],
    temperature=0.7,
)

print(response.choices[0].message.content)