import os
from dotenv import load_dotenv
from google import genai

# Load .env file
load_dotenv()

# Read API key from environment
api_key = os.getenv("GEMINI_API_KEY")

# Create Gemini client
client = genai.Client(api_key=api_key)

# Your prompt
prompt = "Explain what an API key is in 2 simple lines."

# Send prompt to Gemini
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

# Print model response
print(response.text)