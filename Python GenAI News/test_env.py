import os
from dotenv import load_dotenv

from langchain_community.llms import OpenAI

load_dotenv()  # take environment variables from .env (especially openai api key)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"Your API Key: {OPENAI_API_KEY}")


llm = OpenAI(temperature=0.1,model="gpt-4-turbo")