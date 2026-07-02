import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.llms import Anthropic
import google.generativeai as genai

CLAUDE_PROMPT = """
Answer ONLY from the context below.
If you don't know, say "I don't know".

Context:
{context}

Question:
{question}
"""

GEMINI_PROMPT = CLAUDE_PROMPT


def claude_generate(context, question):
    llm = Anthropic(temperature=0)
    prompt = CLAUDE_PROMPT.format(context=context, question=question)
    return llm.invoke(prompt)


def gemini_generate(context, question):
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return "Gemini API key missing."

    genai.configure(api_key=api_key)

    model = genai.GenerativeModel("gemini-1.5-flash")
    prompt = GEMINI_PROMPT.format(context=context, question=question)

    response = model.generate_content(prompt)

    return response.text if hasattr(response, "text") else str(response)