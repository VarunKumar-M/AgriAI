import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from retriever import retriever  # Importing TXT retriever
from langchain_google_genai import GoogleGenerativeAI  # Gemini AI
from googletrans import Translator  # Language translation

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Google API Key is missing. Set it in a .env file or as an environment variable.")

# Initialize AI & Translator
gemini = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)
translator = Translator()

# FastAPI App
app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(request: QueryRequest):
    query = request.question

    # Translate Query to English
    detected_lang = translator.detect(query).lang
    if detected_lang != "en":
        query = translator.translate(query, src=detected_lang, dest="en").text

    # Retrieve Relevant Text
    retrieved_text = retriever.retrieve_relevant_text(query)

    # Construct AI Prompt
    prompt = f"""
    You are an expert in agriculture. Your responses should follow these rules:

    1. **Strictly factual for agriculture-related topics**: If the question is about **farming, crops, soil, irrigation, pesticides, fertilizers, livestock, agricultural marketing, or government schemes**, provide **precise, expert-level responses**.
    2. **Allow conversational flexibility**: If the user asks a **behavioral, opinion-based, or general question** (e.g., farming experiences, personal opinions, ethical farming), respond freely with a natural, engaging, and conversational tone.
    3. **Reject completely off-topic questions**: If the question is entirely **unrelated to agriculture and not behavioral**, refuse to answer politely by saying:

       *"I'm designed to focus on agriculture-related topics. However, if you want to discuss farming, crops, soil health, or government schemes, I'm happy to help!"*

    **User's Question:** {query}

    **Relevant Context from Documents:**  
    {retrieved_text}

    **Instructions:**  
    - If the question is **agriculture-related**, provide a **clear and expert** response.  
    - If the user asks for **detailed information**, provide an in-depth answer.  
    - If the question is **about farming behavior, experiences, or general life**, be free-minded and conversational.  
    - If the question is **completely unrelated**, politely refuse to answer.  
    """

    # Get response from Gemini AI
    try:
        response = gemini.invoke(prompt)
        if not response:
            response = "I'm not sure, but you can ask about farming techniques, soil health, or crop management."
    except Exception:
        response = "I'm currently unable to process your request. Please try again later."

    # Translate response back to original language
    if detected_lang != "en":
        response = translator.translate(response, src="en", dest=detected_lang).text

    return {"answer": response}
