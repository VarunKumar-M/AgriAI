import os
import logging
from fastapi import FastAPI, HTTPException, Request
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

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    question: str

@app.get("/")
async def root():
    return {"message": "AgriAI API is running. Use /ask endpoint to interact."}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    query = request.question
    logger.info(f"Received question: {query}")

    # Translate Query to English
    detected_lang = translator.detect(query).lang
    if detected_lang != "en":
        query = translator.translate(query, src=detected_lang, dest="en").text
        logger.info(f"Translated question to English: {query}")

    # Retrieve Relevant Text
    retrieved_text = retriever.retrieve_relevant_text(query)
    logger.info(f"Retrieved relevant text: {retrieved_text[:200]}...")  # Log first 200 chars

    # Construct AI Prompt
    prompt = f"""
    You are an expert in agriculture. Your responses should follow these rules:

    1. **Strictly factual for agriculture-related topics**: If the question is about **farming, crops, soil, irrigation, pesticides, fertilizers, livestock, agricultural marketing, or government schemes**, provide **precise, expert-level responses**.
    2. **Allow conversational flexibility**: If the user asks a **behavioral, opinion-based, or general question** (e.g., farming experiences, personal opinions, ethical farming), respond freely with a natural, engaging, and conversational tone.
    3. **Reject completely off-topic questions**: If the question is entirely **unrelated to agriculture and not behavioral**, refuse to answer politely.

    **User's Question:** {query}
    **Relevant Context from Documents:**  
    {retrieved_text}
    """

    # Get response from Gemini AI
    try:
        response = gemini.invoke(prompt)
        if not response:
            response = "I'm not sure, but you can ask about farming techniques, soil health, or crop management."
        logger.info(f"Generated AI response: {response[:200]}...")
    except Exception as e:
        logger.error(f"Error generating AI response: {e}")
        raise HTTPException(status_code=500, detail="AI processing error. Try again later.")

    # Translate response back to original language
    if detected_lang != "en":
        response = translator.translate(response, src="en", dest=detected_lang).text
        logger.info(f"Translated response back to {detected_lang}: {response[:200]}...")

    return {"answer": response}

# Handle unsupported methods
@app.api_route("/ask", methods=["GET", "PUT", "DELETE"])
async def unsupported_method(request: Request):
    raise HTTPException(status_code=405, detail="Method Not Allowed. Use POST instead.")
