import os
import streamlit as st
from dotenv import load_dotenv
from retriever import retriever  # Importing TXT retriever
from langchain_google_genai import GoogleGenerativeAI  # Gemini AI for final response
from googletrans import Translator  # Auto-detect & translate languages

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("Google API Key is missing. Set it in a .env file or as an environment variable.")

# Initialize Gemini AI
gemini = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY)

# Initialize Translator
translator = Translator()

# Streamlit UI
st.set_page_config(page_title="AgriGPT", page_icon="🌱", layout="wide")
st.title("🌱 AgriGPT - Your Agriculture Chatbot")
st.write("Ask anything about agriculture, farming, or crop management.")

# Function: Detect & Translate Query to English
def translate_to_english(query):
    try:
        detected_lang = translator.detect(query).lang
        if detected_lang != "en":
            query = translator.translate(query, src=detected_lang, dest="en").text
        return query, detected_lang
    except Exception:
        return query, "en"  # Default to English if translation fails

# Function: Translate Response Back to User's Language
def translate_response(response, target_lang):
    try:
        if target_lang != "en":
            response = translator.translate(response, src="en", dest=target_lang).text
        return response
    except Exception:
        return response  # Return original response if translation fails

# Function: Get AI Response
def get_agri_response(query):
    prompt = f"""
    You are an expert in agriculture and must **only** answer agriculture-related questions.
    If the user's question is **not** about agriculture, farming, soil, crops, irrigation, pesticides, fertilizers, livestock, 
    agricultural marketing, or government schemes for farmers, then **refuse to answer** by saying:

    "I'm designed to answer only agriculture-related questions. Please ask about farming, crops, soil health, or government schemes."

    **User Question:** {query}

    **Instructions:**  
    - If the question is **related to agriculture**, provide a **clear, accurate, and concise** response.  
    - If the user asks for **detailed information**, give an in-depth response.  
    - If the question is **unrelated to agriculture**, politely refuse to answer.  
    - Do **not generate any response outside agriculture topics**.  
    """

    try:
        response = gemini.invoke(prompt)
        return response if response else "I'm not sure, but you can ask about farming techniques, soil health, or crop management."
    except Exception:
        return "I'm currently unable to process your request. Please try again later."

# User Input
query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        with st.spinner("Processing..."):  # Show loading indicator
            translated_query, user_lang = translate_to_english(query)  # Auto-translate query to English
            retrieved_text = retriever.retrieve_relevant_text(translated_query)  # Retrieve relevant text
            
            response = get_agri_response(translated_query)
            final_response = translate_response(response, user_lang)  # Translate back to user's language

        st.write("**Answer:**")
        st.write(final_response)
    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.caption("🤖 Powered by Google Gemini 1.5 Flash & Local Document Retrieval")
