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
    detected_lang = translator.detect(query).lang
    if detected_lang != "en":
        query = translator.translate(query, src=detected_lang, dest="en").text
    return query, detected_lang

# Function: Translate Response Back to User's Language
def translate_response(response, target_lang):
    if target_lang != "en":
        response = translator.translate(response, src="en", dest=target_lang).text
    return response

# User Input
query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        translated_query, user_lang = translate_to_english(query)  # Auto-translate query to English
        retrieved_text = retriever.retrieve_relevant_text(translated_query)  # Retrieve relevant text

        # Construct AI Prompt
        prompt = f"""
        You are an expert in agriculture. Answer the following question in simple, accurate language.

        **User Question:** {translated_query}

        **Instructions:**  
        - Give a **clear, concise** response.  
        - Provide **detailed explanations** only if the user explicitly asks for it.  
        - If unsure, give a general agricultural answer instead of leaving it blank.  
        """

        # Get response from Gemini AI
        try:
            response = gemini.invoke(prompt)
            if not response:
                response = "I'm not sure, but you can ask about farming techniques, soil health, or crop management."
        except Exception:
            response = "I'm currently unable to process your request. Please try again later."

        # Translate response back to user's original language
        final_response = translate_response(response, user_lang)

        st.write("**Answer:**")
        st.write(final_response)

    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.caption("🤖 Powered by Google Gemini 1.5 Flash & Local Document Retrieval")
