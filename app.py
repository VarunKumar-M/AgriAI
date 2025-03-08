import streamlit as st
import requests

# Streamlit UI
st.set_page_config(page_title="AgriGPT", page_icon="🌱", layout="wide")
st.title("🌱 AgriGPT - Your Agriculture Chatbot")
st.write("Ask anything about agriculture, farming, or crop management.")

# API Endpoint (Replace with your deployed FastAPI URL if running on cloud)
API_URL = "http://127.0.0.1:8000/ask"

# User Input
query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query:
        response = requests.post(API_URL, json={"question": query})

        if response.status_code == 200:
            answer = response.json().get("answer", "No response received.")
            st.write("**Answer:**")
            st.write(answer)
        else:
            st.error("Failed to get a response. Please try again later.")

    else:
        st.warning("Please enter a question.")

# Footer
st.markdown("---")
st.caption("🤖 Powered by FastAPI & Google Gemini 1.5 Flash")
