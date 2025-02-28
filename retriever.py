import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # Using Gemini embeddings

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
DATA_DIR = "data/"  # Folder containing TXT files

if not GEMINI_API_KEY:
    raise ValueError("Google API Key is missing. Set it in a .env file or as an environment variable.")

class TXTDataRetriever:
    def __init__(self):
        self.vectorstore = None
        self.load_data()

    def load_data(self):
        """Loads text files, splits into chunks, and stores embeddings in FAISS."""
        if not os.path.exists(DATA_DIR):
            raise RuntimeError(f"Data directory '{DATA_DIR}' not found.")

        documents = []
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GEMINI_API_KEY)

        # Load all TXT files in the data folder
        for filename in os.listdir(DATA_DIR):
            if filename.endswith(".txt"):
                filepath = os.path.join(DATA_DIR, filename)
                loader = TextLoader(filepath)
                try:
                    docs = loader.load()
                    split_docs = text_splitter.split_documents(docs)
                    documents.extend(split_docs)
                except Exception as e:
                    print(f"⚠️ Error loading {filename}: {e}")

        if not documents:
            raise RuntimeError("❌ No valid text data found. Make sure .txt files exist in the 'data/' folder.")

        # Store in FAISS vector database
        self.vectorstore = FAISS.from_documents(documents, embeddings)

    def retrieve_relevant_text(self, query):
        """Retrieves relevant text from FAISS using similarity search."""
        if not self.vectorstore:
            raise RuntimeError("Vector store is not initialized.")

        results = self.vectorstore.similarity_search(query, k=3)  # Retrieve top 3 matches
        
        if not results:
            return "No relevant information found in the dataset."

        return "\n\n".join([doc.page_content for doc in results])

# Initialize retriever instance once
retriever = TXTDataRetriever()
