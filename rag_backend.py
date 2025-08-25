import json
import requests
import re
import time
import os
from dotenv import load_dotenv

# --- LangChain & PDF Processing Imports ---
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate

# Load environment variables from a .env file
load_dotenv()

# --- RAG Functions ---

# --- RAG Functions ---
def load_and_index_documents(docs_directory="docs"):
    """
    Loads documents from a directory, splits them into chunks, and indexes them
    into a ChromaDB vector store. This version saves the embeddings to disk.
    """
    # Define the directory where the vector store will be saved
    persist_directory = "./chroma_db"
    
    # Create an embedding model
    # Note: Replace with a Google model if you prefer a different embedding
    # service, e.g., 'GoogleGenerativeAIEmbeddings'
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Check if the vector store already exists
    if os.path.exists(persist_directory) and os.listdir(persist_directory):
        print("Vector store already exists. Loading from disk...")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        print("Vector store loaded successfully.")
    else:
        print("Vector store not found. Indexing documents...")
        
        # Load documents from the 'docs' directory
        documents = []
        for file_name in os.listdir(docs_directory):
            if file_name.endswith(".pdf"):
                file_path = os.path.join(docs_directory, file_name)
                loader = PyMuPDFLoader(file_path)
                documents.extend(loader.load())

        if not documents:
            print("No PDF files found in the 'docs' directory.")
            return None

        print(f"Loaded {len(documents)} pages from PDFs.")

        # Split documents into chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Split documents into {len(chunks)} chunks.")

        # Index the chunks into ChromaDB and persist to disk
        print("Indexing documents into ChromaDB...")
        vector_store = Chroma.from_documents(
            documents=chunks, 
            embedding=embeddings,
            persist_directory=persist_directory
        )
        vector_store.persist()
        print("Indexing complete and saved to disk.")
    
    return vector_store.as_retriever()


def call_gemini_api(prompt):
    """
    Makes an actual API call to the Gemini API with exponential backoff for retries.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "API key not found. Please set the GEMINI_API_KEY environment variable."

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key=" + api_key
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}]
    }
    
    max_retries = 3
    delay = 1  # Initial delay in seconds

    for i in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429 and i < max_retries - 1:
                print(f"Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2
            else:
                print(f"HTTP error occurred: {e}")
                return "An error occurred while connecting to the assistant. Please try again."
        except requests.exceptions.RequestException as e:
            print(f"API call failed: {e}")
            return "An error occurred while connecting to the assistant. Please try again."

    return "Max retries exceeded. Unable to connect to the assistant."

def generate_rag_response(query: str, retriever) -> str:
    """
    Generates a response using the RAG approach with the vector store retriever.
    """
    # Step 1: Retrieval
    relevant_docs = retriever.get_relevant_documents(query)
    
    if not relevant_docs:
        return "I'm sorry, I couldn't find any relevant information in the documents."

    # Combine retrieved documents into a single string
    context = "\n---\n".join([doc.page_content for doc in relevant_docs])

    # Step 2: Prompt Construction
    prompt_template = PromptTemplate.from_template("""
    You are a helpful and friendly internal company assistant. Use the following company documentation to answer the user's question. 
    If you cannot find the answer in the provided documentation, politely state that you don't have that information.

    --- Company Documentation ---
    {context}
    --- End of Documentation ---

    User's question: {query}
    
    Your response should be based *only* on the provided documentation. Do not invent information.
    """)

    prompt = prompt_template.format(context=context, query=query)

    # Step 3: Call the LLM
    return call_gemini_api(prompt)

# Example usage (for testing purposes)
if __name__ == '__main__':
    # Initialize the vector store and retriever once at the start of your application
    retriever_instance = load_and_index_documents()
    flag = True

    if retriever_instance:
        while flag:
            test_query = input("Q:")
            print(f"\nUser Query: {test_query}")
            if test_query == 'q':
                flag = False
                exit()
            response = generate_rag_response(test_query, retriever_instance)
            print(f"Assistant Response: {response}")

       