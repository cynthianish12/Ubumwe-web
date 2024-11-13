import os
import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware
import logging

# Load environment variables
load_dotenv()

# Set up logging for better error tracking
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Configure CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Adjust this if necessary
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load documents
try:
    loader = TextLoader('data.txt')
    documents = loader.load()
except Exception as e:
    logger.error(f"Error loading documents: {str(e)}")
    raise HTTPException(status_code=500, detail="Error loading documents.")

# Split documents
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
splitted_docs = text_splitter.split_documents(documents)

# Create embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

# Create and save vector store with error handling
try:
    vector_store = FAISS.from_documents(splitted_docs, embeddings)
    vector_store.save_local('index')
except Exception as e:
    logger.error(f"Error creating FAISS vector store: {str(e)}")
    raise HTTPException(status_code=500, detail="Error creating vector store.")

# Load vector store
try:
    vector_store = FAISS.load_local('index')
except Exception as e:
    logger.error(f"Error loading vector store: {str(e)}")
    raise HTTPException(status_code=500, detail="Error loading vector store.")

# Request model
class QueryRequest(BaseModel):
    query: str

# Token tracker function
def track_token_usage(response):
    usage = response['usage']
    prompt_tokens = usage.get('prompt_tokens', 0)
    completion_tokens = usage.get('completion_tokens', 0)
    total_tokens = usage.get('total_tokens', 0)
    logger.info(f"Prompt tokens: {prompt_tokens}, Completion tokens: {completion_tokens}, Total tokens: {total_tokens}")
    return total_tokens

# Generate response function
def generate_response(query: str) -> str:
    docs = vector_store.similarity_search(query)

     # If no relevant documents are found, respond with a fallback message
    if not docs:
        return (
            "I'm sorry, I can only assist with topics related to gender equality, violence reporting, "
            "USAID’s support for women’s health, family issues, child protection, food security, and early childhood development. "
            "How can I assist you with one of these?"
        )
        
    combined_content = "\n\n".join([doc.page_content for doc in docs])

    response = openai.Completion.create(
        engine="gpt-3.5-turbo",  # Update to a more recent model if needed
        prompt=f"Answer the following question based on the provided information:\n\n{combined_content}\n\nQuestion: {query}\n\nAnswer:",
        max_tokens=200
    )

    # Track token usage
    total_tokens_used = track_token_usage(response)
    logger.info(f"Total tokens used for this request: {total_tokens_used}")
    
    return response.choices[0].text.strip()

# FastAPI endpoint for querying
@app.post("/query")
async def get_response(query_request: QueryRequest):
    logger.info(f"Received query: {query_request.query}")  # Log the request to console
    query = query_request.query
    try:
        response = generate_response(query)
        return {"response": response}
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        raise HTTPException(status_code=500, detail="Error generating response.")

# Run FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
