import os
import openai
from dotenv import load_dotenv
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()

# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Load the index
vector_store = FAISS.load_local('index')

def generate_response(query: str, model: str = "davinci") -> str:
    """
    Generate a response to the provided query using the OpenAI API.

    Parameters:
    - query (str): The question or query to be answered.
    - model (str): The OpenAI model to use for generating the response.

    Returns:
    - str: The generated response from the OpenAI API.
    """
    try:
        docs = vector_store.similarity_search(query)
        combined_content = "\n\n".join([doc.page_content for doc in docs])
        
        response = openai.Completion.create(
            engine=model,
            prompt=f"Answer the following question based on the provided information:\n\n{combined_content}\n\nQuestion: {query}\n\nAnswer:",
            max_tokens=200
        )
        return response.choices[0].text.strip()

    except Exception as e:
        print(f"Error generating response: {str(e)}")
        return "An error occurred while generating the response."

# Example query execution
if __name__ == "__main__":
    query = input("Enter your query: ")  # Allow user input for flexibility
    response = generate_response(query)
    print(response)
