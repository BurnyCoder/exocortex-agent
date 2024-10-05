import os
from dotenv import load_dotenv
import logging

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


from langchain_community.embeddings import OpenAIEmbeddings
from pinecone import Pinecone
from openai import OpenAI

client = OpenAI()
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Database hyperparameters
index_name = "exocortex"
dimension = 1536


db = None


def generate_embeddings(text):
    return embeddings.embed_query(text)  # Updated method name


def initialize():
    # Initialize Pinecone client with corrected import
    pinecone_client = Pinecone(
        api_key=PINECONE_API_KEY, environment="us-west-2"
    )  # Updated to use PineconeClient for initialization
    db = pinecone_client.Index(index_name)  # Updated to use PineconeClient for index operations

    return db


def retrieve(query, top_k=5, return_vectors=True, return_in_string=False, retrieve_all=False): 
    """
    Retrieves the top_k most similar vectors to the query or all vectors if retrieve_all is True, optionally excluding vector values from the result and returning the result as a string. When not retrieving all, vector number values are not returned.

    Args:
    query (str): The query string to retrieve similar vectors for.
    top_k (int): The number of top similar vectors to retrieve. Ignored if retrieve_all is True.
    return_vectors (bool): Whether to include vector values in the result. Defaults to True.
    return_in_string (bool): Whether to return the results as a formatted string. Defaults to False.
    retrieve_all (bool): Whether to retrieve all vectors. Defaults to False.

    Returns:
    dict or str: A dictionary with vector IDs as keys and their corresponding metadata (and optionally vectors) as values, or a formatted string if return_in_string is True.
    """
    if not retrieve_all:
        try:
            query_vector = generate_embeddings(query)
            search_results = db.query(vector=query_vector, top_k=top_k, include_metadata=True, include_values=return_vectors)
            if not search_results:
                raise ValueError("Query did not return any results.")
            formatted_results = search_results
            return formatted_results
        except Exception as e:
            logging.error(f"Error during quering database execution: {e}")
            return "Error during quering database execution"
    else:
        try:
            # Fetch the current number of vectors in the Pinecone database to determine the range of IDs
            index_stats = db.describe_index_stats()
            vector_count = index_stats['total_vector_count']

            # Fetch all vectors by their IDs
            fetched_vectors = db.fetch(ids=[str(i) for i in range(vector_count)])['vectors']
            formatted_vectors = search_results
            return formatted_vectors
        except Exception as e:
            logging.error(f"Error in fetching all vectors from Pinecone database: {e}")
            return "Error in fetching all vectors"