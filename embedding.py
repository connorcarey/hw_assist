# from langchain_aws.embeddings.bedrock import BedrockEmbeddings
#
# def get_embedding_function():
#     """
#     Return embedding function to be used throughout project.
#     --Currently using AWS Bedrock. 
#
#     Returns:
#         function: The embedding function to be used.
#     """
#     embeddings = BedrockEmbeddings(
#         credentials_profile_name="default",
#         region_name="us-east-2" # closest AWS server (Ohio) to College Station
#     )
#
#     return embeddings

# from langchain_community.embeddings.ollama import OllamaEmbeddings
#
# def get_embedding_function(): # Too slow, not going to run this locally...
#     embeddings = OllamaEmbeddings(model="nomic-embed-text") 
#     return embeddings

from langchain_openai import OpenAIEmbeddings

def get_embedding_function():
    """
    Return embedding function to be used throughout project.
    --Currently using OpenAI because I have credits :P 

    Returns:
        function: The embedding function to be used.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    return embeddings

