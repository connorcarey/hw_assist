from langchain_aws.embeddings.bedrock import BedrockEmbeddings

def get_embedding_function():
    """
    Return embedding function to be used throughout project.
    Currently using AWS Bedrock.

    Returns:
        function: The embedding function to be used.
    """
    embeddings = BedrockEmbeddings(
        credentials_profile_name="default",
        region_name="us-east-2" # closest AWS server (Ohio) to College Station
    )

    return embeddings
