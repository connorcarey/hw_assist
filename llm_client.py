import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

from embedding import get_embedding_function

CHROMA_PATH = "chroma"

INSTRUCTIONS = """
You are a homework assistant tool.
You must provide the most relevant steps in order to complete the assignment. 
Do this in a easy-to-understand manner. You must outline EVERY step needed, 
even if this results in long output. Use relevant information from the document(s).
"""

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Follow the instructions based on the above context: {question}
"""


def main():
    # Create CLI for specific queries on documents.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)
#    response()

def response():
    return query_rag(INSTRUCTIONS)

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    # print(prompt)

    model = OllamaLLM(model="llama3.1:8b")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
