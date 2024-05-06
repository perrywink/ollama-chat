import argparse

from embeds import get_embedding_function
from langchain.vectorstores.chroma import Chroma
from config import CHROMA_PATH, PROMPT_TEMPLATE
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama


def query_rag(query_text: str):
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )
    results = db.similarity_search(query=query_text, k=3)
    context_text = "\n\n---\n\n".join([doc.page_content for doc in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="llama2")
    response_text = model.invoke(prompt)

    sources = [doc.metadata.get("id", None) for doc in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


if __name__ == "__main__":
    main()
