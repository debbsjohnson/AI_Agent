import os
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.embeddings.openai import OpenAIEmbedding

from dotenv import load_dotenv

env_file = os.getenv('ENV_FILE', '/Users/mac/PycharmProjects/RAG/.env')
load_dotenv(env_file)


def get_index(data, index_name):
    index = None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("No API key found for OpenAI. Please set the OPENAI_API_KEY environment variable.")

    if not os.path.exists(index_name):
        print("building index: ", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True, embed_model=OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY")))
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(StorageContext.from_defaults(persist_dir=index_name, embed_model=OpenAIEmbedding(api_key=os.getenv("OPENAI_API_KEY"))))

    return index


pdf_path = os.path.join("data", "uk.pdf")

if not os.path.exists(pdf_path):
    print(f"Error: File {pdf_path} does not exist.")
else:
    print(f"Loading file from {os.path.abspath(pdf_path)}")
    uk_pdf = SimpleDirectoryReader(input_files=[pdf_path]).load_data()
    uk_index = get_index(uk_pdf, "uk")
    uk_engine = uk_index.as_query_engine()


