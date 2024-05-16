# Welcome to Cloud Functions for Firebase for Python!
# To get started, simply uncomment the below code or create your own.
# Deploy with `firebase deploy`

from firebase_functions import https_fn
from firebase_admin import initialize_app


from typing import Any
from flask import jsonify
from waitress import serve
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.vectorstores import DocArrayInMemorySearch
from IPython.display import display, Markdown
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.indexes import VectorstoreIndexCreator
from langchain_experimental.agents.agent_toolkits.csv.base import create_csv_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
import tiktoken
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import Chroma
import os
from firebase_admin import credentials, storage, db, firestore
from langchain_anthropic import ChatAnthropic
from getpass import getpass

import git

# URL of the Git repository to clone
repo_url = 'https://Beloved1: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@github.com/Beloved1/Data'

# git.Repo.clone_from(repo_url, './Data')

# Set the API key as an environment variable
os.environ["OPENAI_API_KEY"] = "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"


# def load_and_process_pdfs(folder_path):
#     documents = []
#     for file in os.listdir(folder_path):
#           file_path = os.path.join(folder_path, file)
#           if not file.endswith('.git'):
#             print(file_path)
#             loader = UnstructuredFileLoader(file_path)
#             documents.extend(loader.load())

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     splits = text_splitter.split_documents(documents)
#     return splits

# folder_path = "./Data"
# splits = load_and_process_pdfs(folder_path)


persist_directory = 'db'


## here we are using OpenAI embeddings but in future we will swap out to local embeddings
embedding = OpenAIEmbeddings()

vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

anthropic_api_key = "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"

# chat completion llm
llm = ChatAnthropic(
    anthropic_api_key=anthropic_api_key,
    model_name="claude-3-sonnet-20240229",  # change "opus" -> "sonnet" for speed
    temperature=0.0
)

memory = ConversationBufferMemory(
memory_key='chat_history', return_messages=True)
conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(),
        memory=memory
        )

def get_answer(query):
  result = conversation_chain({"question": query})
  return result["answer"]


@https_fn.on_call()
def test(req: https_fn.CallableRequest) -> Any:
    try:
        query_text = req.data['query']
    except KeyError:
        # Throwing an HttpsError so that the client gets the error details.
        raise https_fn.HttpsError(code=https_fn.FunctionsErrorCode.INVALID_ARGUMENT,
                                    message=('The function must be called with one argument, "query",'
                                        " containing the message request."))
    response_text = get_answer(query_text)
    return jsonify({'response': response_text})
