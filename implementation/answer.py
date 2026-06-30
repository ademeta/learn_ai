import os
from pathlib import Path
from dotenv import load_dotenv
import yaml
from jinja2 import Template

from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, convert_to_messages
from langchain.documents import Document

load_dotenv()

# ingest file location
here = Path(__file__).resolve().parent
# project path 
proj_path = Path(__file__).resolve().parents[1]

# embedding
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# load config
import yaml
with open(os.path.join(proj_path, "config.yml"), "r") as f:
    raw_yml = f.read()
config = yaml.safe_load(raw_yml)


db_path = os.path.join(proj_path, config["DB_NAME"])
system_prompt = config["SYSTEM_PROMPT"]

# load in saved vector collections
vectorstore = Chroma(persist_directory=db_path, 
                     embedding_function=embedding_function)
retriever = vectorstore.as_retriever()

# llm
llm = ChatOpenAI(temperature=0, model_name=config["MODEL"])

def fetch_context(query: str) -> list[Document]:
    return retriever.invoke(query, k=config["RETRIEVAL_K"])


def combined_question(query: str,
                     history: list[dict] = []) -> str:
    """
    combine all the user's message into a single string
    """
    prior_questions = "\n".join(m["content"] for m in history if m["role"] =="user")
    return prior_questions + "\n" + query


def answer_question(question: str, 
                    history: list[dict] = []):
    """
    Answer questions and have knowledge of previosuly asked questions.
    """
    combined = combined_question(question, history)
    docs = fetch_context(combined)
    context= "\n\n".join(doc.page_content for doc in docs)
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
    messages = [SystemMessage(content=system_prompt)]
    messages.extend(convert_to_messages(history))
    messages.append(HumanMessage(content=question))
    response = llm.invoke(messages)
    return response.content

if __name__ == "__main__":
    pass