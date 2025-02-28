from flask import Flask, render_template, jsonify, request
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from pinecone import Pinecone, ServerlessSpec

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')  # Get Gemini API key

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"

# Check if the index exists. If not, create it.
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Gemini setup
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro-latest")

system_prompt = (
    "You are an AI assistant for medical question-answering tasks. "
    "Use the retrieved context to provide accurate answers. "
    "If you don't know the answer, state that clearly. "
    "Keep the response within three sentences."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

def query_gemini(context, input_text):
    full_prompt = system_prompt.format(context=context) + "\n" + input_text
    response = model.generate_content(full_prompt)
    return response.text

def rag_chain(retriever, input_text):
    retrieved_docs = retriever.get_relevant_documents(input_text)
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    return query_gemini(context, input_text)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    response = rag_chain(retriever, msg)  # Use the RAG chain
    print("Response : ", response)
    return str(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
