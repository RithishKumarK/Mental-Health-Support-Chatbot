import os
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
import gradio as gr

# Initialize LLM
def initialize_llm():
    return ChatGroq(
        temperature=0,
        groq_api_key=os.environ.get("GROQ_API_KEY"),  # safer: set as env variable
        model_name="llama-3.3-70b-versatile"
    )

def create_vector_db():
    loader = DirectoryLoader(
        "data",  
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vector_db = Chroma.from_documents(texts, embeddings, persist_directory="vectorstore")
    vector_db.persist()
    print("ChromaDB created successfully!")
    return vector_db

def setup_qa_chain(vector_db, llm):
    retriever = vector_db.as_retriever()

    prompt = ChatPromptTemplate.from_template("""
You are a compassionate mental health assistant.
Use the following context to answer the user calmly and supportively.

Context:
{context}

User question:
{question}

Helpful response:
""")

    chain = (RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
             | prompt
             | llm)
    return chain


def chat_response(user_input, history=[]):
    if not user_input.strip():
        return "Please provide a valid input.", history
    response = qa_chain.invoke(user_input)
    history.append((user_input, response.content))
    return "", history


llm = initialize_llm()
if not os.path.exists("vectorstore"):
    vector_db = create_vector_db()
else:
    embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory="vectorstore", embedding_function=embeddings)

qa_chain = setup_qa_chain(vector_db, llm)


with gr.Blocks(theme="Nymbo/Nymbo_Theme") as app:
    chatbot = gr.ChatInterface(fn=chat_response, title="Mental Health Chatbot")

if __name__ == "__main__":
    app.launch(share=True)  
