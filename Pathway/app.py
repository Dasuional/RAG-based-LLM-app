import os

from dotenv import load_dotenv
# from langchain.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_together import TogetherEmbeddings
# from langchain.chains import RetrievalQA
from langchain_together import ChatTogether
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

import streamlit as st

# load the environment variables
load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

def load_document(file_path):
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
    return documents

def setup_vectorstore(documents):
    embeddings = TogetherEmbeddings(
    model="togethercomputer/m2-bert-80M-8k-retrieval",
    # device="cuda"
    )
    text_splitter = RecursiveCharacterTextSplitter(
        # separator="/n",
        chunk_size=250,
        chunk_overlap=0
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = InMemoryVectorStore.from_texts(
    doc_chunks,
    embedding=embeddings,
    # device='cuda'
    )
    return vectorstore

def create_chain(vectorstore):
    LLM = ChatTogether(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    )
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=LLM,
        output_key="answer",
        memory_key="chat_history",
        return_messages=True
    )

    template = """You are a highly knowledgeable assistant with access to a comprehensive geographical database.
    The database contains detailed information about
    zip codes, type, primary_city, acceptable_cities, unacceptable_cities,
    states, county, timezone, country,
    of various locations.

    Given the following query and the relevant information from the database:

    {context}
    Question: {question}

    Extract and provide the relevant details from the context, such as the city, state, county, timezone,
    and any other relevant information. If specific details are unavailable in the context, respond accordingly.

    Please provide a detailed response using the available data.
    Helpful Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    chain = ConversationalRetrievalChain.from_llm(
        llm=LLM,
        retriever=retriever,
        chain_type="map_reduce",
        memory=memory,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        verbose=True
    )
    return chain

st.set_page_config(
    page_title="Chat with Doc",
    page_icon="📄",
    layout="centered"
)

st.title("🤖✨ Chat with OWN_DOC -Pathway/LangChain")

# initialize the chat history in streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader(label="Upload your pdf file", type=["pdf"])

if uploaded_file:
    file_path = f"{working_dir}/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())


    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = setup_vectorstore(load_document(file_path))

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask AI...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)


    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": user_input})
        assistant_response = response["answer"]
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
