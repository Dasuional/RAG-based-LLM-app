import os

from dotenv import load_dotenv
import pdf2image
import pytesseract
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

api_key = os.getenv('TOGETHERAI_API_KEY', '646603e61b8603a7cfe6acbebcdc5071a3ed1021a6447d6b282da99f40930579')

llm = ChatTogether(
    api_key=api_key,
    model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    )

file_path = 'documents.pdf'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe' 
@st.cache_resource
def load_pdf():
    loader = UnstructuredPDFLoader(file_path)
    documents = loader.load()
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

vectordb = load_pdf()

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
    llm=llm,
    chain_type='stuff',
    retriever=vectordb.vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
    input_key = 'question'
)

st.title('Ask AI...')

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

prompt = st.chat_input('pass your propmt here dumbfuck')

if prompt:
    st.chat_message('user').markdown(prompt)

    st.session_state.messages.append({'role':'user' , 'content':prompt})

    response = chain.run(prompt)

    # st.chat_message('assistant').markdown(response)

    # st.session_state.messages.append({'role':'assistant' , 'content':response})

    response_content = response.content

    response_content = response_content.strip()  

    st.chat_message('assistant').markdown(response_content)

    st.session_state.messages.append({'role': 'assistant', 'content': response_content})
