RAG-Based Real Estate LLM App

This project is a Retrieval-Augmented Generation (RAG) based Language Learning Model (LLM) application built using Langchain and Pathway frameworks. It helps users retrieve real estate data and ask questions related to various real estate properties and locations. The application is deployed using Streamlit.

Features

    1)Real-time querying of real estate data.
    2)Integration with large language models (LLMs) to generate insightful answers.
    3)Use of vector-based search for efficient retrieval.
    4)Built with Langchain's document loaders and Pathway for LLM-based processing.
  
Table of Contents

    1) Features
    2) Tech Stack
    3) Setup & Installation
    4) Usage
    5) How it works
    6) Future Improvements
    7) Contributing
    8) License
    
Tech Stack

    1) Python: Main language for the backend logic.
    2) Langchain: For chaining LLMs with retrieval systems.
    3) Pathway: Used for integrating LLMs.
    4) Streamlit: Web framework for building the user interface.
    5) Vector Store: In-memory vector database for fast search.
    
Setup & Installation
 1) Clone the repository:
   
        git clone https://github.com/your-username/real-estate-rag-llm.git
        cd real-estate-rag-llm
   
 2) Install the necessary dependencies: Make sure you have Python 3.8+ installed. Install the dependencies listed in requirements.txt:

        pip install -r requirements.txt

 3) Set environment variables: Create a .env file in the root directory to store your API keys and other secrets:

        makefile
        TOGETHERAI_API_KEY=your_api_key_here
        
 4) Run the application: After setting up everything, you can start the Streamlit app by running:

        bash
        streamlit run src/main.py
   
Usage

    1) Open the Streamlit app in your browser at http://localhost:8501.
    2) Enter your real estate-related queries (e.g., "What properties are available in San Francisco?").
    3) The app will retrieve relevant data from the vector store and generate a response using the LLM.
    
How it works

    1) Document Loading: Real estate data is loaded from a dataset and split into chunks using RecursiveCharacterTextSplitter to be efficiently stored in the vector store.
    2) Embedding: The chunks are embedded into vectors using a model such as SentenceTransformer, and stored in an InMemoryVectorStore.
    3) Querying: When a user inputs a query, it is matched with the vector embeddings, and relevant documents are retrieved.
    4) LLM Integration: The retrieved documents are passed to the language model, which generates a detailed response based on the query and the context.
    
Example Queries

    1) "Show me houses in California under $500,000."
    2) "What is the average price of homes in New York?"
    3) "List properties with 3 bedrooms in Texas."
    
Future Improvements

    1) Add support for more real estate data sources.
    2) Deploy the app on a cloud platform (e.g., Heroku or AWS).
    3) Improve the retrieval system for better accuracy with large datasets.
    4) Implement user authentication for personalized data access.
    
Contributing
Contributions are welcome! Please feel free to submit a Pull Request or file an issue if you have suggestions or bug reports.




