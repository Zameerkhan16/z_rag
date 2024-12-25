import streamlit as st
from langchain.document_loaders import UnstructuredURLLoader,PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

st.title("OdinSchool Rag based chatbot")
st.sidebar.header("Configuration related to OpenAI")
api_key = st.sidebar.text_input("OpenAI Key Required", type="password")

st.header("Please ask any query related to Odinschool's website")
question = st.text_input("Enter your question")

if api_key:
    try:
        # Load the urls and process data
        # URLs=["https://www.odinschool.com"]
        URLs=["https://fullstackacademy.in"]
        loaders = UnstructuredURLLoader(urls=URLs)
        # loaders = PyPDFDirectoryLoader("pdfs")
        data = loaders.load()

        # split the data into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        chunks=text_splitter.split_documents(data)

        # create embeddings using sentence-transformers model with the help of HFE class
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # store it in vector database
        vectordatabase = FAISS.from_documents(chunks,embedding_model)

        # Initialize the LLM
        llm = OpenAI(api_key=api_key)

        # Prompt Template

        template = """Use the context strictly to provide a concise answer. If you don't know just say don't know.
        {context}
        Question:{question}
        Helpful Answer:"""
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # retriever to fetch the context
        chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=vectordatabase.as_retriever(),
                                            chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
        
        if question:
            with st.spinner("Searching......."):
                answer=chain.run(question)
            st.subheader("Generated Answer:")
            st.write(answer)
    except Exception as e:
        st.error(f"We ran into an error:{str(e)}")
else:
    st.warning("Please enter your OpenAI API key in the sidebar")