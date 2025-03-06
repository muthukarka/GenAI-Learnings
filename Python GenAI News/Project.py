import os
import streamlit as st
import time

import urllib3
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import openai

# for Streamlit local
# os.environ['OPENAI_API_KEY'] = ""

import os
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env (especially openai api key)


openai.api_key = os.getenv('OPENAI_API_KEY')
print("open api key ",openai.api_key)
llm = OpenAI(temperature=0.7, openai_api_key=openai.api_key)

st.title(" News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

for i in range(3):
    url = st.sidebar.text_input(label=str(i + 1))
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
faiss_index_dir = "faiss_index"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

embeddings = OpenAIEmbeddings()

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls, verify_ssl=False)
    main_placeholder.text("Data Loading...Started...")
    data = loader.load()

    # Splitting data

    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )

    main_placeholder.text("Text Splitter...Started..")

    docs = text_splitter.split_documents(data)

    # Create embeddings and FAISS index
    # embeddings = OpenAIEmbeddings()
    vectorindex_openai = FAISS.from_documents(docs, embeddings)

    main_placeholder.text("Embedding Vector Building...Started..")
    time.sleep(2)

    # Save FAISS index
    vectorindex_openai.save_local(faiss_index_dir)
    main_placeholder.text(f"FAISS index saved to {faiss_index_dir}")

# query
query = main_placeholder.text_input("Question: ")
# main_placeholder.text("Query Started..")
print("file_path ", faiss_index_dir)
# main_placeholder.text(file_path)
if query:
    if os.path.exists(faiss_index_dir):
        try:
            print("file_path ", faiss_index_dir)
            vectorstore = FAISS.load_local(faiss_index_dir, embeddings, allow_dangerous_deserialization=True)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)

            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)

        except Exception as e:
            st.error(f"Error loading FAISS index: {e}")


    else:
        st.warning(f"FAISS index not found. Please process URLs first.")  # Use st.warning
