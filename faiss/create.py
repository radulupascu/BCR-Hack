import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfMerger
import os

def load_and_split_pdfs(file_paths: list, chunk_size: int = 3080):
  loaders = [PyPDFLoader(file_path) for file_path in file_paths]
  pages = []
  for loader in loaders:
    pages.extend(loader.load())

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=int(chunk_size / 10),
      separators=[
        "\n\n",
        "\n",
        "'",
        "\uf0fc",
        "'",
        ",",
        "-",
        ":",
        " ",
        ".",
        ",",
        "',",
        ";",
        "\u200B",  # Zero-width space
        "\uff0c",  # Fullwidth comma
        "\u3001",  # Ideographic comma
        "\uff0e",  # Fullwidth full stop
        "\u3002",  # Ideographic full stop
        "",
    ],
    )
  docs = text_splitter.split_documents(pages)
  return docs

# docs = load_and_split_pdfs(['files/Contact.pdf'])

embeddings = OpenAIEmbeddings(openai_api_key='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')




if __name__ == "__main__":
    
    pdfs = os.listdir('/Users/radu/dev/BCR-Hack/dcs')
    # merger = PdfMerger()

    # docs = []
    # for pdf in pdfs:
    #   pdf = 'dcs/' + pdf

    docs = load_and_split_pdfs(["data.pdf"], chunk_size=880)
    # docs = load_and_split_pdfs(docs)

    
    # docs = load_and_split_pdfs(dup) 
    db = FAISS.from_documents(docs, embeddings)
    print(1)
    db.save_local("db")
    print(2)
    #     merger.append(pdf)

    # merger.write("result.pdf")
    # merger.close()


    new_db = FAISS.load_local("db", embeddings)
    query = input("Enter the path to the file you want to compare: ")
    similar_response = new_db.similarity_search(query, k=3)
    page_contents_array = [doc.page_content for doc in similar_response]

    for point in page_contents_array:
        print(point.split("\n"), "\n") 