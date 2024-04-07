import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

FILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files")
)

load_dotenv()

# 1. Vectorise the sales response csv data

def load_and_split_pdfs(file_paths: list, chunk_size: int = 256):
  loaders = [PyPDFLoader(file_path) for file_path in file_paths]
  pages = []
  for loader in loaders:
    pages.extend(loader.load())

  text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=chunk_size,
      chunk_overlap=int(chunk_size / 10),
      strip_whitespace=True,
    )
  docs = text_splitter.split_documents(pages)
  return docs

embeddings = OpenAIEmbeddings(openai_api_key='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')
db = FAISS.load_local("db", embeddings) 

# make a save of the database and here load it in 
# then add the function to add a file on the spot to compare to existing data

# 2. Function for similarity search

def retrieve_info(query, k = 4):
    similar_response = db.similarity_search(query, k=k)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array


# 3. Setup LLMChain & prompts
llm = ChatOpenAI(openai_api_key='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', temperature=0,
                #   model="gpt-3.5-turbo-16k-0613",
                  model="gpt-4",
                  )

template = """
Esti un reprezentativ al BCR pe o platforma de customer support care are rolul de a da informatii utile clientilor.
Iti dau intrebarea clientului si tu ii vei raspunde cat mai bine bazat pe informatiile din documentele date.
Trimit la client bazat pe informatia din documente si pe baza la urmatoarele reguli:

1/ Nu ai voie sa iei date din surse externe.
2/ Raspunsul trebuie sa fie un rezumat precis al datelor usor de inteles.
3/ Daca datele din documente nu sunt relevante, spune ca nu ai informatii.
4/ Precizeaza si din ce articol ai luat informatia.

Daca dai un raspuns bun o sa primesti un bonus de 200$ cu care poti sa iti cumperi ce vrei.

Mesajul de la client este:
{message}

Aici sunt date din documente:
{data}

Te rog sa scrii raspunsul cel mai bun pe care sa il trimit clientului:
"""
# 2/ Raspunsul trebuie sa fie exact la fel sau foarte similar cu cel din documente.


prompt = PromptTemplate(
    input_variables=["message", "data"],
    template=template
)

chain = LLMChain(llm=llm,
                 prompt=prompt,
                 verbose=True,
                 )


# 4. Retrieval augmented generation
def generate_response(message, k = 4):
    data = retrieve_info(message, k = k)
    for point in data:
        print(point.split("\n"), "\n") 
    response = chain.run(message=message, data=data)
    return [response, data]

def save_file(uploaded_file):
    """helper function to save documents to disk"""
    file_path = os.path.join(FILES_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="BCR", page_icon="favicon.ico")

    st.header("🏎️ Turbo-George 🏎️")
    message = st.text_area("🤗 Cu ce te putem ajuta? 🤗")
    
    
    if message:
        st.write("🤔 Ma gandescc hmmm... 🤔")

        k = 5
        [result, data] = generate_response(message, k=k)

        st.info(result)

        st.write("Chunkuri de documente cu k= {k}:".format(k=k))
        for point in data: st.write(point.split("\n"))
    
    

if __name__ == '__main__':
    main()
