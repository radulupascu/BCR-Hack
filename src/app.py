import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# 1. Vectorise the sales response csv data
# loader = PyPDFLoader("files/Contact.pdf")
# documents = loader.load_and_split(chunk_size=1000, split_on="\n", chunk_overlap=0)

loader = PyPDFLoader("files/Contact.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3080 , chunk_overlap=4)
documents = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key='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')
db = FAISS.from_documents(documents, embeddings)

# make a save of the database and here load it in 
# then add the function to add a file on the spot to compare to existing data

# 2. Function for similarity search

def retrieve_info(query):
    similar_response = db.similarity_search(query, k=4)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array


# 3. Setup LLMChain & prompts
llm = ChatOpenAI(openai_api_key='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', temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
Esti un reprezentativ al BCR care are rolul de a da informatii utile clientilor.
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

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(message):
    data = retrieve_info(message)
    for point in data:
        print(point.split("\n"), "\n") 
    response = chain.run(message=message, data=data)
    return response


# 5. Build an app with streamlit
def main():
    st.set_page_config(
        page_title="BCR", page_icon="favicon.ico")

    st.header("Customer response generator 💬")
    message = st.text_area("customer message")

    if message:
        st.write("Generating best practice message...")

        result = generate_response(message)

        st.info(result)


if __name__ == '__main__':
    main()
