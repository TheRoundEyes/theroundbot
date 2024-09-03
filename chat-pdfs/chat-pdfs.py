import google.generativeai as genai
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.vectorstores import FAISS
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import LLMChain

from langchain.prompts import PromptTemplate

load_dotenv()

# Set the API key
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)


# Get text from PDF
def get_pdf_text(files):
    text = ""
    for file in files:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text


# Get Chunks from Text, This method splits the texts into 10000 chunks with 1000 overlap
def get_text_chunks(text, chunk_size=10000, chunk_overlap=1000):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks


# Get Embeddings from text and save vector store
def get_vector_store(chunks, index_name="faiss_index"):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(index_name)


# Get Conversation Chain
def get_conversation_chain():
    prompt_template = """
    You're an expert in the field of Insurance. Your job is to answer questions that relate to the context given to you.
    You are given an underwriting authority to make decisions on the spot. You are also given a team of experts to help you with decision making.

    Authority:
    {authority}

    You must then review the policy schedule and check to see if this complies with the authority given to you.

    Policy Schedule:
    {policy_schedule}

    Identify any areas where the policy schedule does not comply with the underwriting authority and provide a recommendation and a detailed summary.
    Also highlight what part of the policy schedule did not comply.
    """
    # Initialize the model
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    # Initialize the prompt template and set the input variables
    prompt = PromptTemplate(template=prompt_template, input_variables=['authority', 'policy_schedule'])

    # Load the QA chain with 'stuff' chain type
    chain = LLMChain(llm=model, prompt=prompt)
    return chain

def user_input(query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load the policy schedule vector store
    policy_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = policy_db.similarity_search(query)

    # Combine the documents to form the policy schedule text
    policy_schedule_text = "\n".join([doc.page_content for doc in docs])

    # Retrieve the authority text from the processed PDF
    authority_text = get_pdf_text(st.session_state['uploaded_authority'])

    # Get the conversation chain
    chain = get_conversation_chain()

    # Run the chain with the appropriate inputs
    response = chain({
        "input_documents": docs,  # Assuming 'docs' is what StuffDocumentsChain expects
        "authority": authority_text,
        "policy_schedule": policy_schedule_text
    })

    # Display the response in Streamlit
    st.write("Reply:", response)




# Main function
def main():
    st.set_page_config("Aquraid 3.0 - Gemini")
    st.header("Chat with Aquraid 3.0 - Gemini")

    user_query = st.text_input("Enter your query")

    if user_query:
        user_input(user_query)

    with st.sidebar:
        st.title("Menu")
        authority = st.file_uploader("Upload Underwriting Authority", type=["pdf"], accept_multiple_files=True)
        schedule = st.file_uploader("Upload Underwriting Schedule", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit and Process"):
            with st.spinner("Processing file/s..."):
                authority_text = get_pdf_text(authority)
                schedule_text = get_pdf_text(schedule)
                chunks = get_text_chunks(schedule_text)
                st.session_state['uploaded_authority'] = authority_text
                get_vector_store(chunks)  # Save vector store for policy schedule
                st.success("Processing Completed")


if __name__ == "__main__":
    main()
