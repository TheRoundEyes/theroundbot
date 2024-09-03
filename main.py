import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import extractor as ext
import analyzer as ana
import concurrent.futures
import chat as chat

st.set_page_config("The Round Bot 3.0 - Gemini", layout="wide")

load_dotenv()

api_key = st.text_input("Enter your API key", type="password")

genai.configure(api_key=api_key)


def process_file(uploaded_file):
    if uploaded_file.name.endswith(".pdf"):
        return ext.extract_text_from_pdf(uploaded_file)
    elif uploaded_file.name.endswith(".docx"):
        return ext.extract_text_from_doc(uploaded_file)
    return ""


model_behaviour = """
You are an expert in processing document structures and extracting data from them.
We will upload a DOC or PDF file, and you need to extract the relevant information based on the given prompt.
You will provide the output in table format. Text-based answers are not acceptable. You have to strictly follow the format below.
Given a document, your task is to extract the text value of the following entities:
{
'Company Name':'',
'Client Name':'',
'Issue Date':'',
'Items':[
     {
          'Item Name':'',
          'Item Quantity':'',
          'Item Total':''
     }
    ],
}
If a Column is not present on the given columns, use the name of the column from the document. If you can't identify the value, leave it empty.
Format the JSON properly as it should be in a dictionary format. Your output should be in JSON format and should only be called "table". 
Remove any other keys from the output.
"""

model = ana.initialise_model("gemini-1.5-flash")
st.header("The Round Bot 3.0 - Gemini")

# Read the prompt in text box
prompt = "Generate a table from the DOC file content"

with st.sidebar:
    uploaded_files_set1 = st.file_uploader("Upload the first set of PDF or DOC files", type=["pdf", "docx"],
                                           accept_multiple_files=True)
    all_texts_set1 = []

    uploaded_files_set2 = st.file_uploader("Upload the second set of PDF or DOC files", type=["pdf", "docx"],
                                           accept_multiple_files=True)
    all_texts_set2 = []

col1, col2 = st.columns(2)

if uploaded_files_set1 and uploaded_files_set2:
    with st.spinner('Processing files...'):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results_set1 = executor.map(process_file, uploaded_files_set1)
            all_texts_set1 = list(results_set1)

            results_set2 = executor.map(process_file, uploaded_files_set2)
            all_texts_set2 = list(results_set2)

    # Combine all texts from both sets
    combined_text_set1 = ' '.join(all_texts_set1)
    combined_text_set2 = ' '.join(all_texts_set2)

    #Store texts to vectorstore
    documents = [combined_text_set1, combined_text_set2]
    vectorstore = chat.create_vectorstore(documents)


    # Analyze both sets of texts using the LLM
    df_set1 = ana.get_response(model, combined_text_set1, model_behaviour, prompt)
    df_set2 = ana.get_response(model, combined_text_set2, model_behaviour, prompt)

    # Display the extracted data
    with col1:
        st.write("Extracted Data from Set 1:")
        with st.expander("Scroll to see more", expanded=True):
            st.write(df_set1)
        st.write("Extracted Data from Set 2:")
        with st.expander("Scroll to see more", expanded=True):
            st.write(df_set2)

    with col2:
        chat_history = []
        user_input = st.text_area("Enter your query here", "")
        if st.button("Send"):
            response = chat.answer_questions(vectorstore, user_input,chat_history)
            st.write(response)
            chat_history.append({"question": user_input, "answer": response})

    # comparison = ana.compare_documents(combined_text_set1, combined_text_set2, model)
    # st.write(comparison)
