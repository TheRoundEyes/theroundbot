from langchain.text_splitter import CharacterTextSplitter
from google.generativeai import GenerativeModel, GenerationConfig
import json
import streamlit as st



def chunk_text(text, chunk_size=2000, chunk_overlap=300, separator=" "):
    splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)

def initialise_model(model_name="gemini-1.5-flash"):
    # Instantiate the model
    model = GenerativeModel(model_name="gemini-1.5-flash-001")

    # Define the generation configuration
    generation_config = GenerationConfig(
        temperature=0.0,  # Adjust temperature as needed
        response_mime_type="application/json"  # Ensures the response is in JSON format
    )
    return model

def get_response(model, model_behaviour, extracted_text, prompt):

    all_dfs = []
    for chunk in chunk_text(extracted_text):
        try:
            response = model.generate_content([model_behaviour, chunk, prompt])
            return response.text
        except json.JSONDecodeError:
            st.error("Failed to parse JSON response.")



def compare_documents(doc1, doc2, model):

    model_behaviour = """
    You are an expert in analyzing and comparing documents.
    You are given two documents to compare and provide a detailed analysis and key differences of these documents.

    When you compare them you must follow the format below:
    Invoice Numbers: 'Difference'
    """

    # Wrap documents in a dictionary or list
    combined_input = f"{model_behaviour}\nDocument 1:\n{doc1}\n\nDocument 2:\n{doc2}\n"


    response = model.generate_content(combined_input)

    return response.text