from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain, LLMChain,StuffDocumentsChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

load_dotenv()
embeddings = OpenAIEmbeddings()

def create_vectorstore(text):

    vectorstore = FAISS.from_texts(text, embeddings)
    return vectorstore


def answer_questions(vectorstore, question, chat_history=[]):
    # Initialize the language model with the new API
    llm = OpenAI(model="gpt-3.5-turbo")  # Updated initialization for the new API

    # Define the prompt template for answering questions
    prompt_template = PromptTemplate(
        input_variables=['context', 'question'],
        template='You are a helpful assistant. Given the context {context} and the question {question}, provide a helpful answer.'
    )

    # Use the correct document variable name
    docs_chain = StuffDocumentsChain(
        llm_chain=LLMChain(llm=llm, prompt=prompt_template),
        document_variable_name="context"
    )

    # Create a simple prompt template for generating follow-up questions
    question_generation_prompt = PromptTemplate(
        input_variables=["context"],
        template="Given the context {context}, what would be a good follow-up question?"
    )
    question_generator = LLMChain(llm=llm, prompt=question_generation_prompt)

    # Create the ConversationalRetrievalChain
    qa_chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        combine_docs_chain=docs_chain,
        question_generator=question_generator
    )

    # Get the response
    response = qa_chain({"question": question, "chat_history": chat_history})
    return response['answer']