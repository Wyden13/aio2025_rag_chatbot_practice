%%writefile app.py
import os
import torch
import streamlit as st
import tempfile

from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub

from langchain_chroma import Chroma
import sentence_transformers

if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'llm' not in st.session_state:
    st.session_state.llm = None

@st.cache_resource
def load_embedding_model(MODEL_NAME):
    embeddings = HuggingFaceEmbeddings(
        model_name = MODEL_NAME
    )
    return embeddings

@st.cache_resource
def load_llm_model(MODEL_NAME):
  # Load model directly
  nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
  )
  model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto"
    # offload_folder = "./offload_dir"
  )
  tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
  model_pipeline = pipeline(
    "text-generation",
    tokenizer = tokenizer,
    max_new_tokens = 512,
    pad_token_id=tokenizer.eos_token_id,
    device_map="auto",
  )
  llm = HuggingFacePipeline(
    pipeline=model_pipeline
  )
  return llm

# Upload Model -> Notify if the model is uploaded successfully
# Upload File -> Notify if the file is uploaded successfully
# Response

def process_pdf(uploaded_file):
  """
  - Upload PDF file
  - Split file into segments of semantic chunks
  - Convert chunks of texts into vectors
  """
  with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
    tmp_file.write(uploaded_file.read())
    uploaded_file = tmp_file.name

  Loader = PyPDFLoader
  try:
      loader = Loader(uploaded_file)
      documents = loader.load()
      if documents:
        # Initialize the Semantic Chunker
        semantic_splitter = text_chunking(st.session_state.embeddings)
        # Divide the content into semantic chunks
        docs = semantic_splitter.split_documents(documents)
        # Convert chunks into vectors and store them in vector database
        vector_db = Chroma.from_documents(
                  documents = docs,
                  embedding = st.session_state.embeddings,
                  )
        retriever = vector_db.as_retriever()
  except Exception as e:
      print(f"an error occured: {e}")

  prompt = hub.pull("rlm/rag-prompt")
  rag_chain=(
      {"content": retriever | format_docs, "question": RunnablePassthrough()}
      | prompt
      | st.session_state.llm
      | StrOutputParser() # get response
  )
  # Combine all document content
  def format_docs(docs):
    return "\n\n".john([doc.page_content for doc in docs])

  # Delete file
  os.unlink(uploaded_file)

  return rag_chain, len(docs)


def text_chunking(embeddings):
    semantic_splitter = SemanticChunker(
    embeddings=embeddings, # Use the embeddings model
    breakpoint_threshold_type = "percentile",
    breakpoint_threshold_amount = 95,
    min_chunk_size = 500, #Minimum chunk size in characters
    add_start_index = True
    )
    return semantic_splitter


def main():
  st.title("RAG Chatbot Assistant")
  st.markdown("""
    **How to use**
    1. **Upload PDF file**: Upload a PDF file then click on button "Process File"
    2. **Questions**: Ask questions related to the file content
  """)
  MODEL_NAME = "lmsys/vicuna-7b-v1.5"

  if not st.session_state.models_loaded:
    st.info("Loading models...")
    st.session_state.embeddings = load_embedding_model(MODEL_NAME)
    st.session_state.llm = load_llm_model(MODEL_NAME)
    st.session_state.models_loaded = True
    st.success("Models loaded successfully!")
    st.rerun()

  # Upload PDF file
  uploaded_file = st.file_uploader("Upload file PDF",type=["pdf"])
  if uploaded_file and st.button("Process file"):
      with st.spinner("Processing"):

        # Process file
        st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)
        st.success("File processed successfully!")
        st.write(f"Total number of chunks: {num_chunks}")
      if st.session_state.rag_chain:
          question = st.text_input("Ask a question:")
      if question:
          with st.spinner("Answering..."):
              output = st.session_state.rag_chain.invoke(question)
              answer = output.split('Answer:')[1].strip()\
                  if 'Answer:' in output else output.strip()
              st.write("**Answer:**")
              st.write(answer)

if __name__ == "__main__":
  main()