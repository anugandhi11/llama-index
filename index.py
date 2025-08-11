import os
import streamlit as st
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from langchain.llms.openai import OpenAI

# --- App Configuration ---
MODEL_NAME = "text-davinci-003"
PROMPT_HELPER_CONFIG = {
    "max_input_size": 4096,
    "num_output": 256,
    "max_chunk_overlap": 20,
}
DATA_DIR = "data"
INDEX_SAVE_PATH = "index.json"

# --- App UI ---
st.title("Ask Llama")
st.markdown("This app allows you to ask questions about Paul Graham's essay, using LlamaIndex and OpenAI.")

# --- Security and API Key Check ---
if 'OPENAI_API_KEY' not in os.environ:
    st.error("🛑 Please set the OPENAI_API_KEY environment variable to use this app.")
    st.stop()

# --- Core Logic ---
@st.cache_resource
def load_index():
    """
    Loads the LlamaIndex vector index from disk, or creates and saves it if not found.
    Uses Streamlit's caching to avoid re-loading on every interaction.
    """
    if os.path.exists(INDEX_SAVE_PATH):
        st.info("Loading index from disk...")
        index = GPTSimpleVectorIndex.load_from_disk(INDEX_SAVE_PATH)
    else:
        st.info("Index not found. Creating and saving a new index... This may take a few moments.")

        # Configure LLM and prompt helper
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name=MODEL_NAME))
        prompt_helper = PromptHelper(**PROMPT_HELPER_CONFIG)

        # Load documents and build index
        documents = SimpleDirectoryReader(DATA_DIR).load_data()
        service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
        index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

        # Save index to disk
        index.save_to_disk(INDEX_SAVE_PATH)

    st.success("✅ Index loaded successfully!")
    return index

try:
    index = load_index()
except Exception as e:
    st.error(f"An error occurred while loading the index: {e}")
    st.stop()

# --- User Interaction ---
query = st.text_input(f"What would you like to ask? (source: {DATA_DIR}/paul_graham_essay.txt)", "")

if st.button("Submit"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        try:
            with st.spinner("Querying the index..."):
                response = index.query(query)
            st.success("Query successful!")
            st.markdown(response) # Use st.markdown for better formatting
        except Exception as e:
            st.error(f"An error occurred during the query: {e}")
