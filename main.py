import streamlit as st
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import validators

# --- Config ---
st.set_page_config(page_title="WebIntel AI", page_icon="🔍", layout="wide")
load_dotenv()

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vectors" not in st.session_state:
    st.session_state.vectors = None

if "sources" not in st.session_state:
    st.session_state.sources = []

# --- Embeddings (Fixed Model) ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

embeddings = get_embeddings()

# --- Sidebar ---
with st.sidebar:
    st.markdown("## 🔍 WebIntel AI")
    api_key = st.text_input("Groq API Key", type="password")

    if api_key:
        os.environ["GROQ_API_KEY"] = api_key

# --- Header ---
st.markdown("<h1 style='text-align:center;'>WebIntel AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Intelligent Web Research Assistant</p>", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Chat", "URL Indexing"])

# =========================
# CHAT TAB
# =========================
with tab1:

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_query = st.chat_input("Ask about indexed content...")

    if user_query:
        if not os.environ.get("GROQ_API_KEY"):
            st.error("Enter Groq API Key.")
        elif not st.session_state.vectors:
            st.error("Index a website first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})

            with st.chat_message("user"):
                st.write(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        llm = ChatGroq(
                            groq_api_key=os.environ["GROQ_API_KEY"],
                            model_name="meta-llama/llama-4-scout-17b-16e-instruct"
                        )

                        prompt = ChatPromptTemplate.from_template(
"""
<system>
    <persona>
        You are WebIntel AI, a strict retrieval-based AI assistant.
        Your knowledge is LIMITED to the provided context.
    </persona>

    <critical_rules>
        - You MUST NOT use prior knowledge.
        - You MUST NOT infer missing details.
        - You MUST NOT speculate.
        - If information is missing, say:
          "I don't know based on the provided context."
    </critical_rules>
</system>

<context>
{context}
</context>

<question>
{input}
</question>

<answer>
Write a high-quality, structured answer strictly derived from the context.
If the answer cannot be found, say:
"I don't know based on the provided context."
</answer>
"""
)

                        document_chain = create_stuff_documents_chain(llm, prompt)
                        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 5})
                        retrieval_chain = create_retrieval_chain(retriever, document_chain)

                        response = retrieval_chain.invoke({"input": user_query})

                        answer = response.get("answer", "No response generated.")
                        st.write(answer)

                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        st.session_state.sources = retriever.invoke(user_query)

                    except Exception as e:
                        st.error(str(e))

    # Show Sources
    if st.session_state.sources:
        with st.expander("Sources"):
            for i, doc in enumerate(st.session_state.sources):
                st.markdown(f"**Source {i+1}**")
                st.write(doc.page_content[:1000])
                if "source" in doc.metadata:
                    st.caption(doc.metadata["source"])
                st.divider()


# =========================
# URL INDEXING TAB
# =========================
with tab2:

    st.markdown("### Index Website")

    url_input = st.text_input("Enter URL", placeholder="https://example.com")

    if st.button("Index Website"):

        if not validators.url(url_input):
            st.error("Invalid URL")
        else:
            with st.spinner("Scraping..."):
                try:
                    loader = WebBaseLoader(
                        url_input,
                        header_template={
                            "User-Agent": (
                                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                "AppleWebKit/537.36 (KHTML, like Gecko) "
                                "Chrome/120.0.0.0 Safari/537.36"
                            )
                        }
                    )

                    docs = loader.load()

                    if not docs:
                        st.error("No content extracted.")
                        st.stop()

                    clean_docs = []

                    blocked_keywords = [
                        "Cloudflare",
                        "Attention Required",
                        "enable cookies"
                    ]

                    for doc in docs:
                        text = doc.page_content.strip()

                        if not text:
                            continue

                        if any(k in text for k in blocked_keywords):
                            st.error("Website blocked scraping.")
                            st.stop()

                        clean_docs.append(doc)

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=100
                    )

                    final_docs = splitter.split_documents(clean_docs)

                    if not final_docs:
                        st.error("Chunking failed.")
                        st.stop()

                    st.session_state.vectors = FAISS.from_documents(
                        final_docs,
                        embeddings
                    )

                    st.success(f"Indexed {len(final_docs)} chunks successfully.")

                except Exception as e:
                    st.error(str(e))
