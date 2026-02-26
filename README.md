🔍 WebIntel AI

WebIntel AI is an AI-powered Website Intelligence Tool that enables contextual question answering over live websites using Retrieval-Augmented Generation (RAG).

Built with Streamlit, Groq Llama 4, and FAISS vector search, the system allows users to index any website and ask grounded questions strictly based on extracted content.

🚀 Features

🔗 Live Website Indexing

🧠 Retrieval-Augmented Generation (RAG)

🤖 Powered by Groq Llama 4

📚 Context-grounded responses (No hallucinations)

🔍 Semantic search using FAISS

🖥 Clean Streamlit UI

🔐 Secure API key input (session-based)

🏗 Architecture

User → Website Scraping → Text Chunking → Embeddings → FAISS Vector Store → Retriever → LLM (Llama 4) → Grounded Answer

Tech Stack:

Streamlit

Groq (Llama 4 Scout)

LangChain

FAISS

HuggingFace Embeddings (all-mpnet-base-v2)

🧠 How It Works

User enters a website URL.

The system scrapes and extracts page content.

Content is chunked and converted into embeddings.

Embeddings are stored in FAISS vector index.

User asks questions.

Retriever fetches relevant chunks.

Llama 4 generates a strictly context-based answer.

If the answer is not found in context, the model responds:

"I don't know based on the provided context."
