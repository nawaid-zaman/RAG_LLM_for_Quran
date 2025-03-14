# Quran-Based RAG System


## Overview

This project is a **Retrieval-Augmented Generation (RAG) system** designed to provide **context-aware responses** to queries related to the Quran. The primary goal is to ensure accuracy in Quranic references and prevent AI models from misquoting verses, a common issue observed in general-purpose LLMs.


## Features

- **Accurate Quranic Responses:** Ensures Quran-related queries are answered with proper context.

- **Prevents Misinformation:** If a question is outside the scope of the Quran, the system responds with "I have not learned this."

- **Powered by LLMs & Embeddings:** Utilizes ChatGPT API (for now) for generating responses and embeddings.

- **User-Friendly Interface:** Built using Streamlit for an interactive experience.

- **Modular & Extendable:** Designed with Python and LangChain, making it adaptable for future improvements.


## Tech Stack

- **Backend:** Python, LangChain

- **Frontend:** Streamlit

- **LLM & Embeddings:** ChatGPT API (to be replaced with Ollama LLMs in future versions)

- **Vector Database**: FAISS (or similar for optimized retrieval)


## How It Works

- **User Input:** The system takes a question related to the Quran.

- **Context Retrieval:** Retrieves relevant Quranic passages using embeddings.

- **LLM Processing:** The model analyzes the query along with the retrieved context.

- **Response Generation:** If the query is Quran-related, an appropriate answer is provided; otherwise, the system returns "I have not learnt this."


