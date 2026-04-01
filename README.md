[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

# ChatDoc AI

> Your personal AI assistant that can chat, summarize documents, generate code, and more — all in one Streamlit app.

---

## 🚀 Features

* **Chat Memory (Multi-turn Conversation):** AI remembers previous interactions for coherent conversations.
* **Multiple Document Support:** Upload and query across PDFs, CSVs, and text files simultaneously.
* **Drag & Drop UI:** Smooth, intuitive interface for uploading multiple files effortlessly.
* **Semantic Search:** Finds the most relevant chunks of text based on your question.
* **AI Summarization:** Summarize large documents or notes in seconds.
* **Code Execution & Debugging:** Run Python snippets safely with automated error handling.
* **Hallucination-Safe RAG Mode:** Uses Retrieval-Augmented Generation (RAG) for citation-backed answers.

---

## 📂 Project Structure

```
chatdoc-ai/
│
├─ app.py                 # Main Streamlit app
├─ utils.py               # Helper functions (file handling, session management)
├─ rag_pipeline.py        # RAG & vector DB logic
├─ session_data/          # Session-specific vector DB and uploaded files
├─ requirements.txt       # Python dependencies
├─ README.md              # Project description & instructions
└─ sample_data/           # Example PDFs, CSVs, text files
```

---

## 💻 Installation

1. Clone the repository:

```bash
git clone https://github.com/sundysd/chatdoc-ai.git
cd chatdoc-ai
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # Linux / Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ⚡ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

* Drag & drop your files to upload multiple documents at once.
* Start chatting — the AI remembers previous conversation turns.
* Switch to **Hallucination-Safe RAG** mode for citation-backed answers.
* Ask code-related questions and see auto-debugged results.

---

## 🧠 How It Works

1. **Document Loading & Splitting:** Supports TXT, PDF, CSV, XLSX/XLS files and splits them into manageable chunks.
2. **Vector Embeddings:** Uses OpenAI embeddings to convert chunks into vector representations.
3. **Vector DB & Semantic Search:** FAISS database retrieves the most relevant document chunks based on your query.
4. **AI Response Generation:** Combines user query + retrieved context to generate accurate answers.
5. **Multi-turn Chat Memory:** Maintains session-based conversation history.
6. **Follow-up Suggestions:** Generates 3 suggested follow-up questions for better exploration of content.

---

## 📈 Why This Project Stands Out

* Production-ready architecture for real-world use.
* Multi-document support and multi-turn chat for a rich experience.
* Smooth drag & drop UI for seamless interaction.
* AI features beyond simple Q&A: summarization, code generation, debugging.

---

## 🛠 Future Improvements

* Add **voice input/output** for a conversational assistant.
* Integrate **more AI models** for specialized tasks.
* Enhance **UI customization** and theme options.

---

## 👤 Author

**Di Sun** – AI Developer & Portfolio Enthusiast
[LinkedIn](https://www.linkedin.com/in/yourprofile) | [GitHub](https://github.com/sundysd)
