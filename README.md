# ChatDoc AI

ChatDoc AI is a secure, multi-document RAG assistant built with Streamlit.

Users bring their own OpenAI API key, upload documents, and chat with grounded answers in a clean, interactive interface.

## Features

- Bring your own API key (BYOK) directly in the UI
- Session isolation for key, files, vector store, and chat history
- Multi-file upload: TXT, PDF, CSV, XLSX, XLS
- Streaming responses for real-time output
- Token usage and estimated cost per answer
- Custom system prompt with ready-to-use prompt presets
- Regenerate last response with different model/temperature
- Follow-up question suggestions after each answer
- Sidebar document metadata, previews, and download actions

## Project Structure

```text
chatdoc-ai/
	app.py
	rag_pipeline.py
	utils.py
	requirements.txt
	README.md
```

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

## Usage Flow

1. Enter your OpenAI API key.
2. Upload one or more documents.
3. Click Build Knowledge Base.
4. Ask questions in chat.
5. Optionally tune model settings and regenerate responses.

## Security Model

- API keys are not shared across users.
- Uploaded files are stored per session folder.
- Vector databases are session-scoped.
- Conversation history is session-scoped.

## Deploy on Streamlit Cloud

1. Push this project to GitHub.
2. Create a new Streamlit Cloud app from the repository.
3. Set the app entry file to app.py.
4. Deploy.

No global OpenAI key is required because each user enters their own key.

## Suggested Repository Name

chatdoc-ai
