import os
import uuid

def get_session_dir(session_id):
    """Create and return session-specific data directory."""
    session_data_dir = os.path.join("session_data", session_id)
    os.makedirs(session_data_dir, exist_ok=True)
    return session_data_dir

def save_uploaded_file(uploaded_file, session_id):
    session_dir = get_session_dir(session_id)
    file_path = os.path.join(session_dir, uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path