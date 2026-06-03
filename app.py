import json
from pathlib import Path
import sqlite3
from datetime import datetime

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"
SUMMARY_NOTES_DIR = Path("vault/meetings/summaries")
FEEDBACK_DB = Path("feedback.db")

st.set_page_config(
    page_title="Team Memory",
    page_icon="🧠",
    layout="wide"
)


def create_feedback_table():
    # Create the feedback table if this is the first time the app is running.
    with sqlite3.connect(FEEDBACK_DB) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question TEXT,
                answer TEXT,
                sources TEXT,
                rating TEXT,
                created_at TEXT
            )
            """
        )


def save_feedback(rating):
    # Store the user's feedback for the current answer.
    sources_json = json.dumps(st.session_state.sources)
    created_at = datetime.now().isoformat(timespec="seconds")

    with sqlite3.connect(FEEDBACK_DB) as connection:
        connection.execute(
            """
            INSERT INTO feedback (
                question,
                answer,
                sources,
                rating,
                created_at
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                st.session_state.question,
                st.session_state.answer,
                sources_json,
                rating,
                created_at,
            ),
        )


def load_source_markdown(source):
    # Read the markdown note for the highest-scoring source.
    if not source:
        return None, "Ask a question to preview the top source."

    file_name = source.get("file_name")

    if not file_name:
        return None, "The top source did not include a file name."

    source_path = SUMMARY_NOTES_DIR / file_name

    if not source_path.exists():
        return None, f"Source markdown file not found: {source_path}"

    return source_path.read_text(encoding="utf-8"), None


create_feedback_table()


if "question" not in st.session_state:
    st.session_state.question = None

if "answer" not in st.session_state:
    st.session_state.answer = None

if "sources" not in st.session_state:
    st.session_state.sources = []


left_column, right_column = st.columns([1, 1])


with left_column:
    st.title("🧠 Team Memory")

    question = st.text_input(
        "Ask a question about your meetings:",
        placeholder="What decisions were made about Qdrant?"
    )

    if st.button("Ask") and question:

        with st.spinner("Searching meeting memory..."):

            response = requests.post(
                API_URL,
                json={"question": question}
            )

            if response.status_code == 200:

                data = response.json()

                st.session_state.question = question
                st.session_state.answer = data["answer"]
                st.session_state.sources = data["sources"]

            else:
                st.error("Failed to contact Team Memory API.")

    if st.session_state.answer:
        st.subheader("Answer")
        st.write(st.session_state.answer)

        feedback_left, feedback_right = st.columns(2)

        with feedback_left:
            if st.button("👍 Helpful"):
                save_feedback("positive")
                st.success("Feedback saved.")

        with feedback_right:
            if st.button("👎 Not helpful"):
                save_feedback("negative")
                st.success("Feedback saved.")

    if st.session_state.sources:
        st.subheader("Sources")

        for source in st.session_state.sources:
            source_label = f"📄 {source['file_name']}"

            if source.get("timestamp"):
                source_label += f" @ {source['timestamp']}"

            st.write(f"{source_label} (score: {source['score']:.3f})")


with right_column:
    st.subheader("Source Preview")

    top_source = st.session_state.sources[0] if st.session_state.sources else None
    markdown_content, error_message = load_source_markdown(top_source)

    if markdown_content:
        st.markdown(markdown_content)
    elif error_message:
        st.info(error_message)
