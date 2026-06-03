from pathlib import Path

import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/ask"
SUMMARY_NOTES_DIR = Path("vault/meetings/summaries")

st.set_page_config(
    page_title="Team Memory",
    page_icon="🧠",
    layout="wide"
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

                st.session_state.answer = data["answer"]
                st.session_state.sources = data["sources"]

            else:
                st.error("Failed to contact Team Memory API.")

    if st.session_state.answer:
        st.subheader("Answer")
        st.write(st.session_state.answer)

    if st.session_state.sources:
        st.subheader("Sources")

        for source in st.session_state.sources:
            st.write(
                f"📄 {source['file_name']} "
                f"(score: {source['score']:.3f})"
            )


with right_column:
    st.subheader("Source Preview")

    top_source = st.session_state.sources[0] if st.session_state.sources else None
    markdown_content, error_message = load_source_markdown(top_source)

    if markdown_content:
        st.markdown(markdown_content)
    elif error_message:
        st.info(error_message)
