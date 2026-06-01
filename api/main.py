import json
from pathlib import Path
import shutil
import subprocess

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import requests

from api.models import ActionItem, Decision, MeetingSummary

app = FastAPI()

VIDEOS_DIR = Path("videos")
SUMMARY_NOTES_DIR = Path("vault/meetings/summaries")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
VIDEOS_DIR.mkdir(exist_ok=True)


class SummarizeRequest(BaseModel):
    # The path to a transcript text file, such as transcripts/sample_meeting.txt.
    transcript_file: str


def get_first_sentence(transcript_text: str) -> str:
    # Use the first sentence as a simple fallback TL;DR.
    first_sentence = transcript_text.split(".", 1)[0].strip()

    if first_sentence:
        return first_sentence

    return "No transcript text was found."


def create_rule_based_summary(transcript_path: Path, transcript_text: str) -> MeetingSummary:
    # These simple keyword lists let us do beginner-friendly rule-based extraction.
    decision_keywords = ["decide", "decision", "agreed"]
    action_item_keywords = ["will", "todo", "follow up", "action item"]
    question_keywords = ["?", "question", "unclear"]

    decisions = []
    action_items = []
    open_questions = []

    # Check each transcript line for simple decision, action item, and question clues.
    for line in transcript_text.splitlines():
        clean_line = line.strip()

        # Skip blank lines so we do not add empty items.
        if not clean_line:
            continue

        lower_line = clean_line.lower()

        if any(keyword in lower_line for keyword in decision_keywords):
            decisions.append(
                Decision(
                    decision=clean_line,
                    source_timestamp=None,
                )
            )

        if any(keyword in lower_line for keyword in action_item_keywords):
            action_items.append(
                ActionItem(
                    task=clean_line,
                    owner=None,
                    due_date=None,
                    source_timestamp=None,
                )
            )

        if any(keyword in lower_line for keyword in question_keywords):
            open_questions.append(clean_line)

    return MeetingSummary(
        title=transcript_path.stem.replace("_", " ").title(),
        tldr=get_first_sentence(transcript_text),
        key_topics=[
            "teamwork",
            "collaboration",
            "scheduling",
        ],
        decisions=decisions,
        action_items=action_items,
        open_questions=open_questions,
        tags=[
            "meeting",
            "auto-summary",
        ],
    )


def create_ollama_summary(transcript_path: Path, transcript_text: str) -> MeetingSummary:
    # Ask Ollama to return JSON that matches our MeetingSummary shape.
    prompt = f"""
You are summarizing a meeting transcript.

Return only valid JSON with these keys:
title, tldr, key_topics, decisions, action_items, open_questions, tags

Rules:
- key_topics, decisions, action_items, open_questions, and tags must be lists.
- each decision must be an object with decision and source_timestamp.
- each action item must be an object with task, owner, due_date, and source_timestamp.
- use null when owner, due_date, or source_timestamp is unknown.

Transcript:
{transcript_text}
"""

    print("Calling Ollama for meeting summary...")
    print(f"Ollama model: {OLLAMA_MODEL}")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        },
        timeout=120,
    )
    response.raise_for_status()

    # Ollama returns the model's text inside the "response" field.
    ollama_response_text = response.json()["response"]
    summary_data = json.loads(ollama_response_text)

    # Add safe defaults in case the model leaves out optional-looking fields.
    summary_data.setdefault("title", transcript_path.stem.replace("_", " ").title())
    summary_data.setdefault("tldr", get_first_sentence(transcript_text))
    summary_data.setdefault("key_topics", [])
    summary_data.setdefault("decisions", [])
    summary_data.setdefault("action_items", [])
    summary_data.setdefault("open_questions", [])
    summary_data.setdefault("tags", ["meeting", "ollama-summary"])

    # Build and validate the Pydantic response object.
    return MeetingSummary(**summary_data)


def save_summary_note(meeting_summary: MeetingSummary, transcript_path: Path):
    # Create the summaries folder if it does not already exist.
    SUMMARY_NOTES_DIR.mkdir(parents=True, exist_ok=True)

    # Save a markdown version of the summary for Obsidian or simple reading.
    meeting_name = transcript_path.stem
    summary_note_path = SUMMARY_NOTES_DIR / f"summary_{meeting_name}.md"

    with summary_note_path.open("w", encoding="utf-8") as summary_file:
        summary_file.write(f"# Summary: {meeting_summary.title}\n\n")

        summary_file.write("## TL;DR\n\n")
        summary_file.write(f"- {meeting_summary.tldr}\n\n")

        summary_file.write("## Key Topics\n\n")
        for topic in meeting_summary.key_topics:
            summary_file.write(f"- {topic}\n")
        summary_file.write("\n")

        summary_file.write("## Decisions\n\n")
        if meeting_summary.decisions:
            for decision in meeting_summary.decisions:
                summary_file.write(f"- {decision.decision}\n")
        else:
            summary_file.write("- None found.\n")
        summary_file.write("\n")

        summary_file.write("## Action Items\n\n")
        if meeting_summary.action_items:
            for action_item in meeting_summary.action_items:
                summary_file.write(f"- {action_item.task}\n")
        else:
            summary_file.write("- None found.\n")
        summary_file.write("\n")

        summary_file.write("## Open Questions\n\n")
        if meeting_summary.open_questions:
            for question in meeting_summary.open_questions:
                summary_file.write(f"- {question}\n")
        else:
            summary_file.write("- None found.\n")
        summary_file.write("\n")

        summary_file.write("## Tags\n\n")
        for tag in meeting_summary.tags:
            summary_file.write(f"- {tag}\n")

    print(f"Markdown summary saved to: {summary_note_path}")


@app.get("/")
def home():
    return {
        "message": "Team Memory API is running"
    }


@app.post("/transcribe")
async def transcribe_video(file: UploadFile = File(...)):
    # Save uploaded video
    save_path = VIDEOS_DIR / file.filename

    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run the processing pipeline
    subprocess.run(
        ["python3", "scripts/process_video.py"],
        check=True
    )

    return {
        "filename": file.filename,
        "saved_to": str(save_path),
        "status": "video processed successfully"
    }


@app.post("/summarize", response_model=MeetingSummary)
def summarize_transcript(request: SummarizeRequest):
    # Turn the transcript path string into a Path object so Python can read it.
    transcript_path = Path(request.transcript_file)

    # Read the transcript text from disk.
    transcript_text = transcript_path.read_text(encoding="utf-8")

    try:
        # Try the local Ollama model first.
        meeting_summary = create_ollama_summary(transcript_path, transcript_text)
        print("Ollama summary completed successfully.")
    except Exception as error:
        # If Ollama is unavailable or returns invalid data, use the old simple rules.
        print(f"Ollama summary failed: {error}")
        print("Fallback logic triggered: using rule-based summary.")
        meeting_summary = create_rule_based_summary(transcript_path, transcript_text)

    save_summary_note(meeting_summary, transcript_path)

    # Return the structured JSON response.
    return meeting_summary
