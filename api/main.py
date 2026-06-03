import json
from pathlib import Path
import re
import shutil
import subprocess

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from qdrant_client import QdrantClient
import requests
from sentence_transformers import SentenceTransformer
from starlette.responses import StreamingResponse

from api.models import ActionItem, Decision, MeetingSummary

app = FastAPI()

VIDEOS_DIR = Path("videos")
SUMMARY_NOTES_DIR = Path("vault/meetings/summaries")
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"
QDRANT_URL = "http://localhost:6333"
QDRANT_COLLECTION = "team_memory"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K_RESULTS = 3
MIN_KEYWORD_MATCH_RATIO = 0.25
VIDEOS_DIR.mkdir(exist_ok=True)


class SummarizeRequest(BaseModel):
    # The path to a transcript text file, such as transcripts/sample_meeting.txt.
    transcript_file: str


class AskRequest(BaseModel):
    # The user's question about indexed meeting summaries.
    question: str


def get_first_sentence(transcript_text: str) -> str:
    # Use the first sentence as a simple fallback TL;DR.
    first_sentence = transcript_text.split(".", 1)[0].strip()

    if first_sentence:
        return first_sentence

    return "No transcript text was found."


def clean_transcript_text(transcript_text: str) -> str:
    # Clean common transcript noise before sending text to summarization.
    cleaned_lines = []

    for line in transcript_text.splitlines():
        clean_line = line.strip()

        # Skip metadata lines from the transcript file.
        if not clean_line or clean_line.lower().startswith("detected language:"):
            continue

        # Remove timestamp ranges like [00:00:01 - 00:00:04].
        clean_line = re.sub(
            r"^\[\d{2}:\d{2}:\d{2}\s*-\s*\d{2}:\d{2}:\d{2}\]\s*",
            "",
            clean_line,
        )

        # Correct a few common speech-to-text mistakes in technical terms.
        clean_line = re.sub(r"\bqdrin\b", "Qdrant", clean_line, flags=re.IGNORECASE)
        clean_line = re.sub(r"\bqdrant\b", "Qdrant", clean_line, flags=re.IGNORECASE)
        clean_line = re.sub(r"\bco-pilot\b", "copilot", clean_line, flags=re.IGNORECASE)

        if clean_line:
            cleaned_lines.append(clean_line)

    return "\n".join(cleaned_lines)


def get_source_timestamp(line: str):
    # Capture a timestamp range if the original transcript line has one.
    timestamp_match = re.match(
        r"^\[(\d{2}:\d{2}:\d{2}\s*-\s*\d{2}:\d{2}:\d{2})\]",
        line.strip(),
    )

    if timestamp_match:
        return timestamp_match.group(1)

    return None


def guess_action_owner(clean_line: str):
    # Very simple owner guess: "Mike will ..." becomes owner "Mike".
    owner_match = re.match(r"^([A-Z][a-z]+)\s+will\b", clean_line)

    if owner_match:
        return owner_match.group(1)

    return None


def extract_key_topics(cleaned_transcript_text: str):
    # Keep fallback topics grounded in words that actually appear in the transcript.
    topic_keywords = {
        "Qdrant": "qdrant",
        "Ollama": "ollama",
        "FastAPI": "fastapi",
        "API documentation": "api documentation",
        "Faster Whisper": "faster whisper",
        "Whisper": "whisper",
        "copilot": "copilot",
        "Phase 2": "phase 2",
    }

    lower_text = cleaned_transcript_text.lower()
    topics = []

    for topic_name, keyword in topic_keywords.items():
        if keyword in lower_text:
            topics.append(topic_name)

    return topics


def create_rule_based_summary(transcript_path: Path, transcript_text: str) -> MeetingSummary:
    # These simple keyword lists let us do beginner-friendly rule-based extraction.
    decision_keywords = ["decide", "decision", "agreed"]
    action_item_keywords = ["will", "todo", "follow up", "action item"]
    question_keywords = ["?", "question", "unclear"]

    decisions = []
    action_items = []
    open_questions = []

    cleaned_transcript_text = clean_transcript_text(transcript_text)

    # Check each transcript line for simple decision, action item, and question clues.
    for line in transcript_text.splitlines():
        source_timestamp = get_source_timestamp(line)
        clean_line = clean_transcript_text(line)

        # Skip blank lines so we do not add empty items.
        if not clean_line:
            continue

        lower_line = clean_line.lower()

        if any(keyword in lower_line for keyword in decision_keywords):
            decisions.append(
                Decision(
                    decision=clean_line,
                    source_timestamp=source_timestamp,
                )
            )

        if any(keyword in lower_line for keyword in action_item_keywords):
            action_items.append(
                ActionItem(
                    task=clean_line,
                    owner=guess_action_owner(clean_line),
                    due_date=None,
                    source_timestamp=source_timestamp,
                )
            )

        if any(keyword in lower_line for keyword in question_keywords):
            open_questions.append(clean_line)

    return MeetingSummary(
        title=transcript_path.stem.replace("_", " ").title(),
        tldr=get_first_sentence(cleaned_transcript_text),
        key_topics=extract_key_topics(cleaned_transcript_text),
        decisions=decisions,
        action_items=action_items,
        open_questions=open_questions,
        tags=[
            "meeting",
            "auto-summary",
        ],
    )


def create_ollama_summary(transcript_path: Path, transcript_text: str) -> MeetingSummary:
    cleaned_transcript_text = clean_transcript_text(transcript_text)

    # Ask Ollama to return JSON that matches our MeetingSummary shape.
    prompt = f"""
You are summarizing a meeting transcript.

Return only valid JSON with these keys:
title, tldr, key_topics, decisions, action_items, open_questions, tags

Rules:
1. Clean the transcript before summarizing.
2. Correct obvious technical term mistakes:
   - QDrin, Qdrin, and QDrant should become Qdrant.
   - co-pilot should become copilot.
3. Extract clean full-sentence items, not timestamp fragments.
4. Preserve owner names when action items are present.
5. Do not invent generic topics like teamwork or collaboration unless they are actually relevant in the transcript.
6. key_topics, decisions, action_items, open_questions, and tags must be lists.
7. each decision must be an object with decision and source_timestamp.
8. each action item must be an object with task, owner, due_date, and source_timestamp.
9. use null when owner, due_date, or source_timestamp is unknown.
10. Keep all text concise and factual.

Cleaned transcript:
{cleaned_transcript_text}
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
    summary_data.setdefault("tldr", get_first_sentence(cleaned_transcript_text))
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
                if decision.source_timestamp:
                    summary_file.write(
                        f"- {decision.decision} ({decision.source_timestamp})\n"
                    )
                else:
                    summary_file.write(f"- {decision.decision}\n")
        else:
            summary_file.write("- None found.\n")
        summary_file.write("\n")

        summary_file.write("## Action Items\n\n")
        if meeting_summary.action_items:
            for action_item in meeting_summary.action_items:
                details = []

                if action_item.owner:
                    details.append(f"Owner: {action_item.owner}")

                if action_item.due_date:
                    details.append(f"Due: {action_item.due_date}")

                if action_item.source_timestamp:
                    details.append(f"Source: {action_item.source_timestamp}")

                if details:
                    summary_file.write(
                        f"- {action_item.task} ({'; '.join(details)})\n"
                    )
                else:
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


def ask_ollama(question: str, context: str) -> str:
    # Build a prompt that asks Ollama for a direct, context-only answer.
    prompt = build_answer_prompt(question, context)

    print("Calling Ollama for question answering...")
    print(f"Ollama model: {OLLAMA_MODEL}")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )
    response.raise_for_status()

    answer = response.json()["response"].strip()
    print("Ollama response received.")
    return answer


def build_answer_prompt(question: str, context: str) -> str:
    # Keep the answer prompt in one helper so /ask and /query use the same logic.
    return f"""
You are answering questions about meeting memory.

Rules:
1. Use ONLY the provided sources.
2. Search across ALL sources, not just the first one.
3. If multiple sources contain relevant information, combine them.
4. Include names and owners exactly as written in the source.
5. Prefer direct factual answers over summaries.
6. Do not guess or add information that is not in the sources.
7. Answer using relevant context even if the wording is not exact.
8. Treat "decided", "agreed", "standardize", "focus on", and "next milestone" as decision signals.
9. Treat markdown section headings like "## Decisions", "## Action Items", and "## Open Questions" as highly reliable evidence.
10. Do not say the information is missing if a relevant source contains an item under a matching section.
11. For questions about "all", "what decisions", "what action items", or "open questions", list all relevant items found across all sources.
12. For questions like "summarize all discussions about X", summarize matching source content about X instead of requiring the exact phrase "discussion about X".
13. Treat SOURCE 1 as the primary source because it has the highest retrieval score.
14. Use lower-ranked sources only if they clearly mention the same topic as the user question.
15. If the question asks about a specific topic, do not include unrelated decisions, action items, or open questions from other sources.
16. For "summarize all discussions about product strategy", focus only on product roadmap, executive copilot, dashboard, user interviews, meeting recordings, and web vs desktop application.
17. Keep answers concise and directly tied to the user's topic.
18. If the answer is truly unrelated to all sources, answer exactly:
I could not find that information in the meeting memory.

Examples:
Question: Who should update API documentation?
Good answer: Mike was assigned to update the API documentation.

Question: What was decided for Phase 2?
Good answer: The team decided to continue using Faster Whisper for Phase 2.

Question: What decisions were made about Qdrant?
Good answer: The team decided to standardize on Qdrant as the vector database.

Question:
{question}

Retrieved sources:
{context}
"""


def extract_question_keywords(question: str):
    # Remove common words so we keep the useful topic words from the question.
    common_words = {
        "what",
        "who",
        "where",
        "when",
        "why",
        "how",
        "all",
        "about",
        "the",
        "is",
        "are",
        "were",
        "was",
        "did",
        "do",
        "does",
        "summarize",
        "discussions",
    }

    words = re.findall(r"[a-zA-Z0-9]+", question.lower())

    return [
        word
        for word in words
        if len(word) > 3 and word not in common_words
    ]


def get_payload_search_text(payload):
    # Search for keywords across the most useful source fields.
    file_name = payload.get("file_name", "")
    text = payload.get("text", "")

    # The markdown text usually contains title and tags, so this keeps things simple.
    return f"{file_name}\n{text}".lower()


def count_keyword_matches(search_text: str, question_keywords):
    # Count how many unique question keywords appear in this source.
    return sum(1 for keyword in question_keywords if keyword in search_text)


def select_sources_for_question(search_points, question: str):
    # Always include the top-ranked source, then filter lower-ranked sources.
    if not search_points:
        return []

    question_keywords = extract_question_keywords(question)
    selected_points = []

    print(f"Question keywords for source filtering: {question_keywords}")

    for index, point in enumerate(search_points):
        payload = point.payload or {}
        file_name = payload.get("file_name")
        search_text = get_payload_search_text(payload)
        keyword_match_count = count_keyword_matches(search_text, question_keywords)

        if question_keywords:
            keyword_match_ratio = keyword_match_count / len(question_keywords)
        else:
            keyword_match_ratio = 0

        # The first result is the highest-ranked source, so always keep it.
        if index == 0:
            selected_points.append(point)
            print(
                f"Source: {file_name} | "
                f"Qdrant score: {point.score} | "
                f"Keyword matches: {keyword_match_count} | "
                "Decision: keep top-ranked source"
            )
            continue

        # Keep extra sources only when they clearly match the question topic.
        should_keep = (
            keyword_match_count > 0
            and keyword_match_ratio >= MIN_KEYWORD_MATCH_RATIO
        )

        if should_keep:
            selected_points.append(point)
            decision = "keep keyword-relevant source"
        else:
            decision = "discard not enough keyword overlap"

        print(
            f"Source: {file_name} | "
            f"Qdrant score: {point.score} | "
            f"Keyword matches: {keyword_match_count} | "
            f"Decision: {decision}"
        )

    return selected_points


def retrieve_meeting_sources(question: str):
    # Load the same embedding model used when indexing the vault.
    print(f"Generating embedding with model: {EMBEDDING_MODEL_NAME}")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    query_vector = embedding_model.encode(question).tolist()
    print("Embedding generation complete.")

    # Search Qdrant for the most relevant meeting summary notes.
    print(f"Searching Qdrant collection: {QDRANT_COLLECTION}")
    qdrant_client = QdrantClient(url=QDRANT_URL)
    search_results = qdrant_client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=TOP_K_RESULTS,
        with_payload=True,
    )
    print(f"Qdrant search returned {len(search_results.points)} result(s).")
    return search_results.points


def build_primary_source_context(search_points):
    # Use only the highest-scoring source in the answer prompt.
    if not search_points:
        print("Selected primary source: none")
        return ""

    primary_source = search_points[0]
    primary_payload = primary_source.payload or {}
    primary_file_name = primary_payload.get("file_name")
    primary_text = primary_payload.get("text", "")

    print(f"Selected primary source: {primary_file_name}")

    return (
        "SOURCE 1:\n"
        f"File: {primary_file_name}\n"
        f"Content:\n{primary_text}"
    )


def build_sources_response(search_points):
    # Return the top retrieved sources for transparency.
    sources = []

    for result in search_points:
        payload = result.payload or {}
        sources.append(
            {
                "score": result.score,
                "file_name": payload.get("file_name"),
                "file_path": payload.get("file_path"),
            }
        )

    return sources


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


@app.post("/ask")
def ask_question(request: AskRequest):
    print(f"Received question: {request.question}")

    search_points = retrieve_meeting_sources(request.question)
    sources = build_sources_response(search_points)
    retrieved_context = build_primary_source_context(search_points)

    # Ask Ollama to answer using the retrieved context.
    answer = ask_ollama(request.question, retrieved_context)

    return {
        "answer": answer,
        "sources": sources,
    }


def stream_ollama_answer(prompt: str):
    print("Streaming started.")

    with requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": True,
        },
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()

        # Ollama streams one JSON object per line.
        for line in response.iter_lines():
            if not line:
                continue

            data = json.loads(line.decode("utf-8"))
            text_chunk = data.get("response", "")

            if text_chunk:
                yield f"data: {text_chunk}\n\n"

            if data.get("done"):
                break

    print("Streaming complete.")
    yield "data: [DONE]\n\n"


@app.post("/query")
def query_stream(request: AskRequest):
    print(f"Received query: {request.question}")

    search_points = retrieve_meeting_sources(request.question)
    retrieved_context = build_primary_source_context(search_points)
    prompt = build_answer_prompt(request.question, retrieved_context)

    return StreamingResponse(
        stream_ollama_answer(prompt),
        media_type="text/event-stream",
    )
