# Team Memory

A local AI-powered meeting memory system that converts meeting videos into searchable structured knowledge.

## Features

- Extract audio from meeting videos using FFmpeg
- Extract video frames/screenshots every 5 seconds
- Transcribe speech locally using Faster Whisper
- Generate timestamped transcripts
- Coalesce transcript segments with nearest video frames
- Generate Obsidian-ready markdown meeting notes
- Fully local pipeline (no cloud required)

---

## Current Pipeline

```text
Video
→ Audio Extraction
→ Frame Extraction
→ Speech-to-Text
→ Timestamp Alignment
→ Structured JSON
→ Obsidian Markdown Note
```

---

## Tech Stack

- Python
- FFmpeg
- Faster Whisper
- Git/GitHub
- Markdown
- Obsidian-compatible notes

---

## Project Structure

```text
team-memory/
│
├── scripts/
│   └── process_video.py
│
├── videos/
├── audio/
├── frames/
├── transcripts/
├── notes/
│
├── README.md
└── .gitignore
```

---

## Example Outputs

### Transcript

```text
[00:00:00 - 00:00:03]
Hi everyone, my name is Dhruv Deshpande.
```

### Coalesced Segment

```json
{
  "start_time": "00:00:00",
  "end_time": "00:00:03",
  "text": "Hi everyone, my name is Dhruv Deshpande.",
  "nearest_frame": "frames/frame_0001.jpg"
}
```

---

## Future Roadmap

- FastAPI endpoints
- Folder watcher automation
- Dockerization
- Speaker diarization
- Vector database integration
- RAG-based meeting search
- Team query interface
- Agent workflows

---

## Status

Phase 1 (Capture Pipeline) in progress.