import json
from datetime import date
from pathlib import Path
import subprocess

from faster_whisper import WhisperModel


# Set up the paths used by this script.
INPUT_VIDEO = Path("videos/sample_meeting.mp4")
AUDIO_FOLDER = Path("audio")
FRAMES_FOLDER = Path("frames")
TRANSCRIPTS_FOLDER = Path("transcripts")
NOTES_FOLDER = Path("notes")
OUTPUT_AUDIO = AUDIO_FOLDER / "sample_meeting.wav"
OUTPUT_FRAME_PATTERN = FRAMES_FOLDER / "frame_%04d.jpg"
OUTPUT_TRANSCRIPT = TRANSCRIPTS_FOLDER / "sample_meeting.txt"
OUTPUT_JSON = TRANSCRIPTS_FOLDER / "sample_meeting_coalesced.json"
OUTPUT_NOTE = NOTES_FOLDER / "sample_meeting.md"


def format_timestamp(seconds):
    # Convert seconds into a beginner-friendly HH:MM:SS timestamp.
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def get_nearest_frame(start_seconds, total_frames):
    # Frames are extracted every 5 seconds.
    # 0-4 seconds use frame_0001.jpg, 5-9 seconds use frame_0002.jpg, and so on.
    calculated_frame_number = int(start_seconds // 5) + 1

    # If the transcript goes past the final extracted frame, use the last frame instead.
    frame_number = min(calculated_frame_number, total_frames)
    return f"frame_{frame_number:04d}.jpg"


def main():
    # Create the output folders if they do not already exist.
    AUDIO_FOLDER.mkdir(exist_ok=True)
    FRAMES_FOLDER.mkdir(exist_ok=True)
    TRANSCRIPTS_FOLDER.mkdir(exist_ok=True)
    NOTES_FOLDER.mkdir(exist_ok=True)
    print("Output folders are ready: audio/, frames/, transcripts/, and notes/")

    # Use ffmpeg to extract the video's audio into a WAV file.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(INPUT_VIDEO),
            str(OUTPUT_AUDIO),
        ],
        check=True,
    )
    print(f"Audio extracted successfully: {OUTPUT_AUDIO}")

    # Use ffmpeg to save one video frame every 5 seconds as a JPG image.
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(INPUT_VIDEO),
            "-vf",
            "fps=1/5",
            str(OUTPUT_FRAME_PATTERN),
        ],
        check=True,
    )
    print(f"Frames extracted successfully: {OUTPUT_FRAME_PATTERN}")

    # Count the actual frame files that ffmpeg created.
    frame_files = sorted(FRAMES_FOLDER.glob("frame_*.jpg"))
    total_frames = len(frame_files)

    # Stop early if no frames were created, because the JSON and note need frame paths.
    if total_frames == 0:
        raise RuntimeError("No frame files were created in the frames folder.")

    print(f"Found {total_frames} extracted frame file(s).")

    # Load the Faster Whisper speech-to-text model.
    print("Loading Faster Whisper model...")
    model = WhisperModel("base")
    print("Faster Whisper model loaded successfully.")

    # Transcribe the extracted audio file.
    print(f"Transcribing audio: {OUTPUT_AUDIO}")
    segments, info = model.transcribe(str(OUTPUT_AUDIO))

    # Store the transcript segments so we can write them to both text and JSON files.
    transcript_segments = []

    for segment in segments:
        transcript_segments.append(
            {
                "start_time": format_timestamp(segment.start),
                "end_time": format_timestamp(segment.end),
                "text": segment.text.strip(),
                "nearest_frame": get_nearest_frame(segment.start, total_frames),
            }
        )

    # Save each transcript segment with its start and end timestamps.
    with OUTPUT_TRANSCRIPT.open("w", encoding="utf-8") as transcript_file:
        transcript_file.write(f"Detected language: {info.language}\n\n")

        for segment in transcript_segments:
            transcript_file.write(
                f"[{segment['start_time']} - {segment['end_time']}] {segment['text']}\n"
            )

    print(f"Transcript saved successfully: {OUTPUT_TRANSCRIPT}")

    # Create a JSON file that connects the video, audio, transcript, frames, and segments.
    coalesced_data = {
        "video_file": str(INPUT_VIDEO),
        "audio_file": str(OUTPUT_AUDIO),
        "transcript_file": str(OUTPUT_TRANSCRIPT),
        "frames_folder": str(FRAMES_FOLDER),
        "segments": transcript_segments,
    }

    # Save the JSON with indentation so it is easy to read.
    with OUTPUT_JSON.open("w", encoding="utf-8") as json_file:
        json.dump(coalesced_data, json_file, indent=2)

    print(f"Coalesced JSON saved successfully: {OUTPUT_JSON}")

    # Create an Obsidian-ready markdown note for this meeting.
    print(f"Creating Obsidian note: {OUTPUT_NOTE}")

    # This placeholder can be replaced later with an AI-generated or human-written summary.
    summary = (
        "This meeting discusses teamwork, collaboration, scheduling challenges, "
        "and improvements for future projects."
    )

    # Write YAML frontmatter, a summary section, and a timeline section.
    with OUTPUT_NOTE.open("w", encoding="utf-8") as note_file:
        note_file.write("---\n")
        note_file.write("title: Sample Meeting\n")
        note_file.write(f"date: {date.today().isoformat()}\n")
        note_file.write(f"source_video: {INPUT_VIDEO}\n")
        note_file.write(f"audio_file: {OUTPUT_AUDIO}\n")
        note_file.write(f"transcript_file: {OUTPUT_TRANSCRIPT}\n")
        note_file.write(f"coalesced_file: {OUTPUT_JSON}\n")
        note_file.write(f"frames_folder: {FRAMES_FOLDER}/\n")
        note_file.write("---\n\n")

        note_file.write("# Sample Meeting\n\n")
        note_file.write("## Summary\n\n")
        note_file.write(f"{summary}\n\n")
        note_file.write("## Timeline\n\n")

        for segment in transcript_segments:
            frame_path = FRAMES_FOLDER / segment["nearest_frame"]
            note_file.write(
                f"### {segment['start_time']} - {segment['end_time']}\n"
            )
            note_file.write(f"{segment['text']}\n\n")
            note_file.write(f"Frame: {frame_path}\n\n")

    print(f"Obsidian note saved successfully: {OUTPUT_NOTE}")

    # Let the user know the whole process completed.
    print("Video processing complete.")


if __name__ == "__main__":
    main()
