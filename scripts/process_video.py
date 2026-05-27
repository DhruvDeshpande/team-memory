import json
from datetime import date
from pathlib import Path
import subprocess
import sys
import time

from faster_whisper import WhisperModel


# Set up the paths used by this script.
DEFAULT_INPUT_VIDEO = Path("videos/sample_meeting.mp4")

# Change this number if you want more or fewer frames.
# For example, 5 means "save one frame every 5 seconds."
FRAME_INTERVAL_SECONDS = 5

# Change this if you want to use a different Faster Whisper model.
# "base" is a good beginner-friendly default.
WHISPER_MODEL = "base"

AUDIO_FOLDER = Path("audio")
FRAMES_FOLDER = Path("frames")
TRANSCRIPTS_FOLDER = Path("transcripts")
VAULT_FOLDER = Path("vault")
MEETINGS_FOLDER = VAULT_FOLDER / "meetings"
PEOPLE_FOLDER = VAULT_FOLDER / "people"
PROJECTS_FOLDER = VAULT_FOLDER / "projects"
DECISIONS_FOLDER = VAULT_FOLDER / "decisions"


def format_timestamp(seconds):
    # Convert seconds into a beginner-friendly HH:MM:SS timestamp.
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    remaining_seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"


def get_nearest_frame(
    start_seconds,
    total_frames,
    video_frames_folder,
    frame_interval_seconds,
):
    # Frames are extracted at a configurable interval.
    # If the interval is 5, 0-4 seconds use frame_0001.jpg,
    # 5-9 seconds use frame_0002.jpg, and so on.
    calculated_frame_number = int(start_seconds // frame_interval_seconds) + 1

    # If the transcript goes past the final extracted frame, use the last frame instead.
    frame_number = min(calculated_frame_number, total_frames)
    frame_filename = f"frame_{frame_number:04d}.jpg"
    return str(video_frames_folder / frame_filename)


def main():
    # Start a timer for the whole video processing pipeline.
    total_start_time = time.perf_counter()

    # Use the video path from the command line if one was provided.
    # Otherwise, use the default sample meeting video.
    if len(sys.argv) > 1:
        input_video = Path(sys.argv[1])
    else:
        input_video = DEFAULT_INPUT_VIDEO

    # Use the Whisper model from the command line if one was provided.
    # Otherwise, use the beginner-friendly default model.
    if len(sys.argv) > 2:
        whisper_model = sys.argv[2]
    else:
        whisper_model = WHISPER_MODEL

    # Use the video's filename without ".mp4" to create matching output filenames.
    video_name = input_video.stem
    meeting_title = video_name.replace("_", " ").title()

    # Build output paths based on the input video name.
    output_audio = AUDIO_FOLDER / f"{video_name}.wav"
    video_frames_folder = FRAMES_FOLDER / video_name
    output_frame_pattern = video_frames_folder / "frame_%04d.jpg"
    output_transcript = TRANSCRIPTS_FOLDER / f"{video_name}.txt"
    output_json = TRANSCRIPTS_FOLDER / f"{video_name}_coalesced.json"
    output_note = MEETINGS_FOLDER / f"{video_name}.md"

    # Create the output folders if they do not already exist.
    AUDIO_FOLDER.mkdir(exist_ok=True)
    FRAMES_FOLDER.mkdir(exist_ok=True)
    video_frames_folder.mkdir(exist_ok=True)
    TRANSCRIPTS_FOLDER.mkdir(exist_ok=True)
    VAULT_FOLDER.mkdir(exist_ok=True)
    MEETINGS_FOLDER.mkdir(exist_ok=True)
    PEOPLE_FOLDER.mkdir(exist_ok=True)
    PROJECTS_FOLDER.mkdir(exist_ok=True)
    DECISIONS_FOLDER.mkdir(exist_ok=True)
    print(
        "Output folders are ready: audio/, frames/, transcripts/, and vault folders/"
    )
    print(f"Processing video: {input_video}")
    print(f"Using Whisper model: {whisper_model}")

    # Use ffmpeg to extract the video's audio into a WAV file.
    audio_start_time = time.perf_counter()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            str(output_audio),
        ],
        check=True,
    )
    audio_extraction_time = time.perf_counter() - audio_start_time
    print(f"Audio extracted successfully: {output_audio}")

    # Use ffmpeg to save one video frame at the configured interval.
    frame_start_time = time.perf_counter()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(input_video),
            "-vf",
            f"fps=1/{FRAME_INTERVAL_SECONDS}",
            str(output_frame_pattern),
        ],
        check=True,
    )
    frame_extraction_time = time.perf_counter() - frame_start_time
    print(f"Frames extracted successfully: {output_frame_pattern}")

    # Count the actual frame files that ffmpeg created.
    frame_files = sorted(video_frames_folder.glob("frame_*.jpg"))
    total_frames = len(frame_files)

    # Stop early if no frames were created, because the JSON and note need frame paths.
    if total_frames == 0:
        raise RuntimeError("No frame files were created in the frames folder.")

    print(f"Found {total_frames} extracted frame file(s).")

    # Load the Faster Whisper speech-to-text model.
    transcription_start_time = time.perf_counter()
    print(f"Loading Faster Whisper model: {whisper_model}")
    model = WhisperModel(whisper_model)
    print(f"Faster Whisper model loaded successfully: {whisper_model}")

    # Transcribe the extracted audio file.
    print(f"Transcribing audio: {output_audio}")
    segments, info = model.transcribe(str(output_audio))

    # Store the transcript segments so we can write them to both text and JSON files.
    transcript_segments = []

    for segment in segments:
        transcript_segments.append(
            {
                "start_time": format_timestamp(segment.start),
                "end_time": format_timestamp(segment.end),
                "text": segment.text.strip(),
                "nearest_frame": get_nearest_frame(
                    segment.start,
                    total_frames,
                    video_frames_folder,
                    FRAME_INTERVAL_SECONDS,
                ),
            }
        )

    # Save each transcript segment with its start and end timestamps.
    with output_transcript.open("w", encoding="utf-8") as transcript_file:
        transcript_file.write(f"Detected language: {info.language}\n\n")

        for segment in transcript_segments:
            transcript_file.write(
                f"[{segment['start_time']} - {segment['end_time']}] {segment['text']}\n"
            )

    transcription_time = time.perf_counter() - transcription_start_time
    print(f"Transcript saved successfully: {output_transcript}")

    # Create a JSON file that connects the video, audio, transcript, frames, and segments.
    json_start_time = time.perf_counter()
    coalesced_data = {
        "video_file": str(input_video),
        "audio_file": str(output_audio),
        "transcript_file": str(output_transcript),
        "frames_folder": str(video_frames_folder),
        "frame_interval_seconds": FRAME_INTERVAL_SECONDS,
        "whisper_model": whisper_model,
        "segments": transcript_segments,
    }

    # Save the JSON with indentation so it is easy to read.
    with output_json.open("w", encoding="utf-8") as json_file:
        json.dump(coalesced_data, json_file, indent=2)

    json_generation_time = time.perf_counter() - json_start_time
    print(f"Coalesced JSON saved successfully: {output_json}")

    # Create an Obsidian-ready markdown note for this meeting.
    markdown_start_time = time.perf_counter()
    print(f"Creating Obsidian note: {output_note}")

    # This placeholder can be replaced later with an AI-generated or human-written summary.
    summary = (
        "This meeting discusses teamwork, collaboration, scheduling challenges, "
        "and improvements for future projects."
    )

    # Faster Whisper gives us the audio duration in seconds.
    duration_seconds = round(info.duration, 2)

    # Write YAML frontmatter, a summary section, and a timeline section.
    with output_note.open("w", encoding="utf-8") as note_file:
        # YAML frontmatter must be the very first thing in the markdown file.
        note_file.write("---\n")
        note_file.write(f"title: {meeting_title}\n")
        note_file.write(f"date: {date.today().isoformat()}\n")
        note_file.write(f"video_file: {input_video}\n")
        note_file.write(f"audio_file: {output_audio}\n")
        note_file.write(f"transcript_file: {output_transcript}\n")
        note_file.write(f"duration_seconds: {duration_seconds}\n")
        note_file.write("tags: []\n")
        note_file.write("attendees: []\n")
        note_file.write(f"coalesced_file: {output_json}\n")
        note_file.write(f"frames_folder: {video_frames_folder}/\n")
        note_file.write(f"frame_interval_seconds: {FRAME_INTERVAL_SECONDS}\n")
        note_file.write(f"whisper_model: {whisper_model}\n")
        note_file.write("---\n\n")

        note_file.write(f"# {meeting_title}\n\n")
        note_file.write("## Summary\n\n")
        note_file.write(f"{summary}\n\n")
        note_file.write("## Timeline\n\n")

        for segment in transcript_segments:
            note_file.write(
                f"### {segment['start_time']} - {segment['end_time']}\n"
            )
            note_file.write(f"{segment['text']}\n\n")
            note_file.write(f"Frame: {segment['nearest_frame']}\n\n")

    markdown_generation_time = time.perf_counter() - markdown_start_time
    print(f"Obsidian note saved successfully: {output_note}")

    # Stop the total pipeline timer after all work is complete.
    total_pipeline_time = time.perf_counter() - total_start_time

    # Print a clean timing summary so it is easy to see which steps took time.
    print("\n[Timing]")
    print(f"Whisper model: {whisper_model}")
    print(f"Audio extraction: {audio_extraction_time:.2f}s")
    print(f"Frame extraction: {frame_extraction_time:.2f}s")
    print(f"Transcription: {transcription_time:.2f}s")
    print(f"JSON generation: {json_generation_time:.2f}s")
    print(f"Markdown generation: {markdown_generation_time:.2f}s")
    print(f"Total pipeline: {total_pipeline_time:.2f}s")

    # Let the user know the whole process completed.
    print("Video processing complete.")


if __name__ == "__main__":
    main()
