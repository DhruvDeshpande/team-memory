from pathlib import Path
import subprocess
import time

from apscheduler.schedulers.background import BackgroundScheduler


# Set up the paths used by this watcher script.
VIDEOS_FOLDER = Path("videos")
PROCESSED_VIDEOS_FILE = Path("processed_videos.txt")
PROCESS_VIDEO_SCRIPT = Path("scripts/process_video.py")


def load_processed_videos():
    # If the file does not exist yet, no videos have been processed.
    if not PROCESSED_VIDEOS_FILE.exists():
        return set()

    # Read each processed filename into a set so lookups are fast and simple.
    with PROCESSED_VIDEOS_FILE.open("r", encoding="utf-8") as file:
        return {line.strip() for line in file if line.strip()}


def save_processed_video(video_filename):
    # Add one processed video filename to the tracking file.
    with PROCESSED_VIDEOS_FILE.open("a", encoding="utf-8") as file:
        file.write(f"{video_filename}\n")


def check_videos_folder():
    print("Checking videos folder...")

    # Make sure the videos folder exists before looking for files inside it.
    VIDEOS_FOLDER.mkdir(exist_ok=True)

    # Load the filenames we have already processed.
    processed_videos = load_processed_videos()

    # Find every .mp4 file in the videos folder.
    video_files = sorted(VIDEOS_FOLDER.glob("*.mp4"))

    # Keep only the videos that are not listed in processed_videos.txt.
    new_videos = [
        video_file
        for video_file in video_files
        if video_file.name not in processed_videos
    ]

    if not new_videos:
        print("No new videos found.")
        return

    for video_file in new_videos:
        print(f"New video found: {video_file.name}")

        # Run the main video processing script for this specific video file.
        subprocess.run(
            ["python3", str(PROCESS_VIDEO_SCRIPT), str(video_file)],
            check=True,
        )

        # Only mark the video as processed if the command above finished successfully.
        save_processed_video(video_file.name)
        print(f"Processing complete: {video_file.name}")


def main():
    # Create a scheduler that runs in the background while this script stays open.
    scheduler = BackgroundScheduler()

    # Check the videos folder every 10 seconds.
    scheduler.add_job(check_videos_folder, "interval", seconds=10)
    scheduler.start()

    print("Video watcher started. Press Ctrl+C to stop.")

    try:
        # Keep the script running so APScheduler can continue checking the folder.
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Stop the scheduler cleanly when the user presses Ctrl+C.
        scheduler.shutdown()
        print("Video watcher stopped.")


if __name__ == "__main__":
    main()
