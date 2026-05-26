from pathlib import Path
import shutil
import subprocess

from fastapi import FastAPI, UploadFile, File

app = FastAPI()

VIDEOS_DIR = Path("videos")
VIDEOS_DIR.mkdir(exist_ok=True)


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