from pathlib import Path
import shutil

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
    save_path = VIDEOS_DIR / file.filename

    with save_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {
        "filename": file.filename,
        "saved_to": str(save_path),
        "status": "video uploaded and saved successfully"
    }