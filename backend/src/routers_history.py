from fastapi import APIRouter, HTTPException
from pathlib import Path
import json

from config import Configuration

router = APIRouter()

def get_notes_dir() -> Path:
    config = Configuration.from_env()
    notes_dir = Path(config.notes_workspace)
    # If starting from backend/src, resolve relative to cwd
    return notes_dir.resolve()

@router.get("/history")
def get_history():
    notes_dir = get_notes_dir()
    index_file = notes_dir / "notes_index.json"
    if not index_file.exists():
        return {"notes": []}
    try:
        with open(index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {"notes": data.get("notes", [])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read index: {e}")

@router.get("/history/{note_id}")
def get_history_detail(note_id: str):
    notes_dir = get_notes_dir()
    note_file = notes_dir / f"{note_id}.md"
    if not note_file.exists():
        raise HTTPException(status_code=404, detail="Note not found")
    try:
        with open(note_file, "r", encoding="utf-8") as f:
            return {"id": note_id, "content": f.read()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read note: {e}")
