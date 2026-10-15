"""Audio file responses; ``FileResponse`` answers ``Range`` requests, so long files can be seeked."""

from pathlib import Path
from urllib.parse import quote

from fastapi.responses import FileResponse

MEDIA_TYPES = {
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".mp3": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".aac": "audio/aac",
    ".webm": "audio/webm",
}


def content_disposition(kind: str, filename: str) -> str:
    ascii_name = filename.encode("ascii", "replace").decode().replace('"', "'")
    return f"{kind}; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


def audio_response(path: Path, download: bool = False) -> FileResponse:
    headers = {"Cache-Control": "private, no-cache", "Content-Disposition": content_disposition("attachment" if download else "inline", path.name)}
    return FileResponse(path, media_type=MEDIA_TYPES.get(path.suffix.lower(), "application/octet-stream"), headers=headers)
