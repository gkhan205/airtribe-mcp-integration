"""
fs_tools.py

Simple file system tools that the LLM can use.
Supports reading .txt, .pdf and .docx files.
"""

import os
from datetime import datetime

from docx import Document
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".docx"]


def get_file_info(filepath):
    """Return basic information (metadata) about a file."""
    stats = os.stat(filepath)
    modified = datetime.fromtimestamp(stats.st_mtime)
    return {
        "name": os.path.basename(filepath),
        "path": filepath,
        "size_bytes": stats.st_size,
        "modified": modified.strftime("%Y-%m-%d %H:%M:%S"),
    }


def read_pdf(filepath):
    """Get the text from every page of a PDF file."""
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text.strip()


def read_docx(filepath):
    """Get the text from every paragraph of a Word (.docx) file."""
    document = Document(filepath)
    lines = []
    for paragraph in document.paragraphs:
        if paragraph.text.strip() != "":
            lines.append(paragraph.text)
    return "\n".join(lines)


def read_txt(filepath):
    """Get the text from a plain text file."""
    with open(filepath, "r", encoding="utf-8") as file:
        return file.read()


def read_file(filepath: str) -> dict:
    """
    Read a resume file (PDF, TXT or DOCX) and return its text.

    Returns a dictionary like:
        {"success": True, "content": "...", "metadata": {...}}
    or, if something goes wrong:
        {"success": False, "error": "..."}
    """
    if not os.path.exists(filepath):
        return {"success": False, "error": f"File not found: {filepath}"}

    if not os.path.isfile(filepath):
        return {"success": False, "error": f"{filepath} is not a file"}

    extension = os.path.splitext(filepath)[1].lower()
    if extension not in SUPPORTED_EXTENSIONS:
        return {
            "success": False,
            "error": f"Unsupported file type '{extension}'. Supported types: {SUPPORTED_EXTENSIONS}",
        }

    try:
        if extension == ".pdf":
            content = read_pdf(filepath)
        elif extension == ".docx":
            content = read_docx(filepath)
        else:
            content = read_txt(filepath)
    except Exception as error:
        return {"success": False, "error": f"Could not read {filepath}: {error}"}

    metadata = get_file_info(filepath)
    metadata["extension"] = extension
    metadata["word_count"] = len(content.split())

    return {"success": True, "content": content, "metadata": metadata}


def list_files(directory: str, extension: str = None) -> list:
    """
    List all files in a directory.

    If an extension is given (like ".pdf" or "pdf"), only those files are returned.
    Returns an empty list if the directory does not exist.
    """
    if not os.path.isdir(directory):
        return []

    if extension and not extension.startswith("."):
        extension = "." + extension

    files = []
    for name in sorted(os.listdir(directory)):
        path = os.path.join(directory, name)

        # skip folders and hidden files like .DS_Store
        if not os.path.isfile(path) or name.startswith("."):
            continue

        if extension and not name.lower().endswith(extension.lower()):
            continue

        files.append(get_file_info(path))

    return files


def write_file(filepath: str, content: str) -> dict:
    """
    Write text to a file. Creates the folders if they don't exist yet.
    """
    try:
        folder = os.path.dirname(filepath)
        if folder:
            os.makedirs(folder, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as file:
            file.write(content)

        return {
            "success": True,
            "path": filepath,
            "message": f"Saved {len(content)} characters to {filepath}",
        }
    except Exception as error:
        return {"success": False, "error": f"Could not write {filepath}: {error}"}


def search_in_file(filepath: str, keyword: str) -> dict:
    """
    Search for a keyword in a file (case-insensitive).

    Every line that contains the keyword is returned along with
    the line before and after it as context.
    """
    if not keyword or keyword.strip() == "":
        return {"success": False, "error": "Keyword cannot be empty"}

    result = read_file(filepath)
    if not result["success"]:
        return result

    keyword = keyword.strip()
    lines = result["content"].split("\n")
    matches = []

    for index, line in enumerate(lines):
        if keyword.lower() in line.lower():
            start = max(0, index - 1)
            end = min(len(lines), index + 2)
            context = " ".join(lines[start:end])
            matches.append(
                {
                    "line_number": index + 1,
                    "line": line.strip(),
                    "context": context.strip(),
                }
            )

    return {
        "success": True,
        "file": filepath,
        "keyword": keyword,
        "match_count": len(matches),
        "matches": matches,
    }
