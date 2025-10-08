import fitz  # PyMuPDF


def extract_text(pdf_bytes: bytes) -> str:
    """Extract text from PDF bytes using PyMuPDF and return as single string."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = []
    for page in doc:
        try:
            texts.append(page.get_text())
        except Exception:
            texts.append("")
    return "\n\n".join(texts)


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200):
    """Simple character-based chunking with overlap."""
    if not text:
        return []
    chunks = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= L:
            break
    return chunks
