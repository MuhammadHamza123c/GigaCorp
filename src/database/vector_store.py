import logging
import re
from pathlib import Path
from typing import Any

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import settings

logger = logging.getLogger(__name__)

_DEFAULT_EMBEDDINGS: HuggingFaceEmbeddings | None = None


def _get_embeddings() -> HuggingFaceEmbeddings:
    global _DEFAULT_EMBEDDINGS
    if _DEFAULT_EMBEDDINGS is None:
        _DEFAULT_EMBEDDINGS = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _DEFAULT_EMBEDDINGS


def _parse_section_headers(text: str) -> list[tuple[int, str]]:
    """Identify section header lines and their 0-indexed line numbers."""
    headers: list[tuple[int, str]] = []
    section_pattern = re.compile(r"^={5,}\s*\nSECTION:\s*(.+)\n={5,}", re.MULTILINE)
    for match in section_pattern.finditer(text):
        line_number = text[: match.start()].count("\n")
        headers.append((line_number, match.group(1).strip()))
    return headers


def _assign_metadata(documents: list[Document], raw_text: str) -> list[Document]:
    """Assign section title and line-range metadata to each chunk."""
    headers = _parse_section_headers(raw_text)
    if not headers:
        for i, doc in enumerate(documents):
            doc.metadata.setdefault("section", "General")
        return documents

    for doc in documents:
        start_line = doc.metadata.get("start_line", 0)
        current_section = "General"
        for line_no, section_name in sorted(headers, key=lambda x: x[0]):
            if line_no <= start_line:
                current_section = section_name
        doc.metadata["section"] = current_section

    return documents


def build_or_load_vector_store() -> FAISS:
    """Build a FAISS index from the FAQ file or load an existing one."""
    embeddings = _get_embeddings()
    index_path: Path = settings.FAISS_INDEX_DIR

    if index_path.exists() and (index_path / "index.faiss").exists():
        logger.info("Loading existing FAISS index from %s", index_path)
        return FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    logger.info("Building FAISS index from FAQ file: %s", settings.FAQ_PATH)
    if not settings.FAQ_PATH.is_file():
        raise FileNotFoundError(f"FAQ file not found at {settings.FAQ_PATH}")

    raw_text: str = settings.FAQ_PATH.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )

    raw_docs: list[Document] = splitter.create_documents(
        [raw_text],
        metadatas=[{"source": str(settings.FAQ_PATH.name)}],
    )

    for i, doc in enumerate(raw_docs):
        offset = raw_text.find(doc.page_content[:50])
        line_number = raw_text[:offset].count("\n") if offset != -1 else 0
        doc.metadata["chunk_id"] = i
        doc.metadata["start_line"] = line_number

    documents = _assign_metadata(raw_docs, raw_text)

    vector_store = FAISS.from_documents(documents, embeddings)
    index_path.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(index_path))
    logger.info("FAISS index saved to %s", index_path)

    return vector_store
