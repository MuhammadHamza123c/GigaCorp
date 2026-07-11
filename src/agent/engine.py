import logging
from typing import Any

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import BaseMessage
from langchain.schema.vectorstore import VectorStore
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_community.chat_message_histories import ChatMessageHistory

from src.agent.prompts import (
    CONTEXTUALIZE_Q_PROMPT,
    CONTEXTUALIZE_Q_SYSTEM_PROMPT,
    QA_PROMPT,
    QA_SYSTEM_PROMPT,
)
from src.config import settings

logger = logging.getLogger(__name__)


_store: dict[str, BaseChatMessageHistory] = {}


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = ChatMessageHistory()
    return _store[session_id]


class RAGEngine:
    """Core RAG execution engine using LangChain + Groq."""

    def __init__(self, vector_store: VectorStore) -> None:
        self._vector_store = vector_store
        self._llm = ChatGroq(
            model=settings.GROQ_MODEL,
            groq_api_key=settings.GROQ_API_KEY,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
        )
        self._retriever = self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6},
        )
        self._chain = self._build_chain()

    def _build_chain(self) -> RunnableWithMessageHistory:
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", CONTEXTUALIZE_Q_PROMPT),
            ]
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", QA_SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", QA_PROMPT),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            self._llm, self._retriever, contextualize_q_prompt
        )

        question_answer_chain = create_stuff_documents_chain(
            self._llm, qa_prompt
        )

        rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )

        return RunnableWithMessageHistory(
            rag_chain,
            _get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    def invoke(
        self, query: str, session_id: str = "default"
    ) -> dict[str, Any]:
        """Process a user query and return the answer with source metadata.

        Returns a dict with keys:
            - answer: str — the LLM's generated text response
            - sources: list[dict] — metadata from each retrieved document
        """
        try:
            result = self._chain.invoke(
                {"input": query},
                config={"configurable": {"session_id": session_id}},
            )
        except Exception as exc:
            logger.exception("RAG chain invocation failed for session %s", session_id)
            return {
                "answer": "Sorry, an internal error occurred while processing your request.",
                "sources": [],
            }

        raw_answer: str = result.get("answer", "")
        raw_docs: list[Any] = result.get("context", [])

        sources: list[dict[str, Any]] = []
        seen: set[str] = set()
        for doc in raw_docs:
            section = doc.metadata.get("section", "Unknown")
            chunk_id = doc.metadata.get("chunk_id", -1)
            key = f"{section}:{chunk_id}"
            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "section": section,
                        "chunk_id": chunk_id,
                        "source_file": doc.metadata.get("source", ""),
                        "start_line": doc.metadata.get("start_line", 0),
                        "content_preview": doc.page_content[:200],
                    }
                )

        return {"answer": raw_answer, "sources": sources}

    @property
    def store(self) -> dict[str, BaseChatMessageHistory]:
        return _store
