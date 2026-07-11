import logging
import sys
import uuid
from pathlib import Path
from typing import Any

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.agent.engine import RAGEngine
from src.database.vector_store import build_or_load_vector_store
from src.config import settings

logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="GigaCorp | Support Agent",
    page_icon="⚡",
    layout="centered",
    initial_sidebar_state="expanded",
)

_CUSTOM_CSS = """
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    .main > .block-container {
        padding-top: 1rem; padding-bottom: 1rem; max-width: 800px;
        background: transparent;
    }
    .header {
        background: rgba(255,255,255,0.06);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 1.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
        text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    .header h1 { color: #e94560; font-size: 1.8rem; font-weight: 700; margin: 0; letter-spacing: 0.5px; }
    .header p { color: #a0aec0; font-size: 0.9rem; margin: 0.3rem 0 0 0; opacity: 0.85; }
    .stChatMessage { border-radius: 12px !important; margin-bottom: 0.5rem !important; }
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #1e3a5f 0%, #16213e 100%) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #e2e8f0 !important;
    }
    .stChatMessage[data-testid="user-message"] p { color: #e2e8f0 !important; }
    .stChatMessage[data-testid="assistant-message"] {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: #e2e8f0 !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1) !important;
    }
    .stChatMessage[data-testid="assistant-message"] p { color: #e2e8f0 !important; }
    .stChatInputContainer {
        border-top: 1px solid rgba(255,255,255,0.1) !important;
        padding-top: 1rem !important;
        background: rgba(255,255,255,0.06) !important;
        border-radius: 0 0 16px 16px !important;
    }
    section[data-testid="stSidebar"] { background: rgba(0,0,0,0.3); backdrop-filter: blur(8px); }
    section[data-testid="stSidebar"] .st-emotion-cache-1cy7q6m { color: #e2e8f0; }
    section[data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #e94560; }
    .source-badge {
        background: #e94560; color: white; padding: 2px 10px; border-radius: 12px;
        font-size: 0.7rem; font-weight: 600; display: inline-block; margin-right: 4px;
    }
    .source-badge-section {
        background: #0f3460; color: white; padding: 2px 10px; border-radius: 12px;
        font-size: 0.7rem; font-weight: 600; display: inline-block;
    }
    .sources-container {
        background: rgba(255,255,255,0.06); border-radius: 10px; padding: 0.8rem;
        margin-top: 0.5rem; border-left: 3px solid #e94560;
    }
    .sources-container p { margin: 0; font-size: 0.85rem; color: #cbd5e1; }
    .status-dot {
        height: 8px; width: 8px; background-color: #22c55e; border-radius: 50%;
        display: inline-block; margin-right: 6px; animation: pulse 2s infinite;
    }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.5; } 100% { opacity: 1; } }
    .custom-divider { border: none; height: 1px; background: linear-gradient(90deg, transparent, #e94560, transparent); margin: 1rem 0; }
    .footer { text-align: center; font-size: 0.7rem; color: #94a3b8; margin-top: 2rem; padding: 0.5rem; }

</style>
"""

_QUICK_ACTIONS = [
    ("🚚", "What are your shipping options?"),
    ("🔄", "How do I return an item?"),
    ("🕐", "What are your business hours?"),
    ("⭐", "Tell me about Premium tier"),
]

_SESSION_ID_KEY = "_session_id"
_MESSAGES_KEY = "messages"
_ENGINE_KEY = "_engine"
_CONV_KEY = "conversation_count"
_PENDING_QA_KEY = "_pending_qa"


def _init_session_state() -> None:
    if _SESSION_ID_KEY not in st.session_state:
        st.session_state[_SESSION_ID_KEY] = str(uuid.uuid4())
    if _MESSAGES_KEY not in st.session_state:
        st.session_state[_MESSAGES_KEY] = []
    if _ENGINE_KEY not in st.session_state:
        try:
            vector_store = build_or_load_vector_store()
            st.session_state[_ENGINE_KEY] = RAGEngine(vector_store)
        except Exception as exc:
            logger.exception("Failed to initialise RAG engine")
            st.error(
                "Could not initialise the knowledge base. "
                "Ensure GROQ_API_KEY is set in .env and the FAQ file exists."
            )
            st.exception(exc)
            st.stop()
    if _CONV_KEY not in st.session_state:
        st.session_state[_CONV_KEY] = 0


def _render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    with st.expander("📄 View Sources", expanded=False):
        for i, src in enumerate(sources, 1):
            section = src.get("section", "Unknown")
            lines = src.get("start_line", 0)
            preview = src.get("content_preview", "")[:150]
            st.markdown(
                f"""<div class="sources-container">
                    <span class="source-badge">#{i}</span>
                    <span class="source-badge-section">{section}</span>
                    <span style="color:#64748b;font-size:0.75rem;"> ~ Line {lines}</span>
                    <p style="margin-top:6px;">{preview}…</p>
                </div>""",
                unsafe_allow_html=True,
            )


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            "<h2 style='color:#e94560;margin-bottom:0;'>⚡ GigaCorp</h2>"
            "<p style='color:#94a3b8;font-size:0.85rem;margin-top:0;'>Support Agent</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#e2e8f0;font-size:0.85rem;'>"
            "<span class='status-dot'></span> Connected</p>",
            unsafe_allow_html=True,
        )
        conv_count = st.session_state.get(_CONV_KEY, 0)
        st.markdown(
            f"<p style='color:#94a3b8;font-size:0.8rem;'>"
            f"Messages this session: <strong style='color:#e2e8f0;'>{conv_count}</strong></p>",
            unsafe_allow_html=True,
        )
        model_name = settings.GROQ_MODEL.split("/")[-1] if "/" in settings.GROQ_MODEL else settings.GROQ_MODEL
        st.markdown(
            f"<p style='color:#94a3b8;font-size:0.75rem;'>Model: {model_name}</p>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#e2e8f0;font-size:0.85rem;font-weight:600;'>💡 Quick Tips</p>",
            unsafe_allow_html=True,
        )
        for tip in ["Ask about shipping times & costs", "Check return policy details",
                     "Compare service tiers", "Inquire about warranty"]:
            st.markdown(f"<p style='color:#94a3b8;font-size:0.8rem;margin:2px 0;'>• {tip}</p>",
                        unsafe_allow_html=True)
        st.markdown("<hr class='custom-divider'>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state[_MESSAGES_KEY] = []
            st.session_state[_CONV_KEY] = 0
            st.session_state.pop(_PENDING_QA_KEY, None)
            if _SESSION_ID_KEY in st.session_state:
                st.session_state[_SESSION_ID_KEY] = str(uuid.uuid4())
            st.rerun()
        st.markdown("<div class='footer'>v1.0 · Powered by Groq + LangChain</div>",
                    unsafe_allow_html=True)


def _render_suggestions() -> None:
    if st.session_state.get(_MESSAGES_KEY):
        return
    cols = st.columns(len(_QUICK_ACTIONS))
    for i, (col, (_, text)) in enumerate(zip(cols, _QUICK_ACTIONS)):
        with col:
            if st.button(text, key=f"qa_{i}", use_container_width=True):
                st.session_state[_MESSAGES_KEY].append({"role": "user", "content": text})
                st.session_state[_CONV_KEY] = st.session_state.get(_CONV_KEY, 0) + 1
                st.session_state[_PENDING_QA_KEY] = text
                st.rerun()


def _process_pending() -> None:
    pending = st.session_state.pop(_PENDING_QA_KEY, None)
    if not pending:
        return
    engine: RAGEngine = st.session_state[_ENGINE_KEY]
    session_id: str = st.session_state[_SESSION_ID_KEY]
    with st.chat_message("assistant"):
        with st.status("🤔 Thinking…", expanded=False) as status:
            st.markdown("🔍 Retrieving relevant documents…")
            result = engine.invoke(pending, session_id=session_id)
            status.update(label="✅ Done", expanded=False)
        answer: str = result.get("answer", "")
        sources: list[dict[str, Any]] = result.get("sources", [])
        st.markdown(answer)
        _render_sources(sources)
    st.session_state[_MESSAGES_KEY].append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


def main() -> None:
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    _render_sidebar()
    st.markdown(
        """<div class="header">
            <h1>⚡ GigaCorp Support Agent</h1>
            <p>Ask me about shipping, returns, business hours, service tiers, orders & warranties</p>
        </div>""",
        unsafe_allow_html=True,
    )
    _init_session_state()

    for message in st.session_state[_MESSAGES_KEY]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                _render_sources(message["sources"])

    _process_pending()
    _render_suggestions()

    if prompt := st.chat_input("Type your question about GigaCorp services…"):
        st.session_state[_MESSAGES_KEY].append({"role": "user", "content": prompt})
        st.session_state[_CONV_KEY] = st.session_state.get(_CONV_KEY, 0) + 1
        with st.chat_message("user"):
            st.markdown(prompt)
        engine: RAGEngine = st.session_state[_ENGINE_KEY]
        session_id: str = st.session_state[_SESSION_ID_KEY]
        with st.chat_message("assistant"):
            with st.status("🤔 Thinking…", expanded=False) as status:
                st.markdown("🔍 Retrieving relevant documents…")
                result = engine.invoke(prompt, session_id=session_id)
                status.update(label="✅ Done", expanded=False)
            answer: str = result.get("answer", "")
            sources: list[dict[str, Any]] = result.get("sources", [])
            st.markdown(answer)
            _render_sources(sources)
        st.session_state[_MESSAGES_KEY].append(
            {"role": "assistant", "content": answer, "sources": sources}
        )
        st.rerun()


if __name__ == "__main__":
    main()
