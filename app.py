"""
Simple dark-mode Streamlit UI for exploring the LangGraph workflow.
Run with: streamlit run app.py
"""

from __future__ import annotations

import os
import re

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent import MessagesState, agent


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

st.set_page_config(
    page_title="LangGraph RAG Studio",
    page_icon="🌙",
    layout="wide",
)


def inject_styles() -> None:
    """Apply a simple dark theme while keeping the UI close to the original layout."""
    st.markdown(
        """
        <style>
        :root {
            --bg: #0e1117;
            --panel: #151a23;
            --panel-2: #1b2230;
            --text: #ecf3ff;
            --muted: #98a7bd;
            --line: rgba(167, 184, 210, 0.16);
            --accent: #63b3ed;
            --accent-2: #7dd3fc;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(99, 179, 237, 0.12), transparent 24%),
                linear-gradient(180deg, #0b0f15 0%, #111827 100%);
            color: var(--text);
        }

        .block-container {
            max-width: 1180px;
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        h1, h2, h3, p, li, label, div {
            color: var(--text);
        }

        [data-testid="stSidebar"] {
            background: rgba(17, 24, 39, 0.92);
            border-right: 1px solid var(--line);
        }

        [data-testid="stChatMessage"] {
            background: rgba(21, 26, 35, 0.82);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 0.35rem 0.4rem;
        }

        .hero {
            background: linear-gradient(135deg, rgba(21, 26, 35, 0.95), rgba(27, 34, 48, 0.92));
            border: 1px solid var(--line);
            border-radius: 22px;
            padding: 1.25rem 1.35rem;
            margin-bottom: 1rem;
            box-shadow: 0 16px 40px rgba(0, 0, 0, 0.28);
        }

        .hero-tag {
            display: inline-block;
            padding: 0.3rem 0.65rem;
            border-radius: 999px;
            background: rgba(99, 179, 237, 0.14);
            color: var(--accent-2);
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }

        .hero-title {
            font-size: 2rem;
            margin: 0.75rem 0 0.45rem 0;
            letter-spacing: -0.02em;
        }

        .hero-copy {
            color: var(--muted);
            margin: 0;
            max-width: 52rem;
        }

        .summary-card {
            background: rgba(21, 26, 35, 0.9);
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 1rem;
            margin-bottom: 1rem;
        }

        .metric-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin-bottom: 0.8rem;
        }

        .metric-box {
            background: rgba(99, 179, 237, 0.06);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.25rem;
        }

        .metric-value {
            color: var(--text);
            font-size: 1.35rem;
            font-weight: 700;
        }

        .source-block {
            background: rgba(27, 34, 48, 0.86);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 0.8rem 0.9rem;
            margin-bottom: 0.7rem;
        }

        .source-title {
            font-weight: 700;
            margin-bottom: 0.28rem;
        }

        .source-meta {
            color: var(--muted);
            font-size: 0.84rem;
            margin-bottom: 0.38rem;
        }

        .source-preview {
            color: var(--text);
            font-size: 0.92rem;
            line-height: 1.45;
        }

        @media (max-width: 900px) {
            .metric-grid {
                grid-template-columns: 1fr;
            }

            .hero-title {
                font-size: 1.6rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_inline_latex(text: str) -> str:
    """Convert inline LaTeX delimiters into Streamlit-friendly markdown math."""
    return re.sub(r"\\\((.*?)\\\)", r"$\1$", text, flags=re.DOTALL)


def render_content(content: str) -> None:
    """Render mixed prose and block LaTeX without flattening equations."""
    pattern = re.compile(r"(\\\[(.*?)\\\]|\$\$(.*?)\$\$)", re.DOTALL)
    last_end = 0

    for match in pattern.finditer(content):
        prose = content[last_end:match.start()]
        if prose.strip():
            st.markdown(normalize_inline_latex(prose))

        latex = match.group(2) if match.group(2) is not None else match.group(3)
        if latex and latex.strip():
            st.latex(latex.strip())
        last_end = match.end()

    tail = content[last_end:]
    if tail.strip() or not content.strip():
        st.markdown(normalize_inline_latex(tail))


def build_message_history() -> list[HumanMessage | AIMessage]:
    """Rebuild LangChain messages from Streamlit session state."""
    messages: list[HumanMessage | AIMessage] = []
    for role, content in st.session_state.chat_history:
        if role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))
    return messages


def render_source_card(doc: dict) -> None:
    """Render a compact source/evidence card."""
    preview = doc["content"][:220] + ("..." if len(doc["content"]) > 220 else "")
    st.markdown(
        f"""
        <div class="source-block">
            <div class="source-title">{doc["source"]}</div>
            <div class="source-meta">
                section {doc.get("section", "unknown")} |
                strategy {doc.get("chunk_strategy", "unknown")} |
                score {doc["score"]:.4f}
            </div>
            <div class="source-preview">{preview}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_latest_run(run: dict) -> None:
    """Render the summary card for the latest execution."""
    grounding = run.get("retrieval_grade", "n/a") if run.get("route") == "knowledge_base" else "n/a"
    st.markdown('<div class="summary-card">', unsafe_allow_html=True)
    st.markdown("### Latest Run")
    st.markdown(
        f"""
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">Route</div>
                <div class="metric-value">{run.get("route", "unknown")}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">LLM Calls</div>
                <div class="metric-value">{run.get("llm_calls", 0)}</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Grounding</div>
                <div class="metric-value">{grounding}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if run.get("retrieval_query"):
        st.caption("Retrieval query")
        st.code(run["retrieval_query"], language="text")

    docs = run.get("retrieved_docs", [])
    if docs:
        st.caption("Top retrieved evidence")
        for doc in docs:
            render_source_card(doc)
    st.markdown("</div>", unsafe_allow_html=True)


inject_styles()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "run_history" not in st.session_state:
    st.session_state.run_history = []


st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">Dark Mode</div>
        <div class="hero-title">LangGraph RAG Studio</div>
        <p class="hero-copy">
            Ask questions about the indexed corpus, inspect retrieval behavior, and render math answers
            with proper LaTeX support in a cleaner dark interface.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Execution Trace")
    st.caption("Each turn shows how the graph routed the request and what evidence supported the answer.")

    if not st.session_state.run_history:
        st.info("No runs yet. Ask a question to inspect the workflow.")
    else:
        for idx, run in enumerate(reversed(st.session_state.run_history), start=1):
            turn_number = len(st.session_state.run_history) - idx + 1
            with st.expander(f"Turn {turn_number}", expanded=(idx == 1)):
                st.markdown(f"**Route:** `{run['route']}`")
                st.markdown(f"**LLM Calls:** `{run['llm_calls']}`")
                if run.get("retrieval_query"):
                    st.markdown("**Retrieval Query**")
                    st.code(run["retrieval_query"], language="text")
                if run.get("retrieval_grade"):
                    st.markdown(f"**Retrieval Grade:** `{run['retrieval_grade']}`")
                docs = run.get("retrieved_docs", [])
                if docs:
                    st.markdown("**Retrieved Sources**")
                    for doc in docs:
                        render_source_card(doc)


main_col, side_col = st.columns([1.8, 1], gap="large")

with main_col:
    for role, content in st.session_state.chat_history:
        with st.chat_message(role):
            render_content(content)

with side_col:
    if st.session_state.run_history:
        render_latest_run(st.session_state.run_history[-1])
    else:
        st.markdown(
            """
            <div class="summary-card">
                <h3 style="margin-top:0;">Ready To Explore</h3>
                <p style="color:#98a7bd;">
                    Try a retrieval question about your indexed documents, or ask for a derivative,
                    integral, or symbolic simplification to exercise the math path.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


prompt = st.chat_input("Ask about indexed documents, retrieval evidence, or a math problem...")

if prompt:
    st.session_state.chat_history.append(("user", prompt))
    with main_col:
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Running graph..."):
                initial_state: MessagesState = {
                    "messages": build_message_history(),
                    "llm_calls": 0,
                }
                result = agent.invoke(initial_state)

            final_answer = result.get("final_answer", "")
            render_content(final_answer)
            st.session_state.chat_history.append(("assistant", final_answer))

    st.session_state.run_history.append(
        {
            "route": result.get("route", "unknown"),
            "llm_calls": result.get("llm_calls", 0),
            "retrieval_query": result.get("retrieval_query", ""),
            "retrieval_grade": result.get("retrieval_grade", ""),
            "retrieved_docs": result.get("retrieved_docs", []),
        }
    )
    st.rerun()
