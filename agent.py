"""LangGraph agent with routed retrieval, grounded answering, and symbolic math tools."""

import operator
import os
from pathlib import Path
from typing import Annotated, Literal

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from sympy import Eq, diff, integrate, simplify, solve, sympify
from sympy.core.sympify import SympifyError
from typing_extensions import TypedDict

from embeddings import get_embeddings, load_provider_marker


load_dotenv()


class RetrievedDoc(TypedDict):
    """Serializable retrieval payload for the UI trace."""

    source: str
    section: str
    content: str
    score: float
    chunk_strategy: str


class MessagesState(TypedDict, total=False):
    """Graph state shared between routing, retrieval, and answer nodes."""

    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    route: Literal["math", "knowledge_base", "general"]
    retrieval_query: str
    retrieved_docs: list[RetrievedDoc]
    retrieval_grade: Literal["grounded", "insufficient"]
    final_answer: str


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


@tool
def divide(a: int, b: int) -> float:
    """Divides a by b."""
    return a / b


@tool
def solve_math(problem: str) -> str:
    """Solve symbolic math tasks such as simplify, solve equations, differentiate, and integrate."""
    text = problem.strip()
    lower = text.lower()

    try:
        if lower.startswith("simplify"):
            expr_text = text[len("simplify") :].strip(": ").strip()
            result = simplify(sympify(expr_text))
            return f"Simplified result: {result}"

        if lower.startswith("differentiate"):
            expr_text = text[len("differentiate") :].strip()
            if " with respect to " not in expr_text:
                return "Please specify the variable, for example: differentiate x**3 with respect to x"
            expr_part, variable = expr_text.split(" with respect to ", maxsplit=1)
            result = diff(sympify(expr_part.strip()), sympify(variable.strip()))
            return f"Derivative: {result}"

        if lower.startswith("integrate"):
            expr_text = text[len("integrate") :].strip()
            if " with respect to " not in expr_text:
                return "Please specify the variable, for example: integrate x**2 with respect to x"
            expr_part, variable = expr_text.split(" with respect to ", maxsplit=1)
            result = integrate(sympify(expr_part.strip()), sympify(variable.strip()))
            return f"Integral: {result}"

        if lower.startswith("solve"):
            payload = text[len("solve") :].strip()
            variable = None
            if " for " in payload:
                payload, variable = payload.rsplit(" for ", maxsplit=1)
                variable = variable.strip()

            if "=" in payload:
                left, right = payload.split("=", maxsplit=1)
                equation = Eq(sympify(left.strip()), sympify(right.strip()))
            else:
                equation = sympify(payload)

            result = solve(equation, sympify(variable)) if variable else solve(equation)
            return f"Solution: {result}"

        result = simplify(sympify(text))
        return f"Evaluated result: {result}"
    except (SympifyError, ValueError, TypeError) as exc:
        return f"Could not solve the problem symbolically: {exc}"


MATH_TOOLS = [add, multiply, divide, solve_math]
MATH_TOOLS_BY_NAME = {tool_.name: tool_ for tool_ in MATH_TOOLS}

MODEL_NAME = "openai/gpt-oss-20b"
DOC_TOP_K = 4

base_model = ChatGroq(model=MODEL_NAME, temperature=0)
math_model = base_model.bind_tools(MATH_TOOLS)


def _last_user_message(messages: list[AnyMessage]) -> str:
    """Return the latest user utterance from the graph message history."""
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
    return ""


def _load_vectorstore() -> FAISS:
    """Load the FAISS index with the same embedding backend used during ingestion."""
    index_path = os.path.join(os.path.dirname(__file__), "faiss_index")
    provider = load_provider_marker(Path(index_path))
    embeddings, _ = get_embeddings(provider)
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def _format_docs_for_prompt(docs: list[RetrievedDoc]) -> str:
    """Serialize retrieved documents into a compact prompt block."""
    if not docs:
        return "No supporting documents were retrieved."

    parts = []
    for idx, doc in enumerate(docs, start=1):
        parts.append(
            f"[{idx}] source={doc['source']} score={doc['score']:.4f}\n{doc['content']}"
        )
    return "\n\n".join(parts)


def route_question(state: MessagesState) -> dict:
    """Route a request into math, knowledge-base, or general conversation."""
    question = _last_user_message(state["messages"])
    router_prompt = (
        "Classify the user request into exactly one label: "
        "math, knowledge_base, or general.\n"
        "- math: arithmetic, algebra, symbolic manipulation, calculus, or equation solving.\n"
        "- knowledge_base: questions that should use the local document index.\n"
        "- general: normal conversation, writing help, or unsupported questions.\n"
        "Return only the label."
    )
    response = base_model.invoke(
        [
            SystemMessage(content=router_prompt),
            HumanMessage(content=question),
        ]
    )
    route = response.content.strip().lower()
    if route not in {"math", "knowledge_base", "general"}:
        route = "general"
    return {"route": route, "llm_calls": state.get("llm_calls", 0) + 1}


def route_after_classification(
    state: MessagesState,
) -> Literal["math_node", "build_retrieval_query", "general_node"]:
    route = state.get("route", "general")
    if route == "math":
        return "math_node"
    if route == "knowledge_base":
        return "build_retrieval_query"
    return "general_node"


def math_node(state: MessagesState) -> dict:
    """Run a bounded tool-using loop for symbolic or arithmetic math tasks."""
    conversation = [
        SystemMessage(
            content=(
                "You are a math assistant. Use tools for arithmetic and symbolic work.\n"
                "For anything beyond very simple mental math, prefer tool calls.\n"
                "Use `solve_math` for algebra, simplification, calculus, and equations.\n"
                "After using tools, explain the final result clearly.\n"
                "Do not use LaTeX. Write answers in plain readable text."
            )
        )
    ] + state["messages"]
    generated_messages: list[AnyMessage] = []
    llm_calls = state.get("llm_calls", 0)

    for _ in range(5):
        response = math_model.invoke(conversation)
        llm_calls += 1
        conversation.append(response)
        generated_messages.append(response)

        if not getattr(response, "tool_calls", None):
            return {
                "messages": generated_messages,
                "final_answer": response.content,
                "llm_calls": llm_calls,
            }

        for tool_call in response.tool_calls:
            tool_ = MATH_TOOLS_BY_NAME[tool_call["name"]]
            observation = tool_.invoke(tool_call["args"])
            tool_message = ToolMessage(
                content=str(observation),
                tool_call_id=tool_call["id"],
            )
            conversation.append(tool_message)
            generated_messages.append(tool_message)

    content = "I could not complete the math workflow within the tool-call limit."
    return {
        "messages": generated_messages + [AIMessage(content=content)],
        "final_answer": content,
        "llm_calls": llm_calls,
    }


def build_retrieval_query(state: MessagesState) -> dict:
    """Rewrite the user question into a search-friendly retrieval query."""
    question = _last_user_message(state["messages"])
    response = base_model.invoke(
        [
            SystemMessage(
                content=(
                    "Rewrite the user's question into a concise retrieval query for a local "
                    "document knowledge base. Keep important nouns, frameworks, entities, and task words. "
                    "Return only the query text."
                )
            ),
            HumanMessage(content=question),
        ]
    )
    return {
        "retrieval_query": response.content.strip(),
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def retrieve_docs(state: MessagesState) -> dict:
    """Retrieve the top supporting chunks from the local vector index."""
    vectorstore = _load_vectorstore()
    query = state.get("retrieval_query") or _last_user_message(state["messages"])
    matches: list[tuple[Document, float]] = vectorstore.similarity_search_with_score(
        query, k=DOC_TOP_K
    )
    retrieved_docs: list[RetrievedDoc] = [
        {
            "source": doc.metadata.get("source", "unknown"),
            "section": doc.metadata.get("section", "unknown"),
            "content": doc.page_content,
            "score": float(score),
            "chunk_strategy": doc.metadata.get("chunk_strategy", "unknown"),
        }
        for doc, score in matches
    ]
    return {"retrieved_docs": retrieved_docs}


def grade_retrieval(state: MessagesState) -> dict:
    """Decide whether the retrieved context is strong enough to answer safely."""
    question = _last_user_message(state["messages"])
    docs_text = _format_docs_for_prompt(state.get("retrieved_docs", []))
    response = base_model.invoke(
        [
            SystemMessage(
                content=(
                    "Decide whether the retrieved documents contain enough information "
                    "to answer the user's question faithfully.\n"
                    "Reply with exactly one word: grounded or insufficient."
                )
            ),
            HumanMessage(
                content=f"Question:\n{question}\n\nRetrieved context:\n{docs_text}"
            ),
        ]
    )
    grade = response.content.strip().lower()
    if grade not in {"grounded", "insufficient"}:
        grade = "insufficient"
    return {
        "retrieval_grade": grade,
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def route_after_grading(state: MessagesState) -> Literal["answer_with_docs", "fallback_node"]:
    if state.get("retrieval_grade") == "grounded":
        return "answer_with_docs"
    return "fallback_node"


def answer_with_docs(state: MessagesState) -> dict:
    """Generate a cited answer using only retrieved evidence."""
    question = _last_user_message(state["messages"])
    docs = state.get("retrieved_docs", [])
    docs_text = _format_docs_for_prompt(docs)
    response = base_model.invoke(
        [
            SystemMessage(
                content=(
                    "Answer the user's question using only the provided context.\n"
                    "Cite supporting sources inline using [source: filename].\n"
                    "If multiple sources support the answer, cite the strongest ones.\n"
                    "Do not invent facts outside the retrieved context.\n"
                    "Do not use LaTeX unless the user explicitly asks for it."
                )
            ),
            HumanMessage(content=f"Question:\n{question}\n\nContext:\n{docs_text}"),
        ]
    )
    return {
        "messages": [AIMessage(content=response.content)],
        "final_answer": response.content,
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def fallback_node(state: MessagesState) -> dict:
    """Return a transparent fallback when the knowledge base is not sufficient."""
    docs = state.get("retrieved_docs", [])
    if docs:
        sources = ", ".join(sorted({doc["source"] for doc in docs}))
        content = (
            "I could not find enough support in the local knowledge base to answer that "
            f"confidently. I checked: {sources}. Try asking a narrower AI/ML question or "
            "add more documents to the index."
        )
    else:
        content = (
            "I could not find any relevant material in the local knowledge base for that "
            "question. Try ingesting more documents or asking about the indexed topics."
        )
    return {
        "messages": [AIMessage(content=content)],
        "final_answer": content,
    }


def general_node(state: MessagesState) -> dict:
    """Handle non-retrieval, non-math requests with a lightweight assistant prompt."""
    response = base_model.invoke(
        [
            SystemMessage(
                content=(
                    "You are a helpful assistant. Answer clearly and briefly. "
                    "If the user is asking about project capabilities, mention that the "
                    "local knowledge base depends on the documents currently indexed in sample_docs. "
                    "Do not use LaTeX unless the user explicitly asks for it."
                )
            )
        ]
        + state["messages"]
    )
    return {
        "messages": [response],
        "final_answer": response.content,
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


graph = StateGraph(MessagesState)
graph.add_node("route_question", route_question)
graph.add_node("math_node", math_node)
graph.add_node("build_retrieval_query", build_retrieval_query)
graph.add_node("retrieve_docs", retrieve_docs)
graph.add_node("grade_retrieval", grade_retrieval)
graph.add_node("answer_with_docs", answer_with_docs)
graph.add_node("fallback_node", fallback_node)
graph.add_node("general_node", general_node)

graph.add_edge(START, "route_question")
graph.add_conditional_edges(
    "route_question",
    route_after_classification,
    ["math_node", "build_retrieval_query", "general_node"],
)
graph.add_edge("build_retrieval_query", "retrieve_docs")
graph.add_edge("retrieve_docs", "grade_retrieval")
graph.add_conditional_edges(
    "grade_retrieval",
    route_after_grading,
    ["answer_with_docs", "fallback_node"],
)
graph.add_edge("math_node", END)
graph.add_edge("answer_with_docs", END)
graph.add_edge("fallback_node", END)
graph.add_edge("general_node", END)

agent = graph.compile()
