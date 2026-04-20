# DocuMind

## What It Does

- Routes user queries into one of three paths: `math`, `knowledge_base`, or `general`.
- Uses a local FAISS vector index for mixed-format knowledge-base retrieval.
- Rewrites user questions into retrieval-optimized search queries.
- Grades retrieved context before answering, so the system can refuse weakly grounded answers.
- Generates cited answers with inline source references.
- Uses true recursive chunking for prose documents and table-aware batching for structured files.
- Supports symbolic math tool calling for algebra, simplification, differentiation, and integration.
- Falls back to local embeddings when the remote embedding provider is unavailable or rate-limited.
- Exposes graph decisions, retrieval scores, and grounding status in a more professional Streamlit console.

## Architecture

The LangGraph workflow is:

`route_question -> build_retrieval_query -> retrieve_docs -> grade_retrieval -> answer_with_docs`

Alternative routes:

- `route_question -> math_node`
- `route_question -> general_node`
- `grade_retrieval -> fallback_node`

This makes the project stronger than a basic chatbot because the graph has explicit decision points, observable state transitions, and a grounding check before answer generation.

## Tech Stack

- `LangGraph` for workflow orchestration
- `LangChain` for model and tool interfaces
- `Groq` for chat inference
- `Google Generative AI Embeddings` with a local hash fallback for vector embeddings
- `FAISS` for local similarity search
- `PyMuPDF` for PDF ingestion
- `Pandas` for spreadsheet and tabular ingestion
- `BeautifulSoup` for HTML parsing
- `Streamlit` for the debugging and demo UI
- `SymPy` for symbolic math solving

## Recommended Knowledge Base

If you want a large, high-signal corpus for this project, choose a focused AI engineering knowledge base instead of a generic dump.

Best choice:

- `LangChain + LangGraph + Hugging Face + Transformers + RAG architecture notes + selected arXiv paper summaries`

Why this is a strong resume choice:

- It is large enough to make retrieval quality and chunking decisions matter.
- It stays aligned with the agent/RAG theme of your project.
- It lets you talk about grounded QA over real technical documentation, not toy paragraphs.
- It creates clear opportunities for section-aware chunking because docs, guides, API references, and papers have different structures.

## Chunking Strategy

The ingestion pipeline now applies format-aware chunking:

- Markdown files use section-aware chunking based on headers before recursive splitting.
- PDF files use `PyMuPDF` extraction page by page, then recursive chunking over extracted prose.
- Text files use recursive chunking with paragraph, line, and sentence-level fallback.
- Excel and delimited files are normalized into sheet-aware or dataset-aware text blocks before chunking.
- HTML files extract both visible page text and table content so retrieval can answer from prose and structured data.
- Chunk size and overlap are adjusted dynamically based on document length and paragraph density.
- Each chunk stores metadata like source, section, chunk strategy, and chunk parameters for better debugging and retrieval inspection.

## Project Structure

- [agent.py](/Users/aakashnath/Desktop/LangGraph%20RAG/agent.py) contains the graph and node logic
- [app.py](/Users/aakashnath/Desktop/LangGraph%20RAG/app.py) contains the Streamlit UI
- [ingest.py](/Users/aakashnath/Desktop/LangGraph%20RAG/ingest.py) builds the FAISS index from local `.txt`, `.md`, `.pdf`, spreadsheet, delimited, and HTML files
- [sample_docs](/Users/aakashnath/Desktop/LangGraph%20RAG/sample_docs) stores the source documents

## Setup

Create a `.env` file with:

```bash
GROQ_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Build the vector index:

```bash
python ingest.py
```

Run the Streamlit app:

```bash
streamlit run app.py
```
- Add unit tests for each graph branch and failure mode.
