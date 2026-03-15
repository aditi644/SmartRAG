# 🔍 SmartRAG — Advanced Retrieval-Augmented Generation Pipeline

An advanced, production-quality RAG (Retrieval-Augmented Generation) pipeline built without LangChain, using native OpenAI APIs and ChromaDB. This project demonstrates intelligent document ingestion, LLM-powered chunking, vector storage, and sophisticated query answering with **reranking** and **query rewriting**.

---

## 📖 What This Project Does

SmartRAG ingests a knowledge base of Markdown documents (organized by type), processes them into semantically rich chunks using an LLM, stores embeddings in a persistent ChromaDB vector store, and answers natural language questions with high accuracy by combining:

- **LLM-driven chunking** — instead of splitting by character count, an LLM decides how to divide documents into meaningful chunks with headlines, summaries, and original text
- **OpenAI embeddings** — each chunk is embedded using `text-embedding-3-large` for dense vector retrieval
- **Reranking** — retrieved results are reordered by relevance using a second LLM call
- **Query rewriting** — user questions are refined before retrieval to maximize knowledge base hit rate
- **Conversational context** — multi-turn history is maintained for coherent follow-up answers

The demo knowledge base is built around a fictional insurance tech company called **Insurellm**, with documents covering company info, products, employee records, and contracts.

---

## 🗂️ Project Structure

```
project/
├── day5.ipynb             # Main Jupyter notebook with the full pipeline
├── knowledge-base/        # Source Markdown documents (organized by type)
│   ├── company/
│   ├── contracts/
│   ├── employees/
│   └── products/
├── preprocessed_db/       # ChromaDB persistent vector store (auto-generated)
└── requirements.txt       # Python dependencies
```

---

## 🚀 How It Works

### Phase 1 — Ingestion Pipeline

**Step 1: Load Documents**  
All `.md` files in the `knowledge-base/` directory are loaded recursively. Each document is tagged with its `type` (folder name) and `source` path.

**Step 2: LLM-Powered Chunking**  
For each document, a prompt is sent to `gpt-4.1-nano` asking it to split the document into overlapping, semantically coherent chunks. Each chunk contains:
- A `headline` — a short phrase likely to match user queries
- A `summary` — a few sentences answering common questions
- `original_text` — the verbatim source text

This is **document pre-processing**: the LLM rewrites chunks in the format most useful for retrieval. The result is stored as a `Result` object with `page_content` and `metadata`.

**Step 3: Embed & Store**  
All chunks are embedded in a single batch call to `text-embedding-3-large` and stored in a persistent ChromaDB collection (`preprocessed_db/docs`). The demo run produced **402 chunks** from 76 documents.

---

### Phase 2 — Advanced Query Pipeline

**Query Rewriting**  
Before retrieval, the user's question is rewritten into a concise knowledge-base query using an LLM. This improves recall for vague or conversational questions (e.g., `"Who won the IIOTY award?"` → `"Who received the IIOTY award?"`).

**Retrieval**  
The rewritten query is embedded and used to fetch the top-K most similar chunks from ChromaDB (default K=10–20).

**Reranking**  
Retrieved chunks are passed back to the LLM, which reorders them by relevance to the original question. This two-stage retrieval pipeline (dense retrieval + LLM reranking) dramatically improves answer quality.

**Answer Generation**  
The top reranked chunks, along with conversation history, are injected into a system prompt and the user's question is answered by the LLM.

---

## 🛠️ Setup & Installation

### Prerequisites

- Python 3.11+
- An OpenAI API key

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Set Up Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### Run the Notebook

```bash
jupyter notebook day5.ipynb
```

Or convert and run as a script:

```bash
jupyter nbconvert --to script day5.ipynb
python day5.py
```

---

## ⚙️ Configuration

Key configuration variables at the top of the notebook:

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `gpt-4.1-nano` | LLM for chunking, reranking, and answering |
| `embedding_model` | `text-embedding-3-large` | OpenAI embedding model |
| `DB_NAME` | `preprocessed_db` | ChromaDB persistence directory |
| `collection_name` | `docs` | ChromaDB collection name |
| `KNOWLEDGE_BASE_PATH` | `knowledge-base/` | Root directory for source documents |
| `AVERAGE_CHUNK_SIZE` | `500` | Target chunk size (characters) for LLM guidance |
| `RETRIEVAL_K` | `10–20` | Number of chunks to retrieve before reranking |

---

## 📊 Vector Store Visualization

The notebook includes both 2D and 3D t-SNE visualizations of the ChromaDB vector store using Plotly. Chunks are color-coded by document type:

- 🟠 **Orange** — Company documents
- 🔴 **Red** — Contract documents  
- 🟢 **Green** — Employee records
- 🔵 **Blue** — Product documents

---

## 🧩 Key Components

| Component | Description |
|---|---|
| `fetch_documents()` | Loads all `.md` files from the knowledge base |
| `make_prompt(document)` | Builds the LLM prompt for chunking |
| `process_document(document)` | Calls the LLM to chunk one document |
| `create_chunks(documents)` | Batches chunking across all documents |
| `create_embeddings(chunks)` | Embeds chunks and saves to ChromaDB |
| `rewrite_query(question, history)` | Rewrites the user question for better retrieval |
| `fetch_context_unranked(question)` | Retrieves top-K chunks from ChromaDB |
| `rerank(question, chunks)` | LLM-reranks retrieved chunks by relevance |
| `fetch_context(question)` | Full retrieval pipeline (rewrite → fetch → rerank) |
| `answer_question(question, history)` | End-to-end RAG answer generation |

---

## 💡 Advanced RAG Techniques Demonstrated

1. **No LangChain** — maximum flexibility using native APIs
2. **LLM-based semantic chunking** — smarter than character-based splitting
3. **Document pre-processing** — chunks are rewritten to be more queryable
4. **25% chunk overlap** — improves retrieval at document boundaries
5. **Reranking** — two-stage retrieval with LLM reranker
6. **Query rewriting** — refines queries for better knowledge base alignment
7. **Structured outputs** — uses Pydantic models with `response_format` for reliable parsing
8. **Multi-turn conversation** — history-aware RAG responses

---

## 📝 Example Usage

```python
# Single question
answer, chunks = answer_question("Who won the IIOTY award?")
print(answer)
# → "Maxine Thompson won the prestigious Insurellm IIOTY Innovator Award in 2023."

# Multi-turn conversation
history = []
answer, chunks = answer_question("Tell me about Carllm.", history)
history.append({"role": "user", "content": "Tell me about Carllm."})
history.append({"role": "assistant", "content": answer})

follow_up, chunks = answer_question("What pricing tiers does it offer?", history)
```

---

## 📦 Dependencies

See `requirements.txt` for the full list. Key packages:

- `openai` — embeddings and LLM inference
- `chromadb` — persistent vector store
- `litellm` — unified LLM completion interface
- `pydantic` — structured output models
- `python-dotenv` — environment variable management
- `scikit-learn` — t-SNE for visualization
- `plotly` — interactive 2D/3D plots
- `tqdm` — progress bars

---

## 🔒 Notes

- The ingestion step (chunking 76 documents) takes approximately **8–9 minutes** due to sequential LLM calls. For production use, consider parallelizing with `multiprocessing.Pool`.
- Re-running the ingestion will **delete and recreate** the ChromaDB collection.
- The `preprocessed_db/` directory should be committed to version control if you want to skip re-ingestion.

---

## 📄 License

This project is provided for educational purposes. Feel free to adapt it for your own knowledge base and use case.
