# AI Extraction & Validation

A document intelligence platform that extracts structured data from unstructured documents and validates extraction results against configurable rule sets. Powered by Azure OpenAI LLMs via LangChain/LangGraph.

## Extraction

Converts PDFs, scanned images, and Excel spreadsheets into structured JSON output with configurable field schemas.

### Features

- **Long document handling** — PDFs over 100 pages are automatically split into overlapping page batches (40 pages per batch, 5-page overlap), extracted independently, then consolidated into a single coherent result via LLM
- **Complex table extraction** — Large tables with hundreds of rows are extracted iteratively in 50-row chunks with precise resume points, ensuring no rows are missed or duplicated
- **Dual extraction pipeline** — PDF/image documents are processed through both a Vision Language Model (VLM) for layout-aware extraction and pdfplumber OCR for character accuracy, cross-referencing both sources for reliable results
- **Structured output** — Configurable field schemas with field name, data type (text, date, table, etc.), and optional remarks for business rules, formatting instructions, or column definitions
- **Logic operations & aggregation** — Field remarks support business rules that instruct the LLM to perform calculations, comparisons, unit conversions, and aggregation during extraction
- **Iterative refinement** — A LangGraph state machine runs an extract → reflect → re-extract loop, continuing until all fields are marked complete or the maximum iteration count is reached
- **Image preprocessing** — Automatic page orientation detection and correction via LLM, whitespace cropping via OpenCV morphological operations, contrast enhancement (CLAHE), and output size enforcement
- **Multi-document partitioning** — Bundled PDFs containing multiple logical documents are split into separate files by visual similarity analysis
- **Excel support** — Spreadsheets are converted to Markdown via MarkItDown, then extracted with the same iterative chunked approach, supporting multi-sheet workbooks

### How It Works

```
Document URL → Download → Detect Type
  ├─ PDF/Image → Render pages → OpenCV preprocessing → Orientation correction
  │              → VLM extraction + OCR cross-reference → Structured JSON
  └─ Excel     → MarkItDown conversion → LLM extraction → Structured JSON
```

For long documents (>50 pages), the pipeline adds a batching layer:
```
Long PDF → Split into overlapping batches → Extract each batch → LLM consolidation → Final result
```

## Validation

Compares extraction results against natural-language rules for compliance checking.

### Capabilities

- **LLM-based compliance review** — Multiple extraction results are evaluated against natural-language rules that can specify comparisons, calculations, anomaly checks, completeness checks, and more
- **Scoring** — Each rule produces a score from 0 (completely inconsistent) to 10 (all values match exactly), along with a concise reason
- **Evidence generation** — Every score is backed by structured Markdown evidence tables showing the source values, violations or matches, and notes on semantic mappings used
- **Concurrent evaluation** — Up to 20 rules are evaluated in parallel for fast turnaround
- **Semantic field matching** — Automatically maps equivalent field names across sources (e.g., "Company Name" ↔ "account_name" ↔ "beneficiary_name") with string normalization
- **Cross-document comparison** — Validates consistency across multiple extraction results from different source documents

## Architecture

| Component | Stack |
|-----------|-------|
| **Backend API** | FastAPI (Python 3.12), async endpoints with background task processing |
| **AI Engine** | Azure OpenAI (GPT-5.4 / GPT-5.4 Mini) via LangChain + LangGraph |
| **Frontend** | Streamlit with cookie-based authentication |
| **Database** | Azure Cosmos DB (async SDK) |
| **Storage** | Azure Blob Storage for document upload and processed files |
| **Image Processing** | OpenCV, PyMuPDF, pdf2image, pdfplumber, Pillow |

### API Design

Extraction and validation are long-running operations. The API uses a background task pattern:
1. Client submits a job → receives a job ID immediately with "queued" status
2. A background task processes the job (queued → processing → succeeded/failed)
3. Client polls for results using the job ID

## Quick Start

### Backend (FastAPI)
```bash
pip install -r requirements-fastapi.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend (Streamlit)
```bash
pip install -r requirements-streamlit.txt
cd streamlit
streamlit run login.py --server.port 8501
```

### Configuration

Create a `.env` file at the project root with:
```
AZURE_OPENAI_API_KEY=<Azure OpenAI API key>
AZURE_OPENAI_ENDPOINT=<Azure OpenAI endpoint URL>
AZURE_DEPLOYMENT_GPT54M=<Azure deployment name for GPT-5.4 Mini>
AZURE_DEPLOYMENT_GPT54=<Azure deployment name for GPT-5.4>
AZURE_OPENAI_API_VERSION=<Azure OpenAI API version, e.g. 2025-04-01-preview>
AZURE_BLOB_CONNECTION_STRING=<Azure Blob Storage connection string>
COSMOS_URL=<Azure Cosmos DB URL>
COSMOS_KEY=<Azure Cosmos DB key>
API_KEY=<API key for X-API-Key header authentication>
```
