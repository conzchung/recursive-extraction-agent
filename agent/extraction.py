import time, io, json, asyncio
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, Annotated
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END, START
from langchain_community.callbacks import get_openai_callback
import zipfile

from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy
from pdf2image import pdfinfo_from_bytes


from extraction_utils import (
    load_file_to_bytes_async,
    convert_text_from_excel,
    process_file_async,
    extract_content_with_ocr,
    extract_content_from_excel,
    extract_content_with_vlm,
    merge_dicts,
    extract_filename_from_url,
    split_excel_sheets_to_bytes,
    convert_buffer_to_base64_list,
    separate_documents_by_image,
    split_pdf,
    upload_blob_and_get_url,
    clean_markdown_tables,
    consolidate_extractions,
)

from models import GPT54m_args, GPT54_args
from clients import blob_service_client

from logging import getLogger

import warnings
warnings.filterwarnings('ignore')

# Set up logging
logger = getLogger(__name__)

container_name = 'aiagentdocs'


class ExtractionState(TypedDict):
    file_path: str
    file_type: str
    selected_fields: list[str]
    sensitivity: int
    selected_model: dict
    images: list[str]
    ocr_content: str 
    extraction_progress: dict
    status: dict
    iterations: int
    max_iter: int
    page_numbers: list[int]
    long_input: bool
    
    
async def load_and_preprocess_document(state: ExtractionState):
    """LangGraph node: download the file and prepare it for extraction.

    For Excel files: converts to markdown text via MarkItDown.
    For PDFs/images: converts pages to base64 images and runs OCR to
    produce a text layer.

    Returns:
        State update dict with ``file_type``, ``images`` (if PDF/image),
        and ``ocr_content``.
    """
    print("*** Preprocessing ***")
    logger.info("*** Preprocessing Started ***")

    file_path = state['file_path']
    file_data, file_type = await load_file_to_bytes_async(file_path)

    print(f'File type: {file_type}')
    logger.info(f"File type detected: {file_type}")

    if file_type == 'xlsx':
        ocr_content = convert_text_from_excel(file_data)
        cleaned_ocr_content = clean_markdown_tables(ocr_content)
        
        return {
            'file_type': file_type,
            'ocr_content': cleaned_ocr_content,
        }
    else:
        sensitivity = state['sensitivity']
        page_numbers = state.get('page_numbers')
        
        images = await process_file_async(
            file_data=file_data, 
            file_type=file_type, 
            kernel_size=sensitivity*100,
            page_numbers=page_numbers
            )
        ocr_content = extract_content_with_ocr(file_data, file_type, page_numbers)
        
        return {
            'file_type': file_type,
            'images': images,
            'ocr_content': ocr_content
        }
        

async def extract_and_transform_data(state: ExtractionState):
    """LangGraph node: run LLM extraction on the preprocessed content.

    Dispatches to ``extract_content_from_excel`` (Excel) or
    ``extract_content_with_vlm`` (PDF/image).  Merges new results into
    any existing ``extraction_progress`` from prior iterations.

    Returns:
        State update dict with updated ``extraction_progress`` and
        incremented ``iterations`` counter.
    """
    print("*** Extracting ***")
    logger.info("*** Extracting Data ***")

    file_type = state['file_type']
    selected_fields = state['selected_fields']
    ocr_content = state['ocr_content']
    iterations = state['iterations']
    selected_model = state['selected_model']
    
    long_input = state['long_input']
    page_numbers = state.get('page_numbers')

    iterations += 1
    print(f"Iterations: {iterations}")
    logger.info(f"Iteration count: {iterations}")

    current_progress = state.get('extraction_progress')

    start_time = time.time()

    if file_type == 'xlsx':
        extracted_dict = await extract_content_from_excel(
            ocr_content, 
            selected_fields, 
            current_progress,
            selected_model
            )
    else:
        images = state['images']
        if not page_numbers:
            page_numbers = [i + 1 for i in range(len(images))]
            
        extracted_dict = await extract_content_with_vlm(
            images,
            selected_fields,
            page_numbers,
            ocr_content,
            current_progress,
            long_input,
            selected_model
        )

    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(f"Extraction on {file_type} @ elapsed time: {elapsed_time:.2f} seconds")
    logger.info(f"Extraction on {file_type} completed in {elapsed_time:.2f} seconds")
    # print(f"Extraction Status: {extracted_dict.get('note', 'None')}")
    logger.info(f"Extraction Status: {extracted_dict.get('note', 'None')}")

    if not current_progress:
        new_progress = extracted_dict
    else:
        new_progress = merge_dicts(current_progress, extracted_dict)

    return {
        'extraction_progress': new_progress,
        'iterations': iterations
    }
    

def refine_extraction_schema(state: ExtractionState):
    """LangGraph node: unwrap and normalise the final extraction result.

    Strips the ``extraction_progress`` / ``multiple_extractions``
    wrapper so the caller receives a clean dict of extracted fields.

    Returns:
        State update dict with the flat ``extraction_progress`` and the
        ``status`` note from the LLM.
    """
    print("*** Refining ***")
    logger.info("*** Refining Extraction ***")

    current_progress = state["extraction_progress"]
    status = current_progress.get('note')
    
    refined_progress = current_progress["extraction_progress"]

    if 'multiple_extractions' in refined_progress.keys():
        target = refined_progress.get('multiple_extractions')
    else:
        target = refined_progress

    print("*** Completed ***")
    logger.info("*** Extraction Process Completed ***")

    return {
        'extraction_progress': target,
        'status': status
    }
    
    
def reflect_extraction_progress(state: ExtractionState):
    """LangGraph conditional edge: decide whether to loop or finish.

    Returns ``"refine"`` if all fields are marked complete or the max
    iteration count is reached; otherwise returns ``"extract"`` to
    continue the extraction loop.
    """
    print("*** Reflecting ***")
    logger.info("*** Reflecting on Extraction Progress ***")
    
    iterations = state["iterations"]
    max_iter = state["max_iter"]
    current_progress = state["extraction_progress"]

    note = current_progress.get('note', {})
    progress = note.get('progress', 'incomplete')

    if isinstance(progress, dict):
        all_completed = all(v == 'completed' for v in progress.values())
    elif isinstance(progress, str):
        all_completed = progress == 'completed'
    else:
        all_completed = False

    if all_completed or iterations >= max_iter:
        print("Refine")
        logger.info("Proceeding to refine extraction")
        return "refine"
    else:
        print("Extract")
        logger.info("Continuing extraction process")
        return "extract"
    
    
def recursive_document_extractor():
    """Build and compile the iterative extraction LangGraph.

    Graph topology::

        START → preprocess → extract ←→ reflect
                                           ↓
                                        refine → END

    The extract/reflect loop repeats until all fields are complete or
    ``max_iter`` is reached.  Each node is cached in-memory so
    identical inputs short-circuit.

    Returns:
        A compiled ``StateGraph`` ready for ``ainvoke``.
    """
    workflow = StateGraph(ExtractionState)
    
    # Define the nodes
    workflow.add_node("preprocess", load_and_preprocess_document, cache_policy=CachePolicy())
    workflow.add_node("extract", extract_and_transform_data, cache_policy=CachePolicy())
    workflow.add_node("refine", refine_extraction_schema, cache_policy=CachePolicy())
    
    # Build graph
    workflow.add_edge(START, "preprocess")
    workflow.add_edge("preprocess", "extract")
    workflow.add_conditional_edges(
        "extract",
        reflect_extraction_progress,
        {
            "extract": "extract",
            "refine": "refine"
        },
    )
    workflow.add_edge("refine", END)
    
    graph = workflow.compile(cache=InMemoryCache())
    return graph


async def extraction_workflow(
    file_path: str,
    selected_fields: list[str],
    sensitivity: int = 5,
    max_iter: int = 20,
    batch_size: int = 40,
    overlap: int = 5,
    selected_model: dict = GPT54_args,
    *,
    recursion_limit: int = 100,
    long_doc_page_threshold: int = 50,
) -> tuple[dict, dict]:
    """Top-level entry point: extract structured data from a document.

    Handles three document scenarios:
    1. **Excel** — single-pass extraction.
    2. **Short PDF/image** (≤ ``long_doc_page_threshold`` pages) —
       single-pass extraction.
    3. **Long PDF** (> threshold) — splits into overlapping page batches,
       extracts each independently, then consolidates results via LLM.

    Args:
        file_path: URL or local path to the source document.
        selected_fields: Field descriptions the LLM should extract.
        sensitivity: Image preprocessing kernel size multiplier (1–5).
        max_iter: Max extract→reflect loop iterations per batch.
        batch_size: Pages per batch for long documents.
        overlap: Overlapping pages between consecutive batches.
        selected_model: LLM config dict (e.g. ``GPT54_args``).
        recursion_limit: LangGraph recursion safety limit.
        long_doc_page_threshold: Page count that triggers batching.

    Returns:
        A tuple of ``(extraction_result, token_usage)`` where
        *extraction_result* is the extracted fields dict and
        *token_usage* contains prompt/completion/total token counts.

    Raises:
        ValueError: If batch_size/overlap constraints are violated.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= batch_size:
        raise ValueError("overlap must be < batch_size")

    extraction_graph = recursive_document_extractor()

    base_state: Dict[str, Any] = {
        "file_path": file_path,
        "selected_fields": selected_fields,
        "sensitivity": sensitivity,
        "extraction_progress": None,
        "iterations": 0,
        "max_iter": max_iter,
        "selected_model": selected_model,
    }

    config = {"recursion_limit": recursion_limit}

    full_token_usage = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }

    def _add_usage(usage: dict) -> None:
        for k in full_token_usage:
            full_token_usage[k] += usage.get(k, 0)

    def _extract_progress(extraction_states: dict) -> dict:
        return {
            "extraction_progress": extraction_states.get("extraction_progress"),
            "note": extraction_states.get("status", {}),
        }

    async def _run_graph(state: dict) -> Tuple[dict, dict]:
        with get_openai_callback() as cb:
            extraction_states = await extraction_graph.ainvoke(state, config)

        usage = {
            "total_tokens": cb.total_tokens,
            "prompt_tokens": cb.prompt_tokens,
            "completion_tokens": cb.completion_tokens,
        }
        return _extract_progress(extraction_states), usage

    def _iter_page_batches(num_pages: int) -> List[List[int]]:
        pages = list(range(1, num_pages + 1))
        step = batch_size - overlap
        return [pages[i : i + batch_size] for i in range(0, len(pages), step)]

    async def _get_num_pages(file_data, file_type: str) -> int:
        ft = (file_type or "").strip().lower()

        if ft == "pdf":
            pdf_bytes = file_data.getvalue()
            info = await asyncio.to_thread(pdfinfo_from_bytes, pdf_bytes)
            return int(info.get("Pages", 0))

        return 1

    file_data, file_type = await load_file_to_bytes_async(file_path)
    print(f"File type: {file_type}")

    if (file_type or "").strip().lower() == "xlsx":
        state = {**base_state, "page_numbers": None, "long_input": False}
        result, usage = await _run_graph(state)
        _add_usage(usage)

        final_result = result.get("extraction_progress") or {}
        return final_result, full_token_usage

    num_pages = await _get_num_pages(file_data, file_type)
    if num_pages <= 0:
        return {}, full_token_usage

    if num_pages > long_doc_page_threshold:
        print(f" // Document pages > {long_doc_page_threshold}, running long input context pipeline")

        page_batches = _iter_page_batches(num_pages)
        batch_extractions: List[str] = []

        for idx, page_numbers in enumerate(page_batches, start=1):
            print(f" // Batch {idx}/{len(page_batches)} pages: {page_numbers[0]}-{page_numbers[-1]}")

            state = {**base_state, "page_numbers": page_numbers, "long_input": True}
            batch_result, usage = await _run_graph(state)
            _add_usage(usage)

            batch_extractions.append(json.dumps(batch_result, ensure_ascii=False))

        consolidated_result, usage = await consolidate_extractions(
            batch_extractions=batch_extractions,
            selected_fields=selected_fields,
        )
        _add_usage(usage)

        if isinstance(consolidated_result, dict):
            final_result = consolidated_result.get("result", consolidated_result) or {}
        elif isinstance(consolidated_result, list):
            final_result = consolidated_result[0] if len(consolidated_result) == 1 else {"items": consolidated_result}
        else:
            final_result = {}

        return final_result, full_token_usage

    print(f" // Document pages <= {long_doc_page_threshold}, running standard input context pipeline")
    state = {**base_state, "page_numbers": None, "long_input": False}
    result, usage = await _run_graph(state)
    _add_usage(usage)

    final_result = result.get("extraction_progress") or {}
    return final_result, full_token_usage


async def partition_document(
    file_path: str, 
    container_name: str = "aiagentdocs", 
    blob_service_client = blob_service_client,
    selected_model: Literal["basic", "advanced"] = "advanced"
    ) -> str:
    """
    Partition a document into multiple files based on file type, bundle them into a ZIP file,
    and upload the ZIP to Azure Blob Storage.

    Args:
        file_path (str): The path or URL of the file to partition.
        container_name (str): The name of the Azure Blob Storage container.
        blob_service_client: The Azure Blob Service Client for uploading files.

    Returns:
        str: The blob path (URL) of the uploaded ZIP file containing all split files.
    """
    try:
        # Load file data (synchronous, can be offloaded to executor if needed)
        file_data, file_type = await load_file_to_bytes_async(file_path)
        print(f'File type: {file_type}')
        logger.info(f"File type detected: {file_type}")
        base_name = extract_filename_from_url(file_path)

        # Create a BytesIO buffer to store the ZIP file
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
            if file_type == 'xlsx':
                sheet_bytes_dict = split_excel_sheets_to_bytes(file_data)
                
                for key, value in sheet_bytes_dict.items():
                    cleaned_key = key.replace(" ", "_")
                    zip_entry_name = f"{base_name}_{cleaned_key}.xlsx"
                    # Extract bytes from BytesIO object
                    value_bytes = value.getvalue()
                    zipf.writestr(zip_entry_name, value_bytes)
                    
            else:  # Assuming other file types are PDFs or similar
                image_datas = convert_buffer_to_base64_list(file_data, file_type)
                
                if selected_model == "basic":
                    selected_llm_args = GPT54m_args
                else:
                    selected_llm_args = GPT54_args
                    
                # Await the async function for separating documents
                idx_response = await separate_documents_by_image(image_datas, selected_llm_args=selected_llm_args)
                page_list = [item['page_indices'] for item in idx_response['document_index']]
                split_pdfs = split_pdf(file_data, page_list)
                
                for i, pdf in enumerate(split_pdfs, start=1):
                    zip_entry_name = f"{base_name}_partition_{i}.pdf"
                    # Extract bytes from BytesIO object
                    pdf_bytes = pdf.getvalue()
                    zipf.writestr(zip_entry_name, pdf_bytes)

        # Prepare the ZIP file for upload
        zip_buffer.seek(0)  # Reset buffer position to the beginning
        zip_blob_name = f'ai_partition_documents/{base_name}_partitions.zip'
        
        # Use the async upload function
        zip_blob_path = await upload_blob_and_get_url(
            container_name=container_name,
            blob_name=zip_blob_name,
            data=zip_buffer.getvalue(),
            blob_service_client=blob_service_client
        )
        logger.info(f"Uploaded ZIP file to blob path: {zip_blob_path}")

        return zip_blob_path

    except Exception as e:
        logger.error(f"Error in partition_document: {str(e)}")
        raise ValueError(f"Failed to partition document: {str(e)}")