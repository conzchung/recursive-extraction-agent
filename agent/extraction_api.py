import uuid
import pytz
import logging
from fastapi import (
    BackgroundTasks, 
    HTTPException, 
    APIRouter, 
    File, 
    UploadFile, 
    Depends, 
    Path, 
    Body, 
    Query
)
from pydantic import BaseModel, Field
from typing import Annotated, Any, List, Optional, Dict
from enum import Enum
import time
from datetime import datetime
import traceback

from extraction import extraction_workflow, partition_document
from extraction_utils import upload_blob_and_get_url
from models import GPT54m_args, GPT54_args
from clients import (
    extraction_container,
    config_container,
    blob_service_client as _shared_blob_client,
    get_api_key,
)

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

extraction_router = APIRouter()

hk_timezone = pytz.timezone("Asia/Hong_Kong")


@extraction_router.post("/upload_document", tags=["Extraction"])
async def upload_document(
    file: UploadFile = File(...),
    api_key: str = Depends(get_api_key)
) -> Dict[str, Any]:
    """
    Endpoint to upload a document to Azure Blob Storage and return a public blob URL.
    
    Args:
        file (UploadFile): The file to upload.
        api_key (str): API key for authentication (injected via Depends).
    
    Returns:
        Dict[str, Any]: The blob URL of the uploaded file.
    
    Raises:
        HTTPException: If an error occurs during upload (400 or 500).
    """
    try:
        # Validate file and filename
        if not file:
            raise HTTPException(status_code=400, detail="No file provided for upload.")
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

        # Define container and blob name
        container_name = 'aiagentdocs'
        # Safely parse filename and extension
        filename_parts = file.filename.rsplit('.', 1) if '.' in file.filename else [file.filename, 'unknown']
        file_name = filename_parts[0]
        file_type = filename_parts[1] if len(filename_parts) > 1 else 'unknown'
        blob_name = f"ai_extraction_documents/{file_name}.{file_type}"

        # Read file content (async operation)
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Upload and get blob URL (async operation)
        blob_url = await upload_blob_and_get_url(
            container_name=container_name,
            blob_name=blob_name,
            data=file_content,
            blob_service_client=_shared_blob_client,
            expiry=7  # 7 days expiry for SAS token
        )

        logger.info(f"Uploaded file {file.filename} to Azure Blob Storage as {blob_name}")
        return {
            "status": "success",
            "data": {
                "blob_url": blob_url,
                "filename": file.filename
            }
        }
    
    except Exception as e:
        logger.error(f"Error uploading file {file.filename if file else 'unknown'}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Define the Enum for model types
class PartitionModelType(str, Enum):
    BASIC = "basic"
    ADVANCED = "advanced"
    
class PartitionRequest(BaseModel):
    file_path: str
    model: PartitionModelType = Field(default=PartitionModelType.ADVANCED)

@extraction_router.post("/partition_document", tags=["Extraction"])
async def partition_document_endpoint(
    request: PartitionRequest = Body(...),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """
    Endpoint to partition a document into multiple files based on file type, bundle them into a ZIP file,
    and upload the ZIP to Azure Blob Storage.

    Args:
        request (PartitionRequest): Request body containing the file path.
        api_key (str): API key for authentication (injected via Depends).

    Returns:
        Dict[str, Any]: The blob URL of the uploaded ZIP file containing the partitioned files.

    Raises:
        HTTPException: If an error occurs during partitioning or upload (400 or 500).
    """
    try:
        # Extract file_path from request body
        file_path = request.file_path

        # Validate file_path
        if not file_path:
            raise HTTPException(status_code=400, detail="No file path provided for partitioning.")

        # Call the async partition_document function
        zip_blob_path = await partition_document(
            file_path=file_path,
            selected_model=request.model.value  # Pass the model value (e.g., "GPT5m", "GPT5", or "GPT51")
        )
        
        logger.info(f"Partitioned document from {file_path} and uploaded ZIP to {zip_blob_path}")
        return {
            "status": "success",
            "data": {
                "zip_blob_url": zip_blob_path
            }
        }
    
    except ValueError as e:
        logger.error(f"ValueError during partitioning of {file_path}: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error partitioning document from {file_path}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def convert_selected_fields(selected_fields):
    """Format configuration field dicts into prompt-ready strings.

    Each field dict ``{columnName, dataType, remarks}`` is converted to a
    bracketed string like ``{{ Field Name: X, Data Type: Y, Remarks: Z }}``
    suitable for embedding in an LLM extraction prompt.

    Args:
        selected_fields: List of field dicts from the Configurations container.

    Returns:
        List of formatted field description strings.
    """
    temp_list = []
    for item in selected_fields:
        text = "{{ "
        text += f"Field Name: {item['columnName']}, Data Type: {item['dataType']}"
        if item['remarks']:
            text += f", Remarks: {item['remarks']}"
        text += " }}"
        temp_list.append(text)

    return temp_list


# =========================
# Extraction Service Endpoints
# =========================
class ExtractionModelType(str, Enum):
    BASIC = "basic"
    ADVANCED = "advanced"

class ExtractionRequest(BaseModel):
    extraction_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    file_path: str
    configuration_id: str
    sensitivity: int = Field(default=5, ge=1, le=5)
    max_iter: int = Field(default=20, ge=1, le=50)
    model: ExtractionModelType = Field(default=ExtractionModelType.ADVANCED)
    user_id: str  # NEW field to store ownership of the extraction

@extraction_router.post("/extract_document", tags=["Extraction"])
async def extract_document(
    extraction_request: "ExtractionRequest",
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """
    Endpoint to extract data from a document using the extraction workflow and track status in Cosmos DB.
    The extraction process runs as a background task to avoid timeout issues.
    """

    try:
        hk_time = datetime.now(hk_timezone)
        formatted_time = hk_time.strftime("%Y-%m-%d %H:%M:%S")
        iso_formatted_time = hk_time.isoformat(timespec="seconds")

        config_id = extraction_request.configuration_id

        # Async, parameterized query for configuration
        query = "SELECT * FROM c WHERE c.configId = @config_id"
        items_paged = config_container.query_items(
            query=query,
            parameters=[{"name": "@config_id", "value": config_id}],
            # enable_cross_partition_query=True,
        )
        config_items = [item async for item in items_paged]

        if not config_items:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration with ID {config_id} not found",
            )

        config = config_items[0]
        selected_fields = config.get("fieldsToExtract", [])
        if not selected_fields:
            raise HTTPException(
                status_code=400,
                detail=f"No fields to extract found in configuration {config_id}",
            )

        logger.info(f"Retrieved configuration for configuration_id: {config_id}")

        fields_to_extract = convert_selected_fields(selected_fields)

        # Initial queued record in Cosmos DB (async upsert)
        new_item = {
            "id": str(extraction_request.extraction_id),
            "extractionId": str(extraction_request.extraction_id),
            "status": "queued",
            "filePath": extraction_request.file_path,
            "configId": str(extraction_request.configuration_id),
            "fieldsToExtract": selected_fields,
            "model": extraction_request.model.value,
            "userId": extraction_request.user_id,
            "extractionResult": None,
            "modifiedAtHKTime": formatted_time,
            "modifiedAtIsoTime": iso_formatted_time,
        }
        await extraction_container.upsert_item(new_item)
        logger.info(
            f"Initial record created in Cosmos DB for extraction ID: "
            f"{extraction_request.extraction_id}"
        )

        # --------------------------------------------
        # Background task part – see notes below
        # --------------------------------------------
        # BackgroundTasks supports both sync and async callables.
        # Avoid blocking operations inside async tasks; move CPU-bound or blocking I/O to a worker.
        background_tasks.add_task(
            run_extraction_task,
            extraction_request,
            selected_fields,
            fields_to_extract,
            hk_timezone,
        )

        return {
            "status": "queued",
            "extraction_id": extraction_request.extraction_id,
            "message": "Extraction task has been queued. Use the extraction_id to check status.",
        }

    except HTTPException:
        # Propagate HTTP errors as-is
        raise
    except Exception as e:
        logger.error(f"Error queuing document extraction: {e}")
        traceback.print_exc()

        # Best-effort error record write – don't mask the original failure
        try:
            hk_time = datetime.now(hk_timezone)
            formatted_time = hk_time.strftime("%Y-%m-%d %H:%M:%S")
            iso_formatted_time = hk_time.isoformat(timespec="seconds")

            error_item = {
                "id": str(extraction_request.extraction_id),
                "extractionId": str(extraction_request.extraction_id),
                "status": "failed",
                "filePath": extraction_request.file_path,
                "configId": str(extraction_request.configuration_id),
                "fieldsToExtract": selected_fields if "selected_fields" in locals() else [],
                "model": extraction_request.model.value,
                "userId": extraction_request.user_id,
                "extractionResult": None,
                "error": str(e),
                "tokenUsage": 0,
                "modifiedAtHKTime": formatted_time,
                "modifiedAtIsoTime": iso_formatted_time,
                "timeLapsed": 0,
            }
            await extraction_container.upsert_item(error_item)
            logger.info(
                "Record updated in Cosmos DB with failure for extraction ID: "
                f"{extraction_request.extraction_id}"
            )
        except Exception as log_err:
            logger.error(f"Failed to write error record to Cosmos: {log_err}")

        raise HTTPException(status_code=500, detail="Internal server error")


async def run_extraction_task(
    extraction_request: ExtractionRequest,
    selected_fields: list,
    fields_to_extract: list,
    hk_timezone,
):
    """Background task that runs the full extraction workflow.

    Lifecycle: updates Cosmos status from ``processing`` → ``succeeded``
    or ``failed``.  Any unhandled exception is caught, logged, and
    persisted as a failed record so the client never sees a stuck job.

    Args:
        extraction_request: The original API request payload.
        selected_fields: Raw field dicts from the configuration.
        fields_to_extract: Prompt-formatted field strings.
        hk_timezone: ``pytz`` timezone for timestamp formatting.
    """
    try:
        hk_time = datetime.now(hk_timezone)
        formatted_time = hk_time.strftime("%Y-%m-%d %H:%M:%S")
        iso_formatted_time = hk_time.isoformat(timespec="seconds")

        processing_item = {
            "id": str(extraction_request.extraction_id),
            "extractionId": str(extraction_request.extraction_id),
            "status": "processing",
            "filePath": extraction_request.file_path,
            "configId": str(extraction_request.configuration_id),
            "fieldsToExtract": selected_fields,
            "model": extraction_request.model.value,
            "userId": extraction_request.user_id,
            "extractionResult": None,
            "modifiedAtHKTime": formatted_time,
            "modifiedAtIsoTime": iso_formatted_time,
        }
        await extraction_container.upsert_item(processing_item)
        logger.info(
            f"Status updated to processing for extraction ID: "
            f"{extraction_request.extraction_id}"
        )

        start_time = time.time()

        model_args = GPT54m_args if extraction_request.model == ExtractionModelType.BASIC else GPT54_args

        extraction_result, token_usage = await extraction_workflow(
            extraction_request.file_path,
            fields_to_extract,
            extraction_request.sensitivity,
            extraction_request.max_iter,
            selected_model=model_args,
        )

        hk_time = datetime.now(hk_timezone)
        formatted_time = hk_time.strftime("%Y-%m-%d %H:%M:%S")
        iso_formatted_time = hk_time.isoformat(timespec="seconds")
        time_elapsed = time.time() - start_time

        logger.info(
            "Extraction workflow succeeded for ID "
            f"{extraction_request.extraction_id}"
        )

        updated_item = {
            "id": str(extraction_request.extraction_id),
            "extractionId": str(extraction_request.extraction_id),
            "status": "succeeded",
            "filePath": extraction_request.file_path,
            "configId": str(extraction_request.configuration_id),
            "fieldsToExtract": selected_fields,
            "model": extraction_request.model.value,
            "userId": extraction_request.user_id,
            "extractionResult": extraction_result,
            "error": None,
            "tokenUsage": token_usage,
            "modifiedAtHKTime": formatted_time,
            "modifiedAtIsoTime": iso_formatted_time,
            "timeLapsed": time_elapsed,
        }
        await extraction_container.upsert_item(updated_item)
        logger.info(
            "Record updated in Cosmos DB with success for extraction ID: "
            f"{extraction_request.extraction_id}"
        )

    except Exception as e:
        logger.error(f"Error during background extraction: {e}")
        traceback.print_exc()

        hk_time = datetime.now(hk_timezone)
        formatted_time = hk_time.strftime("%Y-%m-%d %H:%M:%S")
        iso_formatted_time = hk_time.isoformat(timespec="seconds")
        time_elapsed = time.time() - start_time if "start_time" in locals() else 0

        error_item = {
            "id": str(extraction_request.extraction_id),
            "extractionId": str(extraction_request.extraction_id),
            "status": "failed",
            "filePath": extraction_request.file_path,
            "configId": str(extraction_request.configuration_id),
            "fieldsToExtract": selected_fields,
            "model": extraction_request.model.value,
            "userId": extraction_request.user_id,
            "extractionResult": None,
            "error": str(e),
            "tokenUsage": token_usage if "token_usage" in locals() else 0,
            "modifiedAtHKTime": formatted_time,
            "modifiedAtIsoTime": iso_formatted_time,
            "timeLapsed": time_elapsed,
        }
        await extraction_container.upsert_item(error_item)
        logger.info(
            "Record updated in Cosmos DB with failure for extraction ID: "
            f"{extraction_request.extraction_id}"
        )

    # finally:
        # Optional: close the client explicitly if desired
        # await client.__aexit__(None, None, None) if hasattr(client, "__aexit__") else None
        

@extraction_router.get("/get_configuration/{config_id}", tags=["Extraction"])
async def get_configuration(
    config_id: str,
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """
    Endpoint to retrieve a configuration by its ID from Cosmos DB.
    """

    try:
        # Use a parameterized query instead of string interpolation
        query = "SELECT * FROM c WHERE c.configId = @config_id"
        items_paged = config_container.query_items(
            query=query,
            parameters=[{"name": "@config_id", "value": config_id}],
            # enable_cross_partition_query=True,
        )

        # Async iteration instead of list(...)
        config_items = [item async for item in items_paged]

        if not config_items:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration with ID {config_id} not found",
            )

        config = config_items[0]
        logger.info(f"Retrieved configuration for config_id: {config_id}")

        # Remove metadata fields
        metadata_fields = ["_rid", "_self", "_etag", "_attachments", "_ts"]
        config = {k: v for k, v in config.items() if k not in metadata_fields}

        return {
            "status": "success",
            "data": config,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving configuration for config_id {config_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")
    
    
@extraction_router.get("/get_configurations", tags=["Extraction"])
async def get_configurations(
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """
    Endpoint to retrieve all configuration IDs from Cosmos DB, sorted by configId and modifiedAtHKTime.
    """

    try:
        query = "SELECT c.configId, c.modifiedAtHKTime FROM c"

        items_paged = config_container.query_items(
            query=query,
            # enable_cross_partition_query=True,
        )

        config_items = [item async for item in items_paged]

        if not config_items:
            logger.info("No configurations found in the database.")
            return {
                "status": "success",
                "data": [],
            }

        config_items.sort(
            key=lambda x: x.get("configId", ""),
        )

        logger.info(f"Retrieved {len(config_items)} configurations.")
        return {
            "status": "success",
            "data": config_items,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving configurations: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")
    
    
@extraction_router.get("/get_user_extractions", tags=["Extraction"])
async def get_user_extractions(
    user_id: str,
    num_of_records: int = Query(
        30,
        ge=1,
        le=100,
        description="Number of records to return (default: 30)",
    ),
    api_key: str = Depends(get_api_key),
) -> List[Dict[str, Any]]:
    """
    Retrieve extractions for a given user_id, ordered by latest modified first.
    """
    try:
        query = f"""
        SELECT TOP {num_of_records}
            c.id,
            c.extractionId,
            c.status,
            c.filePath,
            c.configId,
            c.model,
            c.timeLapsed,
            c.modifiedAtIsoTime,
            c.modifiedAtHKTime
        FROM c
        WHERE c.userId = @user_id
        ORDER BY c.modifiedAtIsoTime DESC
        """

        items_iter = extraction_container.query_items(
            query=query,
            parameters=[{"name": "@user_id", "value": user_id}],
            # partition_key=user_id,
            max_item_count=num_of_records,  # optional, hint per page
        )

        items: List[Dict[str, Any]] = [item async for item in items_iter]

        if not items:
            raise HTTPException(
                status_code=404,
                detail=f"No extractions found for user_id: {user_id}",
            )

        return items

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving extractions for user_id {user_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")
    

@extraction_router.get("/fetch_extraction_result/{extraction_id}", tags=["Extraction"])
async def get_extraction_result(
    extraction_id: str,
    api_key: str = Depends(get_api_key)
) -> Dict[str, Any]:
    """
    Endpoint to retrieve an extraction result by its ID from Cosmos DB.
    """

    try:
        # Use a parameterized query instead of string interpolation
        query = "SELECT * FROM c WHERE c.extractionId = @extraction_id"
        items_paged = extraction_container.query_items(
            query=query,
            parameters=[{"name": "@extraction_id", "value": extraction_id}],
            # enable_cross_partition_query=True,
        )

        # Async iteration instead of list()
        extraction_items = [item async for item in items_paged]

        if not extraction_items:
            raise HTTPException(
                status_code=404,
                detail=f"Extraction result with ID {extraction_id} not found",
            )

        result = extraction_items[0]
        logger.info(f"Retrieved extraction result for extraction_id: {extraction_id}")

        # Remove metadata fields from the result
        metadata_fields = ['_rid', '_self', '_etag', '_attachments', '_ts']
        result = {k: v for k, v in result.items() if k not in metadata_fields}

        return {
            "status": "success",
            "data": result,
        }

    except HTTPException:
        # Let FastAPI handle HTTPExceptions
        raise
    except Exception as e:
        logger.error(f"Error retrieving extraction result for extraction_id {extraction_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


class FieldToExtract(BaseModel):
    columnName: str
    dataType: str
    remarks: str = ""

class UpdateConfigurationRequest(BaseModel):
    fieldsToExtract: List[FieldToExtract]
    
@extraction_router.put("/update_configuration/{config_id}", tags=["Extraction"])
async def update_configuration(
    config_id: str = Path(..., description="The configId of the configuration to update or create"),
    update_req: UpdateConfigurationRequest = Body(...),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """
    Update fieldsToExtract in a configuration and set modified time.
    If configuration does not exist, create a new one.
    Returns the updated/created document including all metadata.
    """

    try:
        # Parameterized query (avoid string interpolation in SQL)
        query = "SELECT * FROM c WHERE c.configId = @config_id"
        items_paged = config_container.query_items(
            query=query,
            parameters=[{"name": "@config_id", "value": config_id}],
            # enable_cross_partition_query=True,
        )

        # Async iteration instead of list(...)
        config_items = [item async for item in items_paged]

        hk_time = datetime.now(hk_timezone)

        if config_items:
            # Update existing config
            config = config_items[0]
            config["fieldsToExtract"] = [field.model_dump() for field in update_req.fieldsToExtract]
            config["modifiedAtHKTime"] = hk_time.strftime("%Y-%m-%d %H:%M:%S")
            config["modifiedAtIsoTime"] = hk_time.isoformat(timespec="seconds")
            action = "updated"
        else:
            # Create new config
            config = {
                "id": config_id,
                "configId": config_id,
                "fieldsToExtract": [field.model_dump() for field in update_req.fieldsToExtract],
                "modifiedAtHKTime": hk_time.strftime("%Y-%m-%d %H:%M:%S"),
                "modifiedAtIsoTime": hk_time.isoformat(timespec="seconds"),
            }
            action = "created"

        # Upsert item (async)
        saved = await config_container.upsert_item(config)
        logger.info(f"Configuration {config_id} {action}.")

        return {
            "status": "success",
            "action": action,
            "data": saved,  # includes all metadata from Cosmos
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating/creating configuration {config_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")
    
    
@extraction_router.delete("/delete_extraction_result/{extraction_id}", tags=["Extraction"])
async def delete_extraction_result(
    extraction_id: str,
    api_key: str = Depends(get_api_key)
) -> Dict[str, Any]:
    """
    Endpoint to delete an extraction result by its ID from Cosmos DB.
    """

    try:
        # Parameterized query instead of string interpolation
        query = "SELECT * FROM c WHERE c.extractionId = @extraction_id"
        items_paged = extraction_container.query_items(
            query=query,
            parameters=[{"name": "@extraction_id", "value": extraction_id}],
            # enable_cross_partition_query=True,
        )

        # Async iteration – do NOT use list()
        extraction_items = [item async for item in items_paged]

        if not extraction_items:
            raise HTTPException(
                status_code=404,
                detail=f"Extraction result with ID {extraction_id} not found"
            )

        # Get the item ID and partition key for deletion
        item_to_delete = extraction_items[0]
        item_id = item_to_delete.get("id")
        partition_key = item_to_delete.get("extractionId")  # adjust if PK is different

        if item_id is None:
            raise HTTPException(
                status_code=500,
                detail="Item found but missing 'id' field; cannot delete."
            )

        # Delete the item from Cosmos DB (async)
        await extraction_container.delete_item(item=item_id, partition_key=partition_key)
        logger.info(f"Deleted extraction result for extraction_id: {extraction_id}")

        return {
            "status": "success",
            "message": f"Extraction result with ID {extraction_id} has been deleted successfully."
        }

    except HTTPException:
        # let FastAPI propagate HTTP-specific errors
        raise
    except Exception as e:
        logger.error(f"Error deleting extraction result for extraction_id {extraction_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")
    
    
@extraction_router.delete("/delete_configuration/{config_id}", tags=["Extraction"])
async def delete_configuration(
    config_id: str = Path(..., description="The configId of the configuration to delete"),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """
    Endpoint to delete a configuration by its ID from Cosmos DB.
    """

    try:
        # Parameterized query (safer than f-string)
        query = "SELECT * FROM c WHERE c.configId = @config_id"
        items_paged = config_container.query_items(
            query=query,
            parameters=[{"name": "@config_id", "value": config_id}],
            # enable_cross_partition_query=True,
        )

        # Async iteration – do NOT use list()
        config_items = [item async for item in items_paged]

        if not config_items:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration with ID {config_id} not found",
            )

        # Get the item ID and partition key for deletion
        item_to_delete = config_items[0]
        item_id = item_to_delete.get("id")
        partition_key = item_to_delete.get("configId")  # adjust if PK is different

        if item_id is None:
            raise HTTPException(
                status_code=500,
                detail="Configuration found but missing 'id' field; cannot delete.",
            )

        # Delete the item from Cosmos DB (async)
        await config_container.delete_item(item=item_id, partition_key=partition_key)
        logger.info(f"Deleted configuration for config_id: {config_id}")

        return {
            "status": "success",
            "message": f"Configuration with ID {config_id} has been deleted successfully.",
        }

    except HTTPException:
        # Let FastAPI handle HTTP errors as-is
        raise
    except Exception as e:
        logger.error(f"Error deleting configuration for config_id {config_id}: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")