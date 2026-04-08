import uuid
import asyncio
import pytz
import logging
from fastapi import (
    BackgroundTasks, 
    HTTPException, 
    APIRouter, 
    Depends, 
    Path, 
    Body, 
    Query
)
from pydantic import BaseModel, Field
from typing import Annotated, Any, List, Optional, Dict
import time
from datetime import datetime
import traceback
from azure.cosmos.exceptions import CosmosResourceNotFoundError

from validation import run_validation_workflow
from clients import (
    extraction_container,
    validation_container,
    rule_container,
    get_api_key,
)

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

validation_router = APIRouter()

hk_timezone = pytz.timezone("Asia/Hong_Kong")


def _escape_braces(value: Any) -> str:
    """Escape ``{`` and ``}`` so the value is safe inside PromptTemplate strings."""
    return str(value).replace("{", "{{").replace("}", "}}")


class ValidationRequest(BaseModel):
    validation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    extraction_ids: List[str] = Field(min_length=1)
    rule_id: str
    external_data: Optional[str] = None
    user_id: str


@validation_router.post("/validate_extractions", tags=["Validation"])
async def validate_extractions(
    validation_request: ValidationRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """Queue a validation job and return the validation_id immediately.

    Writes a ``queued`` record to Cosmos DB, then spawns a background task
    that progresses through ``processing`` → ``succeeded`` / ``failed``.
    Clients should poll ``/fetch_validation_result/{validation_id}`` to
    track progress.
    """
    try:
        hk_time = datetime.now(hk_timezone)
        formatted_time = hk_time.strftime("%Y-%m-%d %H:%M:%S")
        iso_formatted_time = hk_time.isoformat(timespec="seconds")

        queued_item = {
            "id": str(validation_request.validation_id),
            "validationId": str(validation_request.validation_id),
            "status": "queued",
            "ruleId": str(validation_request.rule_id),
            "extractionIds": validation_request.extraction_ids,
            "externalData": validation_request.external_data,
            "userId": validation_request.user_id,
            "validationResult": None,
            "error": None,
            "tokenUsage": {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
            "modifiedAtHKTime": formatted_time,
            "modifiedAtIsoTime": iso_formatted_time,
        }
        await validation_container.upsert_item(queued_item)

        background_tasks.add_task(run_validation_task, validation_request, hk_timezone)

        return {
            "status": "queued",
            "validation_id": validation_request.validation_id,
            "message": "Validation task has been queued. Use the validation_id to check status.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error queuing validation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


async def run_validation_task(
    validation_request: ValidationRequest,
    hk_timezone,
) -> None:
    """Background task that runs the full validation workflow.

    Steps:
    1. Mark Cosmos record as ``processing``.
    2. Fetch all referenced extraction results (concurrently).
    3. Fetch the rule document and its rule sets.
    4. Run ``run_validation_workflow`` (LLM-based compliance checking).
    5. Write ``succeeded`` or ``failed`` result back to Cosmos.

    Any exception — including HTTP errors from missing data — is caught
    and persisted as a ``failed`` record so jobs never stay stuck.
    """
    start_time = time.time()
    result_payload: Dict[str, Any] = {}
    try:
        hk_time = datetime.now(hk_timezone)
        formatted_time = hk_time.strftime("%Y-%m-%d %H:%M:%S")
        iso_formatted_time = hk_time.isoformat(timespec="seconds")

        processing_item = {
            "id": str(validation_request.validation_id),
            "validationId": str(validation_request.validation_id),
            "status": "processing",
            "ruleId": str(validation_request.rule_id),
            "extractionIds": validation_request.extraction_ids,
            "externalData": validation_request.external_data,
            "userId": validation_request.user_id,
            "validationResult": None,
            "error": None,
            "tokenUsage": {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            },
            "modifiedAtHKTime": formatted_time,
            "modifiedAtIsoTime": iso_formatted_time,
        }
        await validation_container.upsert_item(processing_item)

        async def _read_extraction(extraction_id: str) -> Dict[str, Any]:
            try:
                return await extraction_container.read_item(
                    item=extraction_id,
                    partition_key=extraction_id,
                )
            except CosmosResourceNotFoundError:
                raise HTTPException(status_code=404, detail=f"Extraction ID not found: {extraction_id}")

        extraction_records = await asyncio.gather(
            *[_read_extraction(eid) for eid in validation_request.extraction_ids]
        )

        candidates: Dict[str, Any] = {}
        for rec in extraction_records:
            eid = rec.get("extractionId") or rec.get("id")

            if rec.get("status") != "succeeded":
                raise HTTPException(
                    status_code=400,
                    detail=f"Extraction not ready (status={rec.get('status')}): {eid}",
                )

            extraction_result = rec.get("extractionResult")
            if extraction_result is None:
                raise HTTPException(status_code=400, detail=f"Missing extractionResult for extraction ID: {eid}")

            candidates[str(eid)] = _escape_braces(extraction_result)

        if validation_request.external_data is not None:
            candidates["external_data"] = validation_request.external_data

        try:
            rule_doc = await rule_container.read_item(
                item=validation_request.rule_id,
                partition_key=validation_request.rule_id,
            )
        except CosmosResourceNotFoundError:
            raise HTTPException(status_code=404, detail=f"Rule ID not found: {validation_request.rule_id}")

        rule_sets = rule_doc.get("ruleSets", [])
        if not isinstance(rule_sets, list) or not rule_sets:
            raise HTTPException(
                status_code=400,
                detail=f"No ruleSets found for rule ID: {validation_request.rule_id}",
            )

        logger.info(
            "Starting validation workflow. "
            f"validation_id={validation_request.validation_id} "
            f"rules_count={len(rule_sets)} "
            f"extractions_count={len(validation_request.extraction_ids)}"
        )

        result_payload = await run_validation_workflow(
            candidates=candidates,
            rule_sets=rule_sets,
        )

        hk_time = datetime.now(hk_timezone)
        formatted_time = hk_time.strftime("%Y-%m-%d %H:%M:%S")
        iso_formatted_time = hk_time.isoformat(timespec="seconds")
        time_elapsed = time.time() - start_time

        status = "failed" if result_payload.get("error") else "succeeded"

        updated_item = {
            "id": str(validation_request.validation_id),
            "validationId": str(validation_request.validation_id),
            "status": status,
            "ruleId": str(validation_request.rule_id),
            "extractionIds": validation_request.extraction_ids,
            "externalData": validation_request.external_data,
            "userId": validation_request.user_id,
            "candidates": candidates,
            "validationResult": result_payload.get("validation_result"),
            "error": result_payload.get("error"),
            "tokenUsage": result_payload.get(
                "token_usage",
                {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
            ),
            "modifiedAtHKTime": formatted_time,
            "modifiedAtIsoTime": iso_formatted_time,
            "timeLapsed": time_elapsed,
        }
        await validation_container.upsert_item(updated_item)

        logger.info(
            "Validation task completed. "
            f"validation_id={validation_request.validation_id} status={status}"
        )

    except HTTPException as e:
        hk_time = datetime.now(hk_timezone)
        formatted_time = hk_time.strftime("%Y-%m-%d %H:%M:%S")
        iso_formatted_time = hk_time.isoformat(timespec="seconds")
        time_elapsed = time.time() - start_time

        error_item = {
            "id": str(validation_request.validation_id),
            "validationId": str(validation_request.validation_id),
            "status": "failed",
            "ruleId": str(validation_request.rule_id),
            "extractionIds": validation_request.extraction_ids,
            "externalData": validation_request.external_data,
            "userId": validation_request.user_id,
            "validationResult": None,
            "error": str(e.detail),
            "tokenUsage": result_payload.get(
                "token_usage",
                {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
            )
            if isinstance(result_payload, dict)
            else {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
            "modifiedAtHKTime": formatted_time,
            "modifiedAtIsoTime": iso_formatted_time,
            "timeLapsed": time_elapsed,
        }
        await validation_container.upsert_item(error_item)
        logger.error(
            "Validation task failed with HTTPException. "
            f"validation_id={validation_request.validation_id} error={e.detail}"
        )

    except Exception as e:
        traceback.print_exc()

        hk_time = datetime.now(hk_timezone)
        formatted_time = hk_time.strftime("%Y-%m-%d %H:%M:%S")
        iso_formatted_time = hk_time.isoformat(timespec="seconds")
        time_elapsed = time.time() - start_time

        error_item = {
            "id": str(validation_request.validation_id),
            "validationId": str(validation_request.validation_id),
            "status": "failed",
            "ruleId": str(validation_request.rule_id),
            "extractionIds": validation_request.extraction_ids,
            "externalData": validation_request.external_data,
            "userId": validation_request.user_id,
            "validationResult": None,
            "error": f"Unexpected error during validation: {str(e)}",
            "tokenUsage": result_payload.get(
                "token_usage",
                {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
            )
            if isinstance(result_payload, dict)
            else {"total_tokens": 0, "prompt_tokens": 0, "completion_tokens": 0},
            "modifiedAtHKTime": formatted_time,
            "modifiedAtIsoTime": iso_formatted_time,
            "timeLapsed": time_elapsed,
        }
        await validation_container.upsert_item(error_item)
        logger.error(
            "Validation task failed with Exception. "
            f"validation_id={validation_request.validation_id}",
            exc_info=True,
        )
        
        
@validation_router.get("/get_rule/{rule_id}", tags=["Validation"])
async def get_rule(
    rule_id: str,
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """
    Endpoint to retrieve a rule by its ID from Cosmos DB.
    """
    try:
        query = "SELECT * FROM c WHERE c.ruleId = @rule_id"
        items_paged = rule_container.query_items(
            query=query,
            parameters=[{"name": "@rule_id", "value": rule_id}],
        )

        rule_items = [item async for item in items_paged]

        if not rule_items:
            raise HTTPException(
                status_code=404,
                detail=f"Rule with ID {rule_id} not found",
            )

        rule_doc = rule_items[0]
        logger.info(f"Retrieved rule for rule_id: {rule_id}")

        metadata_fields = ["_rid", "_self", "_etag", "_attachments", "_ts"]
        rule_doc = {k: v for k, v in rule_doc.items() if k not in metadata_fields}

        return {"status": "success", "data": rule_doc}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving rule for rule_id {rule_id}: {e}", exc_info=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


@validation_router.get("/get_rules", tags=["Validation"])
async def get_rules(
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """
    Endpoint to retrieve all rule IDs from Cosmos DB, sorted by ruleId and modifiedAtHKTime.
    """
    try:
        query = "SELECT c.ruleId, c.modifiedAtHKTime FROM c"
        items_paged = rule_container.query_items(query=query)

        rule_items = [item async for item in items_paged]

        if not rule_items:
            logger.info("No rules found in the database.")
            return {"status": "success", "data": []}

        rule_items.sort(key=lambda x: x.get("ruleId", ""))

        logger.info(f"Retrieved {len(rule_items)} rules.")
        return {"status": "success", "data": rule_items}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving rules: {e}", exc_info=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


@validation_router.get("/get_user_validations", tags=["Validation"])
async def get_user_validations(
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
    Retrieve validations for a given user_id, ordered by latest modified first.
    """
    try:
        query = f"""
        SELECT TOP {num_of_records}
            c.id,
            c.validationId,
            c.status,
            c.ruleId,
            c.timeLapsed,
            c.modifiedAtIsoTime,
            c.modifiedAtHKTime
        FROM c
        WHERE c.userId = @user_id
        ORDER BY c.modifiedAtIsoTime DESC
        """
        
        items_iter = validation_container.query_items(
            query=query,
            parameters=[{"name": "@user_id", "value": user_id}],
            max_item_count=num_of_records,
        )

        items: List[Dict[str, Any]] = [item async for item in items_iter]

        if not items:
            raise HTTPException(
                status_code=404,
                detail=f"No validations found for user_id: {user_id}",
            )

        return items[:num_of_records]

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving validations for user_id {user_id}: {e}", exc_info=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


@validation_router.get("/fetch_validation_result/{validation_id}", tags=["Validation"])
async def get_validation_result(
    validation_id: str,
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """
    Endpoint to retrieve a validation result by its ID from Cosmos DB.
    """
    try:
        query = "SELECT * FROM c WHERE c.validationId = @validation_id"
        items_paged = validation_container.query_items(
            query=query,
            parameters=[{"name": "@validation_id", "value": validation_id}],
        )

        validation_items = [item async for item in items_paged]

        if not validation_items:
            raise HTTPException(
                status_code=404,
                detail=f"Validation result with ID {validation_id} not found",
            )

        result = validation_items[0]
        logger.info(f"Retrieved validation result for validation_id: {validation_id}")

        metadata_fields = ["_rid", "_self", "_etag", "_attachments", "_ts"]
        result = {k: v for k, v in result.items() if k not in metadata_fields}

        return {
            "status": "success",
            "data": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving validation result for validation_id {validation_id}: {e}", exc_info=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


class RuleSet(BaseModel):
    alias: str
    rule: str


class UpdateRuleRequest(BaseModel):
    ruleSets: List[RuleSet]


@validation_router.put("/update_rule/{rule_id}", tags=["Validation"])
async def update_rule(
    rule_id: str = Path(..., description="The ruleId of the rule to update or create"),
    update_req: UpdateRuleRequest = Body(...),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """
    Update ruleSets in a rule and set modified time.
    If rule does not exist, create a new one.
    Returns the updated/created document including all metadata.
    """
    try:
        query = "SELECT * FROM c WHERE c.ruleId = @rule_id"
        items_paged = rule_container.query_items(
            query=query,
            parameters=[{"name": "@rule_id", "value": rule_id}],
        )
        rule_items = [item async for item in items_paged]

        hk_time = datetime.now(hk_timezone)

        if rule_items:
            rule_doc = rule_items[0]
            rule_doc["ruleSets"] = [rs.model_dump() for rs in update_req.ruleSets]
            rule_doc["modifiedAtHKTime"] = hk_time.strftime("%Y-%m-%d %H:%M:%S")
            rule_doc["modifiedAtIsoTime"] = hk_time.isoformat(timespec="seconds")
            action = "updated"
        else:
            rule_doc = {
                "id": rule_id,
                "ruleId": rule_id,
                "ruleSets": [rs.model_dump() for rs in update_req.ruleSets],
                "modifiedAtHKTime": hk_time.strftime("%Y-%m-%d %H:%M:%S")
                ,
                "modifiedAtIsoTime": hk_time.isoformat(timespec="seconds"),
            }
            action = "created"

        saved = await rule_container.upsert_item(rule_doc)
        logger.info(f"Rule {rule_id} {action}.")

        return {
            "status": "success",
            "action": action,
            "data": saved,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating/creating rule {rule_id}: {e}", exc_info=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


@validation_router.delete("/delete_validation_result/{validation_id}", tags=["Validation"])
async def delete_validation_result(
    validation_id: str,
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """
    Endpoint to delete a validation result by its ID from Cosmos DB.
    """
    try:
        query = "SELECT * FROM c WHERE c.validationId = @validation_id"
        items_paged = validation_container.query_items(
            query=query,
            parameters=[{"name": "@validation_id", "value": validation_id}],
        )

        validation_items = [item async for item in items_paged]

        if not validation_items:
            raise HTTPException(
                status_code=404,
                detail=f"Validation result with ID {validation_id} not found",
            )

        item_to_delete = validation_items[0]
        item_id = item_to_delete.get("id")
        partition_key = item_to_delete.get("validationId")

        if item_id is None:
            raise HTTPException(
                status_code=500,
                detail="Item found but missing 'id' field; cannot delete.",
            )

        await validation_container.delete_item(item=item_id, partition_key=partition_key)
        logger.info(f"Deleted validation result for validation_id: {validation_id}")

        return {
            "status": "success",
            "message": f"Validation result with ID {validation_id} has been deleted successfully.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting validation result for validation_id {validation_id}: {e}", exc_info=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")


@validation_router.delete("/delete_rule/{rule_id}", tags=["Validation"])
async def delete_rule(
    rule_id: str = Path(..., description="The ruleId of the rule to delete"),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """
    Endpoint to delete a rule by its ID from Cosmos DB.
    """
    try:
        query = "SELECT * FROM c WHERE c.ruleId = @rule_id"
        items_paged = rule_container.query_items(
            query=query,
            parameters=[{"name": "@rule_id", "value": rule_id}],
        )

        rule_items = [item async for item in items_paged]

        if not rule_items:
            raise HTTPException(
                status_code=404,
                detail=f"Rule with ID {rule_id} not found",
            )

        item_to_delete = rule_items[0]
        item_id = item_to_delete.get("id")
        partition_key = item_to_delete.get("ruleId")

        if item_id is None:
            raise HTTPException(
                status_code=500,
                detail="Rule found but missing 'id' field; cannot delete.",
            )

        await rule_container.delete_item(item=item_id, partition_key=partition_key)
        logger.info(f"Deleted rule for rule_id: {rule_id}")

        return {
            "status": "success",
            "message": f"Rule with ID {rule_id} has been deleted successfully.",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting rule for rule_id {rule_id}: {e}", exc_info=True)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail="Internal server error")