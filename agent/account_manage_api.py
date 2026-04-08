import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List

import pytz
from fastapi import APIRouter, HTTPException, Path, Body, Depends
from pydantic import BaseModel
import bcrypt

from clients import accounts_container, get_api_key

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

accounts_router = APIRouter()

hong_kong_tz = pytz.timezone('Asia/Hong_Kong')

class AccountRequest(BaseModel):
    username: str
    password: str
    role: str = "user"

class UpdateAccountRequest(BaseModel):
    password: Optional[str] = None
    role: Optional[str] = None

# =========================
# Add or Update Account
# =========================
@accounts_router.post("/accounts", tags=["Accounts"])
async def add_account(
    account_req: AccountRequest = Body(...),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """Create a new account or update an existing one.

    Password is hashed with bcrypt before storage.  If a user with the
    given username already exists, their password and role are overwritten;
    otherwise a new document is created.

    Returns:
        JSON with ``status``, ``action`` ("created" / "updated"), and the
        saved user document (password hash included).
    """
    username = account_req.username.lower()
    password_hash = bcrypt.hashpw(account_req.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    hk_time = datetime.now(hong_kong_tz)

    try:
        # Check if the user exists
        query = "SELECT * FROM c WHERE c.username = @username"
        params = [{"name": "@username", "value": username}]
        items = [item async for item in accounts_container.query_items(query=query, parameters=params, partition_key=username)]

        if items:
            # Update existing user
            user_doc = items[0]
            user_doc["password_hash"] = password_hash
            user_doc["role"] = account_req.role
            user_doc["modifiedAtHKTime"] = hk_time.strftime('%Y-%m-%d %H:%M:%S')
            user_doc["modifiedAtIsoTime"] = hk_time.isoformat(timespec='seconds')
            action = "updated"
        else:
            # Create new user
            user_doc = {
                "id": username,  # Cosmos required field
                "username": username,
                "password_hash": password_hash,
                "role": account_req.role,
                "createdAtHKTime": hk_time.strftime('%Y-%m-%d %H:%M:%S'),
                "createdAtIsoTime": hk_time.isoformat(timespec='seconds'),
                "modifiedAtHKTime": hk_time.strftime('%Y-%m-%d %H:%M:%S'),
                "modifiedAtIsoTime": hk_time.isoformat(timespec='seconds')
            }
            action = "created"

        # Save to Cosmos
        await accounts_container.upsert_item(user_doc)
        logger.info(f"Account '{username}' {action}.")

        return {
            "status": "success",
            "action": action,
            "data": user_doc
        }
    except Exception as e:
        logger.error(f"Error adding/updating account '{username}': {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# Update Account (partial)
# =========================
@accounts_router.put("/accounts/{username}", tags=["Accounts"])
async def update_account(
    username: str = Path(..., description="Username to update"),
    update_req: UpdateAccountRequest = Body(...),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """Partially update an account's password and/or role.

    Only the fields present in the request body are modified; the rest
    remain unchanged.  Raises 404 if the username does not exist.

    Returns:
        JSON with ``status``, ``action`` ("updated"), and the saved
        user document.
    """
    username = username.lower()
    hk_time = datetime.now(hong_kong_tz)

    try:
        # Fetch user
        query = "SELECT * FROM c WHERE c.username = @username"
        params = [{"name": "@username", "value": username}]
        items = [item async for item in accounts_container.query_items(query=query, parameters=params, partition_key=username)]

        if not items:
            raise HTTPException(status_code=404, detail=f"User '{username}' not found.")

        user_doc = items[0]

        # Update fields
        if update_req.password:
            user_doc["password_hash"] = bcrypt.hashpw(update_req.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        if update_req.role:
            user_doc["role"] = update_req.role

        user_doc["modifiedAtHKTime"] = hk_time.strftime('%Y-%m-%d %H:%M:%S')
        user_doc["modifiedAtIsoTime"] = hk_time.isoformat(timespec='seconds')

        # Save
        await accounts_container.upsert_item(user_doc)
        logger.info(f"Account '{username}' updated.")

        return {
            "status": "success",
            "action": "updated",
            "data": user_doc
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating account '{username}': {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# =========================
# Delete Account
# =========================
@accounts_router.delete("/accounts/{username}", tags=["Accounts"])
async def delete_account(
    username: str = Path(..., description="Username to delete"),
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """Delete an account by username.

    Looks up the user document, then removes it from Cosmos DB using the
    ``username`` partition key.  Raises 404 if not found.

    Returns:
        JSON with ``status``, ``action`` ("deleted"), and the username.
    """
    username = username.lower()

    try:
        # Fetch user
        query = "SELECT * FROM c WHERE c.username = @username"
        params = [{"name": "@username", "value": username}]
        items = [item async for item in accounts_container.query_items(query=query, parameters=params, partition_key=username)]

        if not items:
            raise HTTPException(status_code=404, detail=f"User '{username}' not found.")

        user_doc = items[0]
        await accounts_container.delete_item(item=user_doc["id"], partition_key=user_doc["username"])
        logger.info(f"Account '{username}' deleted.")

        return {
            "status": "success",
            "action": "deleted",
            "username": username
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting account '{username}': {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
