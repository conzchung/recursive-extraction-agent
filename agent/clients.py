"""
Shared Azure clients and API key authentication.

All Cosmos DB containers, Blob Storage client, and the FastAPI API-key
dependency are initialised here once and imported by the router modules.
"""

import os
from dotenv import load_dotenv, find_dotenv
from fastapi import Depends, HTTPException
from fastapi.security import APIKeyHeader

from azure.cosmos.aio import CosmosClient as AsyncCosmosClient
from azure.cosmos import CosmosClient as SyncCosmosClient
from azure.storage.blob.aio import BlobServiceClient

load_dotenv(find_dotenv())

# ── Async Cosmos DB (extraction & validation APIs) ──────────────────
_async_cosmos = AsyncCosmosClient(
    url=os.getenv("COSMOS_URL"),
    credential=os.getenv("COSMOS_KEY"),
)
_database = _async_cosmos.get_database_client("AIExtractionDB")

extraction_container = _database.get_container_client("ExtractionResults")
config_container = _database.get_container_client("Configurations")
validation_container = _database.get_container_client("ValidationResults")
rule_container = _database.get_container_client("Rules")
accounts_container = _database.get_container_client("Accounts")

# ── Sync Cosmos DB (account management) ─────────────────────────────
def get_accounts_container():
    """Return the Accounts container via the sync Cosmos SDK."""
    client = SyncCosmosClient(
        url=os.getenv("COSMOS_URL"),
        credential=os.getenv("COSMOS_KEY"),
    )
    database = client.get_database_client("AIExtractionDB")
    return database.get_container_client("Accounts")

# ── Azure Blob Storage ──────────────────────────────────────────────
blob_service_client = BlobServiceClient.from_connection_string(
    os.getenv("AZURE_BLOB_CONNECTION_STRING")
)

# ── API key authentication ──────────────────────────────────────────
API_KEY = os.getenv("API_KEY")
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Depends(_api_key_header)):
    """FastAPI dependency that validates the X-API-Key header.

    Returns:
        The API key string if valid.

    Raises:
        HTTPException: 403 if the key is missing or incorrect.
    """
    if api_key == API_KEY:
        return api_key
    raise HTTPException(status_code=403, detail="Could not validate API Key")
