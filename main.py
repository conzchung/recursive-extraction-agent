import os
import sys
import logging
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv, find_dotenv
import pytz


CURPATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append(CURPATH)
sys.path.append(os.path.join(CURPATH, "agent"))

# Load environment variables
load_dotenv(find_dotenv())

# -------------------------------
# Logging Configuration
# -------------------------------
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detail
    format="[%(levelname)s] %(name)s: %(message)s",
)

# Reduce noise from some libraries
logging.getLogger("uvicorn.error").setLevel(logging.INFO)
logging.getLogger("uvicorn.access").setLevel(logging.INFO)
logging.getLogger("azure").setLevel(logging.WARNING)

# Create a logger for this module
logger = logging.getLogger(__name__)
logger.info("Logging is configured and ready.")

root_path = os.getenv('ROOT_PATH', '')
api_server = os.getenv('API_SERVER', 'http://localhost:8000')

# Ensure URL scheme is correct
if not api_server.startswith(('http://', 'https://')):
    raise ValueError("API_SERVER URL scheme must be 'http' or 'https'")


async def _recover_stuck_tasks():
    """Mark any 'processing' records as 'failed' on startup.

    If the previous process crashed, background tasks that were in-flight
    will have left records stuck at 'processing' forever.  Since no old
    tasks survive a restart, it is safe to fail them all.
    """
    from agent.clients import extraction_container, validation_container

    hk_tz = pytz.timezone("Asia/Hong_Kong")
    hk_time = datetime.now(hk_tz)
    formatted_time = hk_time.strftime("%Y-%m-%d %H:%M:%S")
    iso_formatted_time = hk_time.isoformat(timespec="seconds")

    query = "SELECT * FROM c WHERE c.status = 'processing'"
    recovered = 0

    # --- Extraction ---
    stuck = [item async for item in extraction_container.query_items(query=query)]
    for item in stuck:
        item["status"] = "failed"
        item["error"] = "Process restarted while task was in progress"
        item["modifiedAtHKTime"] = formatted_time
        item["modifiedAtIsoTime"] = iso_formatted_time
        await extraction_container.upsert_item(item)
        logger.warning(f"Recovered stuck extraction: {item.get('extractionId')}")
    recovered += len(stuck)

    # --- Validation ---
    stuck = [item async for item in validation_container.query_items(query=query)]
    for item in stuck:
        item["status"] = "failed"
        item["error"] = "Process restarted while task was in progress"
        item["modifiedAtHKTime"] = formatted_time
        item["modifiedAtIsoTime"] = iso_formatted_time
        await validation_container.upsert_item(item)
        logger.warning(f"Recovered stuck validation: {item.get('validationId')}")
    recovered += len(stuck)

    if recovered:
        logger.info(f"Recovered {recovered} stuck task(s) from previous run")


@asynccontextmanager
async def lifespan(app):
    """FastAPI lifespan handler for startup/shutdown hooks.

    On startup: recovers any Cosmos DB records stuck at 'processing' from
    a previous crash, then logs readiness.
    On shutdown: logs a clean shutdown message.
    """
    await _recover_stuck_tasks()
    logger.info("Startup complete")
    yield
    logger.info("Shutting down")


# Initialize FastAPI
app = FastAPI(
    lifespan=lifespan,
    root_path=root_path,
    servers=[
        { "url": api_server, "description": "UAT Environment" }
    ],
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.get("/")
async def healthcheck():
    """Simple liveness probe — returns 200 if the process is running."""
    return "Service is RUNNING"
    
# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],  # Update with your specific origins
    allow_methods=["*"],  # Update with your specific HTTP methods
    allow_headers=["*"],  # Update with your specific headers
)


from agent import (
    extraction_api, 
    validation_api,
    account_manage_api,
)

# Include routers
app.include_router(extraction_api.extraction_router, prefix="/extraction")
app.include_router(validation_api.validation_router, prefix="/validation")
app.include_router(account_manage_api.accounts_router, prefix="/account_management")


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)