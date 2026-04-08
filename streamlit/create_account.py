import os
import bcrypt
import logging

from azure.cosmos import CosmosClient, PartitionKey, exceptions
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


COSMOS_URL = os.getenv("COSMOS_URL")
COSMOS_KEY = os.getenv("COSMOS_KEY")

# Connect to Cosmos DB
client = CosmosClient(url=COSMOS_URL, credential=COSMOS_KEY)

# Ensure the database exists
DATABASE_ID = "AIExtractionDB"
try:
    database = client.create_database_if_not_exists(id=DATABASE_ID)
except exceptions.CosmosResourceExistsError:
    database = client.get_database_client(DATABASE_ID)

# Create container if not exists (no throughput for serverless)
CONTAINER_ID = "Accounts"
try:
    container = database.create_container_if_not_exists(
        id=CONTAINER_ID,
        partition_key=PartitionKey(path="/username")
    )
except exceptions.CosmosResourceExistsError:
    container = database.get_container_client(CONTAINER_ID)


def load_user(username):
    """Load a user document from the Cosmos DB Accounts container.

    Args:
        username: The username to look up (case-insensitive).

    Returns:
        dict or None: The user document if found, otherwise None.
    """
    username = username.lower()
    query = "SELECT * FROM c WHERE c.username = @username"
    parameters = [{"name": "@username", "value": username}]
    items = list(container.query_items(
        query=query,
        parameters=parameters,
        enable_cross_partition_query=True
    ))
    return items[0] if items else None


def save_user(user):
    """Upsert a user document into the Cosmos DB Accounts container.

    Args:
        user: A dict representing the user document (must include 'id' and
            'username' fields).
    """
    container.upsert_item(user)


def add_or_update_user(username, password, role="user"):
    """Create a new user or update an existing user's password and role.

    The password is hashed with bcrypt before storage. If a user with the
    given username already exists, their password_hash and role are updated
    in place; otherwise a new document is created.

    Args:
        username: The username (case-insensitive).
        password: The plaintext password to hash and store.
        role: The role to assign (default "user").
    """
    username = username.lower()
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    existing_user = load_user(username)
    if existing_user:
        existing_user["password_hash"] = password_hash
        existing_user["role"] = role
        save_user(existing_user)
        print(f"Updated user '{username}' with new password and role '{role}'.")
    else:
        new_user = {
            "id": username,  # Cosmos requires an 'id' field
            "username": username,
            "password_hash": password_hash,
            "role": role
        }
        save_user(new_user)
        print(f"Added new user '{username}' with role '{role}'.")


def verify_user(username, password):
    """Verify a username/password pair against the stored bcrypt hash.

    Args:
        username: The username to verify.
        password: The plaintext password to check.

    Returns:
        tuple: (True, role_string) on success, or (False, None) on failure.
    """
    user = load_user(username)
    if not user:
        return False, None
    if bcrypt.checkpw(password.encode('utf-8'), user["password_hash"].encode('utf-8')):
        return True, user["role"]
    return False, None

if __name__ == "__main__":
    username = "tester"
    password = "password"
    add_or_update_user(username, password, "admin")
    print("✅ User update complete.")

    ok, role = verify_user(username, password)
    print(f"Login success: {ok}, role: {role}")
