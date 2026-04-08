"""Shared utilities for Streamlit pages.

Centralises API wrappers, download helpers, page boilerplate, and constants
that were previously duplicated across individual page files.
"""
import streamlit as st
import requests
import os
import json
import io
import zipfile
import pandas as pd
from typing import Any, Dict, List, Optional
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

# ---------------------------------------------------------------------------
# Environment & API config (loaded once at import time)
# ---------------------------------------------------------------------------
load_dotenv(find_dotenv())

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY")
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATUS_EMOJI: Dict[str, str] = {
    "succeeded": "🟢 Succeeded",
    "failed": "🔴 Failed",
    "processing": "🔄 Processing",
    "queued": "⏳ Queued",
    "not_found": "⚪ Not Found",
    "unknown": "⚪ Unknown",
}

_PAGE_CSS = """
<style>
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 0.5rem !important;
    margin-top: 0rem !important;
}
div[data-testid="stDecoration"] {visibility: hidden; height: 0%; position: fixed;}
div[data-testid="stStatusWidget"] {visibility: hidden; height: 0%; position: fixed;}
</style>
"""

# ---------------------------------------------------------------------------
# Page boilerplate
# ---------------------------------------------------------------------------

def setup_page() -> None:
    """Configure Streamlit page settings and inject shared CSS.

    Must be the first Streamlit call in every page module.
    """
    st.set_page_config(
        page_title="AI Extraction & Validation",
        page_icon="🐱",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)


def require_login(page_label: str = "this page") -> None:
    """Stop the page with an error if the user is not logged in.

    Args:
        page_label: Human-readable name shown in the error message.
    """
    if "username" not in st.session_state:
        st.error(f"You must be logged in to access {page_label}.")
        st.stop()


def render_footer() -> None:
    """Render the standard copyright footer."""
    current_year = datetime.now().year
    st.markdown(
        f"""
        <hr style="margin-top: 2rem; margin-bottom: 0.5rem;">
        <div style="text-align: center; color: gray; font-size: 0.85rem;">
            © {current_year} AI Extraction & Validation
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Response helpers
# ---------------------------------------------------------------------------

def _unwrap_data(resp_json: Any) -> Any:
    """Extract the 'data' payload from an API response envelope.

    Args:
        resp_json: The parsed JSON response from the backend API.

    Returns:
        The value of the 'data' key if present, otherwise the original input.
    """
    if isinstance(resp_json, dict) and "data" in resp_json:
        return resp_json["data"]
    return resp_json


def normalize_status(s: Optional[str]) -> str:
    """Normalize a status string to lowercase, defaulting to 'unknown'.

    Args:
        s: The raw status value (may be None or empty).

    Returns:
        The lowercased, stripped status, or 'unknown' if blank/None.
    """
    return (s or "").strip().lower() or "unknown"


# ---------------------------------------------------------------------------
# Extraction API wrappers
# ---------------------------------------------------------------------------

def fetch_configurations() -> List[Dict]:
    """Fetch all extraction configurations from the backend API.

    Returns:
        A list of configuration documents, or an empty list on failure.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/extraction/get_configurations", headers=HEADERS
        )
        if response.status_code == 200:
            return response.json().get("data", [])
        return []
    except Exception as e:
        st.error(f"Error fetching configurations: {e}")
        return []


def fetch_config_by_id(config_id: str) -> Dict:
    """Fetch a single extraction configuration by its ID.

    Args:
        config_id: The unique configuration identifier.

    Returns:
        The configuration document, or an empty dict on failure.
    """
    try:
        response = requests.get(
            f"{API_BASE_URL}/extraction/get_configuration/{config_id}",
            headers=HEADERS,
        )
        if response.status_code == 200:
            return response.json().get("data", {})
        return {}
    except Exception as e:
        st.error(f"Error fetching configuration {config_id}: {e}")
        return {}


def fetch_extraction_result(extraction_id: str) -> Dict:
    """Fetch the full extraction result for a specific extraction ID.

    Args:
        extraction_id: The unique extraction job identifier.

    Returns:
        The extraction result data, or an empty dict on failure.
    """
    try:
        resp = requests.get(
            f"{API_BASE_URL}/extraction/fetch_extraction_result/{extraction_id}",
            headers=HEADERS,
            timeout=30,
        )
        if resp.status_code == 200:
            return resp.json().get("data", {})
        else:
            st.error(f"Error fetching extraction result: {resp.text}")
            return {}
    except Exception as e:
        st.error(f"Error fetching extraction result: {e}")
        return {}


def fetch_user_extractions(user_id: str, num_of_records: int = 30) -> list:
    """Fetch the most recent extraction jobs for a given user.

    Args:
        user_id: The username whose extractions to retrieve.
        num_of_records: Maximum number of recent records to return.

    Returns:
        A list of extraction record dicts, or an empty list on failure.
    """
    try:
        resp = requests.get(
            f"{API_BASE_URL}/extraction/get_user_extractions",
            params={"user_id": user_id, "num_of_records": num_of_records},
            headers=HEADERS,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"Error fetching extractions: {resp.text}")
            return []
    except Exception as e:
        st.error(f"Error fetching extractions: {e}")
        return []


# ---------------------------------------------------------------------------
# Validation API wrappers
# ---------------------------------------------------------------------------

def fetch_rules() -> List[Dict[str, Any]]:
    """Fetch all validation rule sets from the backend API.

    Returns:
        A list of rule documents, or an empty list on failure.
    """
    try:
        r = requests.get(
            f"{API_BASE_URL}/validation/get_rules", headers=HEADERS, timeout=30
        )
        if r.status_code != 200:
            return []
        data = _unwrap_data(r.json())
        return data if isinstance(data, list) else []
    except Exception as e:
        st.error(f"Error fetching rules: {e}")
        return []


def fetch_rule_by_id(rule_id: str) -> Dict[str, Any]:
    """Fetch a single validation rule by its ID.

    Args:
        rule_id: The unique rule identifier.

    Returns:
        The rule document, or an empty dict on failure.
    """
    try:
        r = requests.get(
            f"{API_BASE_URL}/validation/get_rule/{rule_id}",
            headers=HEADERS,
            timeout=30,
        )
        if r.status_code != 200:
            return {}
        data = _unwrap_data(r.json())
        return data if isinstance(data, dict) else {}
    except Exception as e:
        st.error(f"Error fetching rule '{rule_id}': {e}")
        return {}


def fetch_validation_result(validation_id: str) -> Dict[str, Any]:
    """Fetch the full validation result document by validation ID.

    Handles both the envelope format (with 'data' key) and raw dict
    responses.  Returns a stub with ``status: not_found`` on 404.

    Args:
        validation_id: The unique validation job identifier.

    Returns:
        The validation result data, or an empty dict / not_found stub
        on failure.
    """
    try:
        resp = requests.get(
            f"{API_BASE_URL}/validation/fetch_validation_result/{validation_id}",
            headers=HEADERS,
            timeout=30,
        )
        if resp.status_code == 200:
            payload = resp.json()
            if isinstance(payload, dict) and "data" in payload:
                data = payload.get("data") or {}
                return data if isinstance(data, dict) else {}
            return payload if isinstance(payload, dict) else {}
        if resp.status_code == 404:
            return {"validationId": validation_id, "status": "not_found"}
        st.error(f"Error fetching validation result (HTTP {resp.status_code}): {resp.text}")
        return {}
    except Exception as e:
        st.error(f"Error fetching validation result: {e}")
        return {}


def fetch_user_validations(user_id: str, num_of_records: int = 30) -> List[Dict[str, Any]]:
    """Fetch the most recent validation jobs for a given user.

    Args:
        user_id: The username whose validations to retrieve.
        num_of_records: Maximum number of recent records to return.

    Returns:
        A list of validation record dicts, or an empty list on failure.
    """
    try:
        resp = requests.get(
            f"{API_BASE_URL}/validation/get_user_validations",
            params={"user_id": user_id, "num_of_records": num_of_records},
            headers=HEADERS,
            timeout=30,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data if isinstance(data, list) else []
        if resp.status_code == 404:
            return []
        st.error(f"Error fetching validations (HTTP {resp.status_code}): {resp.text}")
        return []
    except Exception as e:
        st.error(f"Error fetching validations: {e}")
        return []


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def prepare_json_download(data: dict) -> bytes:
    """Serialize data to pretty-printed JSON bytes for download.

    Args:
        data: The dict to serialize.

    Returns:
        UTF-8 encoded JSON string.
    """
    return json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")


def prepare_excel_download(data: dict) -> bytes:
    """Convert data to an Excel workbook in memory.

    Scalar fields go into a 'Metadata' sheet.  Each list-of-dicts or dict
    value gets its own sheet, with names truncated to Excel's 31-char limit
    and de-duplicated as needed.

    Args:
        data: The dict to convert.

    Returns:
        Raw bytes of an ``.xlsx`` file.
    """
    output = io.BytesIO()
    used_sheet_names: set = set()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        summary_data = {k: v for k, v in data.items() if not isinstance(v, (list, dict))}
        if summary_data:
            pd.DataFrame([summary_data]).to_excel(writer, sheet_name="Metadata", index=False)
            used_sheet_names.add("Metadata")

        for key, value in data.items():
            if isinstance(value, (list, dict)):
                cleaned_key = key.replace(" ", "")
                base_name = cleaned_key[:31]
                sheet_name = base_name
                counter = 1
                while sheet_name in used_sheet_names:
                    suffix = f"_{counter}"
                    sheet_name = base_name[:31 - len(suffix)] + suffix
                    counter += 1
                used_sheet_names.add(sheet_name)
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    pd.DataFrame(value).to_excel(writer, sheet_name=sheet_name, index=False)
                elif isinstance(value, dict):
                    pd.DataFrame([value]).to_excel(writer, sheet_name=sheet_name, index=False)
    return output.getvalue()


def prepare_csv_download(data: dict) -> bytes:
    """Convert data to a ZIP archive containing CSV files.

    Scalar fields go into ``Metadata.csv``.  Each list-of-dicts or dict
    value becomes a separate CSV file inside the ZIP.

    Args:
        data: The dict to convert.

    Returns:
        Raw bytes of a ZIP archive containing CSV files.
    """
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zipf:
        summary_data = {k: v for k, v in data.items() if not isinstance(v, (list, dict))}
        if summary_data:
            csv_buffer = io.StringIO()
            pd.DataFrame([summary_data]).to_csv(csv_buffer, index=False)
            zipf.writestr("Metadata.csv", csv_buffer.getvalue())
            csv_buffer.close()

        used_filenames = {"Metadata.csv"}
        for key, value in data.items():
            if isinstance(value, (list, dict)):
                cleaned_key = key.replace(" ", "")
                base_name = cleaned_key[:100]
                csv_filename = f"{base_name}.csv"
                counter = 1
                while csv_filename in used_filenames:
                    suffix = f"_{counter}"
                    csv_filename = f"{base_name[:100 - len(suffix)]}{suffix}.csv"
                    counter += 1
                used_filenames.add(csv_filename)
                csv_buffer = io.StringIO()
                if isinstance(value, list) and value and isinstance(value[0], dict):
                    pd.DataFrame(value).to_csv(csv_buffer, index=False)
                elif isinstance(value, dict):
                    pd.DataFrame([value]).to_csv(csv_buffer, index=False)
                zipf.writestr(csv_filename, csv_buffer.getvalue())
                csv_buffer.close()
    return output.getvalue()
