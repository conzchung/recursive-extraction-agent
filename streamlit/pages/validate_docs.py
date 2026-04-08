import streamlit as st
import requests
import time
import pandas as pd
from typing import Any, List, Dict, Optional, Tuple
from datetime import datetime
from requests.exceptions import RequestException

from shared_utils import (
    setup_page, render_footer, require_login,
    API_BASE_URL, HEADERS, STATUS_EMOJI,
    _unwrap_data, normalize_status,
    fetch_rules, fetch_rule_by_id, fetch_user_extractions,
)

setup_page()
require_login("the validation tool")

MAX_EXTRACTION_IDS = 5


# --- Page-specific helpers (enhanced error handling for readiness checks) ---

def escape_curly_braces(text: str) -> str:
    """Escape curly braces in user-supplied text to prevent format-string issues.

    Args:
        text: The raw text string to escape.

    Returns:
        str: The text with '{' and '}' doubled.
    """
    return text.replace("{", "{{").replace("}", "}}")


def parse_extraction_ids(raw: str) -> List[str]:
    """Parse a free-text input into a de-duplicated list of extraction IDs.

    Accepts commas, spaces, and newlines as delimiters. Preserves the order
    of first appearance while removing duplicates.

    Args:
        raw: The raw text from the user input field.

    Returns:
        List[str]: Ordered, unique extraction ID strings.
    """
    if not raw or not raw.strip():
        return []
    tokens: List[str] = []
    for line in raw.splitlines():
        for part in line.replace(",", " ").split():
            cleaned = part.strip()
            if cleaned:
                tokens.append(cleaned)

    seen = set()
    deduped: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def _fetch_extraction_result_with_status(extraction_id: str) -> Dict[str, Any]:
    """Fetch extraction result, returning status stubs on 404/error.

    Unlike the shared fetch_extraction_result (which returns ``{}`` on error),
    this version always returns a dict with a ``status`` key so that
    check_extractions_ready can report per-ID readiness.

    Args:
        extraction_id: The unique extraction job identifier.

    Returns:
        Dict with at least a 'status' key.
    """
    try:
        r = requests.get(
            f"{API_BASE_URL}/extraction/fetch_extraction_result/{extraction_id}",
            headers=HEADERS,
            timeout=30,
        )

        if r.status_code == 200:
            data = _unwrap_data(r.json())
            return data if isinstance(data, dict) else {"status": "unknown"}

        if r.status_code == 404:
            return {"extractionId": extraction_id, "status": "not_found"}

        st.error(f"Error fetching extraction result for {extraction_id}: {r.text}")
        return {"extractionId": extraction_id, "status": "unknown"}

    except Exception as e:
        st.error(f"Error fetching extraction result for {extraction_id}: {e}")
        return {"extractionId": extraction_id, "status": "unknown"}


def queue_validation(
    validation_id: str,
    extraction_ids: List[str],
    rule_id: str,
    external_data: Optional[str],
    max_retries: int = 2,
    retry_delay: int = 5,
) -> Dict[str, Any]:
    """Submit a validation job to the backend API with retry logic.

    Sends a POST request to queue the validation. The backend processes
    it asynchronously. Retries on network or server errors.

    Args:
        validation_id: A user-defined identifier for this validation job.
        extraction_ids: List of extraction IDs whose results will be validated.
        rule_id: The ID of the rule set to validate against.
        external_data: Optional free-text external data to include in the
            validation context (curly braces should already be escaped).
        max_retries: Maximum number of retry attempts on failure.
        retry_delay: Seconds to wait between retries.

    Returns:
        Dict[str, Any]: The API response body on success, or an empty dict
            if all retries are exhausted.
    """
    payload = {
        "validation_id": validation_id,
        "extraction_ids": extraction_ids,
        "rule_id": rule_id,
        "external_data": external_data,
        "user_id": st.session_state.get("username", "unknown"),
    }

    attempt = 0
    while attempt < max_retries:
        try:
            r = requests.post(
                f"{API_BASE_URL}/validation/validate_extractions",
                json=payload,
                headers=HEADERS,
                timeout=30,
            )
            if r.status_code == 200:
                return r.json()

            st.error(f"Error during validation (Attempt {attempt+1}/{max_retries}): {r.text}")
            attempt += 1
            if attempt < max_retries:
                st.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        except RequestException as e:
            st.error(f"Network error during validation (Attempt {attempt+1}/{max_retries}): {e}")
            attempt += 1
            if attempt < max_retries:
                st.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    st.error("Validation failed after maximum retries.")
    return {}


def combined_unique_ids(a: List[str], b: List[str]) -> List[str]:
    """Merge two lists of IDs into a single de-duplicated list preserving order.

    Args:
        a: First list of ID strings (e.g., from the multiselect widget).
        b: Second list of ID strings (e.g., from manual text input).

    Returns:
        List[str]: Combined unique IDs in order of first appearance.
    """
    out: List[str] = []
    seen = set()
    for eid in a + b:
        if eid and eid not in seen:
            seen.add(eid)
            out.append(eid)
    return out


def check_extractions_ready(extraction_ids: List[str]) -> Tuple[List[Dict[str, Any]], bool, Dict[str, int]]:
    """Check whether all given extraction IDs have succeeded.

    Fetches each extraction's status from the backend and builds a summary
    table. Used to gate validation submission -- all must be 'succeeded'.

    Args:
        extraction_ids: The list of extraction IDs to check.

    Returns:
        tuple: A 3-tuple of:
            - rows: List of per-extraction status dicts for display.
            - all_ready: True only if every extraction has status 'succeeded'.
            - counts: Dict mapping each status string to its occurrence count.
    """
    rows: List[Dict[str, Any]] = []
    all_ready = True
    counts: Dict[str, int] = {}

    for eid in extraction_ids:
        data = _fetch_extraction_result_with_status(eid)
        status_norm = normalize_status(data.get("status"))
        counts[status_norm] = counts.get(status_norm, 0) + 1

        ready = status_norm == "succeeded"
        all_ready = all_ready and ready

        rows.append(
            {
                "extractionId": eid,
                "status": status_norm,
                "statusLabel": STATUS_EMOJI.get(status_norm, STATUS_EMOJI["unknown"]),
                "configId": data.get("configId"),
                "model": data.get("model"),
                "modifiedAtHKTime": data.get("modifiedAtHKTime"),
                "readyToValidate": ready,
            }
        )

    return rows, all_ready, counts


st.title("Content Validation 🛡️")
st.write("- Submission is blocked until all selected extraction IDs are ready.")

username = st.session_state.get("username", "unknown")

if "validation_rules" not in st.session_state:
    st.session_state.validation_rules = fetch_rules()

if "user_extractions" not in st.session_state:
    st.session_state.user_extractions = fetch_user_extractions(username, num_of_records=60)

if "status_check_rows" not in st.session_state:
    st.session_state.status_check_rows = []
if "status_check_all_ready" not in st.session_state:
    st.session_state.status_check_all_ready = False
if "status_check_ids" not in st.session_state:
    st.session_state.status_check_ids = []
if "status_check_ts" not in st.session_state:
    st.session_state.status_check_ts = None
if "status_check_counts" not in st.session_state:
    st.session_state.status_check_counts = {}

if "custom_validation_id" not in st.session_state:
    st.session_state.custom_validation_id = ""

# =========================
# Step 1) Validation ID
# =========================
st.subheader("1) Define Validation ID")

custom_validation_id = st.text_input(
    "Validation ID",
    value=st.session_state.custom_validation_id,
    placeholder="Enter a unique validation ID (e.g. val_20260222_001)",
)

st.session_state.custom_validation_id = (custom_validation_id or "").strip()

# =========================
# Step 2) Choose Rule
# =========================
st.subheader("2) Choose Rule")

rule_dropdown = ["Select a Rule"] + [
    f"{r.get('ruleId', '')} (Modified: {r.get('modifiedAtHKTime', 'N/A')})"
    for r in st.session_state.validation_rules
    if r.get("ruleId")
]
selected_rule_option = st.selectbox("Rule", rule_dropdown, label_visibility="collapsed")

selected_rule_id = None
if selected_rule_option != "Select a Rule":
    selected_rule_id = selected_rule_option.split(" (Modified: ")[0]

if selected_rule_id:
    with st.expander("Preview Rule Sets", expanded=False):
        rule_doc = fetch_rule_by_id(selected_rule_id)
        rule_sets = rule_doc.get("ruleSets", [])
        if isinstance(rule_sets, list) and rule_sets:
            df = pd.DataFrame(rule_sets).rename(columns={"alias": "Alias", "rule": "Rule"})
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.write("No ruleSets found for this rule.")

# =========================
# Step 3) Choose Extractions
# =========================
st.subheader(f"3) Choose Extractions (max {MAX_EXTRACTION_IDS})")

top_col1, top_col2 = st.columns([1, 2], vertical_alignment="bottom")
with top_col1:
    extraction_records_to_fetch = st.selectbox(
        "Records to fetch",
        options=[10, 20, 30, 40, 50],
        index=2,
        key="extraction_records_to_fetch",
    )

only_show_succeeded = True

with top_col2:
    if st.button("Refresh", use_container_width=True):
        st.session_state.validation_rules = fetch_rules()
        st.session_state.user_extractions = fetch_user_extractions(username, extraction_records_to_fetch)
        st.success("Refresh done.")
        st.rerun()

extractions = st.session_state.user_extractions or []
filtered_extractions = (
    [e for e in extractions if normalize_status(e.get("status")) == "succeeded"]
    if only_show_succeeded
    else extractions
)

selected_extraction_ids_from_list: List[str] = []
if filtered_extractions:
    extraction_options: List[str] = []
    for e in filtered_extractions:
        extraction_id = e.get("extractionId", "")
        status = e.get("status", "unknown")
        cfg = e.get("configId", "N/A")
        modified = e.get("modifiedAtHKTime", "N/A")
        extraction_options.append(f"{extraction_id} | {status} | {cfg} | {modified}")

    selected_extraction_labels = st.multiselect(
        "Select from recent extractions",
        options=extraction_options,
        default=[],
        max_selections=MAX_EXTRACTION_IDS,
    )
    selected_extraction_ids_from_list = [lbl.split(" | ")[0].strip() for lbl in selected_extraction_labels if lbl.strip()]
else:
    st.info("No extractions available for the current filter/window.")

manual_extraction_ids_raw = st.text_area(
    "Or paste Extraction IDs (optional, commas/spaces/newlines supported)",
    value="",
    height=110,
)
manual_extraction_ids = parse_extraction_ids(manual_extraction_ids_raw)

combined_ids = combined_unique_ids(selected_extraction_ids_from_list, manual_extraction_ids)

if len(combined_ids) > MAX_EXTRACTION_IDS:
    st.error(f"Max {MAX_EXTRACTION_IDS} extraction IDs allowed per validation. Currently selected: {len(combined_ids)}")

# =========================
# Step 4) External Data
# =========================
st.subheader("4) External Data (optional)")
external_data = st.text_area(
    "External Data (plain text; curly braces will be escaped)",
    value="",
    height=160,
)

st.markdown("---")

check_clicked = st.button("Check Status", width="stretch")
if check_clicked:
    if not combined_ids:
        st.error("Select or paste at least one Extraction ID to check status.")
    elif len(combined_ids) > MAX_EXTRACTION_IDS:
        st.error(f"Max {MAX_EXTRACTION_IDS} extraction IDs allowed.")
    else:
        with st.spinner("Checking extraction statuses..."):
            rows, all_ready, counts = check_extractions_ready(combined_ids)
            st.session_state.status_check_rows = rows
            st.session_state.status_check_all_ready = all_ready
            st.session_state.status_check_ids = combined_ids
            st.session_state.status_check_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.status_check_counts = counts

with st.expander("Status check results", expanded=True if st.session_state.status_check_rows else False):
    if st.session_state.status_check_rows:
        st.write(f"Last check: {st.session_state.status_check_ts}")

        counts = st.session_state.status_check_counts or {}
        summary = " | ".join([f"{STATUS_EMOJI.get(k, k)}: {v}" for k, v in counts.items()])
        if summary:
            st.info(summary)

        st.dataframe(
            pd.DataFrame(st.session_state.status_check_rows)[
                ["extractionId", "statusLabel", "readyToValidate", "configId", "model", "modifiedAtHKTime"]
            ],
            width="stretch",
            hide_index=True,
        )

        if st.session_state.status_check_all_ready:
            st.success("All selected extraction IDs are ready (succeeded).")
        else:
            st.warning("Some selected extraction IDs are not ready. Validation submission will be blocked.")
    else:
        st.write("Click 'Check Status' to validate readiness (status must be 'succeeded').")

btn_col1, btn_col2 = st.columns([2, 1])
with btn_col1:
    validate_clicked = st.button("Queue Validation", width="stretch")
with btn_col2:
    if st.button("Reset", width="stretch"):
        st.session_state.status_check_rows = []
        st.session_state.status_check_all_ready = False
        st.session_state.status_check_ids = []
        st.session_state.status_check_ts = None
        st.session_state.status_check_counts = {}
        st.session_state.custom_validation_id = ""
        st.rerun()

if validate_clicked:
    if not st.session_state.custom_validation_id:
        st.error("Please define a Validation ID.")
    elif not selected_rule_id:
        st.error("Please select a Rule.")
    elif not combined_ids:
        st.error("Please select or paste at least one Extraction ID.")
    elif len(combined_ids) > MAX_EXTRACTION_IDS:
        st.error(f"Max {MAX_EXTRACTION_IDS} extraction IDs allowed per validation.")
    else:
        with st.spinner("Re-checking extraction statuses before submission..."):
            rows, all_ready, counts = check_extractions_ready(combined_ids)
            st.session_state.status_check_rows = rows
            st.session_state.status_check_all_ready = all_ready
            st.session_state.status_check_ids = combined_ids
            st.session_state.status_check_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.status_check_counts = counts

        if not all_ready:
            st.error("Validation blocked: all extraction IDs must be ready (status == succeeded).")
            st.dataframe(
                pd.DataFrame(rows)[["extractionId", "statusLabel", "readyToValidate"]],
                width="stretch",
                hide_index=True,
            )
        else:
            cleaned_external = external_data.strip()
            cleaned_external = escape_curly_braces(cleaned_external) if cleaned_external else None

            st.info("Triggering validation request...")
            result = queue_validation(
                validation_id=st.session_state.custom_validation_id,
                extraction_ids=combined_ids,
                rule_id=selected_rule_id,
                external_data=cleaned_external,
            )

            if result and result.get("status") == "queued":
                returned_id = result.get("validation_id")
                shown_id = returned_id or st.session_state.custom_validation_id
                st.success(f"Validation queued. validation_id: {shown_id}")
                st.info("Check progress later in the validation results page.")
            else:
                st.error("Failed to queue validation task.")

st.markdown("<div style='margin-bottom: 10rem;'></div>", unsafe_allow_html=True)
render_footer()
