import streamlit as st
import pandas as pd
import math
import time

from shared_utils import (
    setup_page, render_footer, require_login, STATUS_EMOJI,
    normalize_status, prepare_json_download,
    fetch_user_validations, fetch_validation_result,
)

setup_page()
require_login("validation monitoring")

username = st.session_state["username"]


def reload_validations() -> None:
    """Re-fetch validation records from the backend and reset pagination."""
    n = int(st.session_state.get("num_of_records", 30))
    with st.spinner("Fetching latest validations..."):
        st.session_state.validation_records = fetch_user_validations(username, n)
    st.session_state.validation_page_number = 0


def reset_page_state() -> None:
    """Clear all cached validation data and pagination from session state."""
    keys_to_clear = [
        k for k in list(st.session_state.keys())
        if str(k).startswith(("validation_doc_", "validation_json_"))
    ]
    for k in keys_to_clear:
        del st.session_state[k]

    if "validation_records" in st.session_state:
        del st.session_state["validation_records"]
    if "validation_page_number" in st.session_state:
        del st.session_state["validation_page_number"]


# =========================
# UI
# =========================
st.title("Validation Results Monitor 📈")
st.write("- Monitor validation status and open/download results when available.")

with st.expander("ℹ️ Instructions", expanded=True):
    st.markdown(
        """
- Status: 🟢 succeeded / 🔴 failed / 🔄 processing / ⏳ queued
- This page shows the most recent N validations for the logged-in account.
- JSON download is supported.
"""
    )

# Controls
ctrl1, ctrl2, ctrl3 = st.columns([1, 1, 1], vertical_alignment="bottom")
with ctrl1:
    if "num_of_records" not in st.session_state:
        st.session_state.num_of_records = 30

    st.selectbox(
        "Records to fetch",
        options=[10, 20, 30, 40, 50],
        index=[10, 20, 30, 40, 50].index(int(st.session_state.num_of_records)),
        key="num_of_records",
    )

with ctrl2:
    if st.button("🔄 Refresh", use_container_width=True):
        reload_validations()
        st.rerun()

with ctrl3:
    if st.button("Reset Page State", use_container_width=True):
        reset_page_state()
        st.rerun()

st.markdown("---")

# Load records
if "validation_records" not in st.session_state:
    reload_validations()

records = st.session_state.get("validation_records", [])
if not records:
    st.info("No validations found yet.")
    st.stop()

# Sort records (latest first)
records.sort(key=lambda x: x.get("modifiedAtIsoTime", "") or "", reverse=True)

# Pagination (with clamping)
ITEMS_PER_PAGE = 10
if "validation_page_number" not in st.session_state:
    st.session_state.validation_page_number = 0

max_page = max(0, math.ceil(len(records) / ITEMS_PER_PAGE) - 1)
if st.session_state.validation_page_number > max_page:
    st.session_state.validation_page_number = 0

start_idx = st.session_state.validation_page_number * ITEMS_PER_PAGE
end_idx = start_idx + ITEMS_PER_PAGE
page_records = records[start_idx:end_idx]

if not page_records:
    st.info("No records on this page. Resetting pagination.")
    st.session_state.validation_page_number = 0
    st.rerun()

# Render records
with st.spinner("Rendering validation records..."):
    for rec in page_records:
        validation_id = rec.get("validationId", "") or rec.get("id", "")
        status = normalize_status(rec.get("status"))
        status_text = STATUS_EMOJI.get(status, STATUS_EMOJI["unknown"])

        st.markdown(f"##### Validation ID: `{validation_id}` - Status: {status_text}")

        rule_id = rec.get("ruleId", "N/A")

        time_elapsed = rec.get("timeLapsed", None)
        if time_elapsed is not None:
            try:
                time_elapsed = f"{math.ceil(float(time_elapsed))}s"
            except (ValueError, TypeError):
                time_elapsed = "N/A"
        else:
            time_elapsed = "N/A"

        modified_time = rec.get("modifiedAtHKTime", "N/A")

        col1, col2, col3 = st.columns([2, 1, 1])
        col1.write(f"**Rule ID:** {rule_id}")
        col2.write(f"**Time Elapsed:** {time_elapsed}")
        col3.write(f"**Last Modified:** {modified_time}")

        with st.expander("📂 View / Download JSON", expanded=False):
            state_doc_key = f"validation_doc_{validation_id}"
            state_json_key = f"validation_json_{validation_id}"

            if state_doc_key not in st.session_state:
                st.session_state[state_doc_key] = None
            if state_json_key not in st.session_state:
                st.session_state[state_json_key] = None

            if st.session_state[state_doc_key] is None:
                if st.button(f"Load Details for {validation_id}", key=f"load_{validation_id}", width="content"):
                    with st.spinner("Fetching validation result..."):
                        doc = fetch_validation_result(validation_id)
                        st.session_state[state_doc_key] = doc
                        st.session_state[state_json_key] = prepare_json_download(doc) if doc else None
                        time.sleep(0.2)
                    st.rerun()
            else:
                doc = st.session_state[state_doc_key] or {}

                extraction_ids = doc.get("extractionIds", [])
                extraction_count = len(extraction_ids) if isinstance(extraction_ids, list) else 0
                st.write(f"**Extraction IDs:** {extraction_count}")

                token_usage = doc.get("tokenUsage") or {}
                if isinstance(token_usage, dict):
                    st.write(
                        f"**Token Usage:** total={token_usage.get('total_tokens', 0)}, "
                        f"prompt={token_usage.get('prompt_tokens', 0)}, "
                        f"completion={token_usage.get('completion_tokens', 0)}"
                    )

                validation_result = doc.get("validationResult")
                if isinstance(validation_result, dict) and validation_result:
                    rows = []
                    for k, v in validation_result.items():
                        if not isinstance(v, dict):
                            continue
                        result_obj = v.get("result") if isinstance(v.get("result"), dict) else {}
                        rows.append(
                            {
                                "rule_key": k,
                                "alias": v.get("alias"),
                                "score": result_obj.get("score"),
                                "reason": result_obj.get("reason"),
                            }
                        )

                    if rows:
                        st.markdown("**Validation Result Summary**")
                        df = pd.DataFrame(rows).sort_values(by="score", ascending=True, na_position="last")
                        st.dataframe(df, width="stretch", hide_index=True)

                    with st.expander("Evidence", expanded=False):
                        for k, v in validation_result.items():
                            if not isinstance(v, dict):
                                continue
                            result_obj = v.get("result") if isinstance(v.get("result"), dict) else {}
                            alias = v.get("alias", k)
                            evidence_md = result_obj.get("evidence_markdown") or ""
                            st.markdown(f"### {alias}")
                            if evidence_md:
                                st.markdown(evidence_md)
                            else:
                                st.write("(no evidence)")

                err = doc.get("error")
                if err:
                    st.error(f"Error: {err}")

                st.download_button(
                    label="📝 Download JSON",
                    data=st.session_state[state_json_key] or prepare_json_download(doc),
                    file_name=f"validation_{validation_id}.json",
                    mime="application/json",
                    width="stretch",
                )

        st.markdown("---")

# Pagination controls
_, prev_col, mid_col, next_col, _ = st.columns([1, 1, 1, 1, 1])
with prev_col:
    if st.session_state.validation_page_number > 0:
        if st.button("⏪ Previous", width="stretch"):
            st.session_state.validation_page_number -= 1
            st.rerun()

with mid_col:
    st.markdown(f"<div style='text-align:center'>Page {st.session_state.validation_page_number + 1} / {max_page + 1}</div>", unsafe_allow_html=True)

with next_col:
    if end_idx < len(records):
        if st.button("Next ⏩", width="stretch"):
            st.session_state.validation_page_number += 1
            st.rerun()

render_footer()
