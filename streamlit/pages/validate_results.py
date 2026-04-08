import streamlit as st
import pandas as pd
import math
from typing import Dict, Any, Optional, List

from shared_utils import (
    setup_page, render_footer, require_login, STATUS_EMOJI,
    normalize_status, prepare_json_download, fetch_validation_result,
)

setup_page()
require_login("validation results")


def _safe_str(x: Any) -> str:
    """Convert a value to string, returning empty string for None.

    Args:
        x: Any value to convert.

    Returns:
        str: The string representation, or '' if x is None.
    """
    return "" if x is None else str(x)


def render_validation_result(doc: Dict[str, Any]) -> None:
    """Render the validation result details into the Streamlit page.

    Displays a summary table of all rule scores, followed by expandable
    sections for each rule showing score, reason, rule text, and evidence
    markdown.

    Args:
        doc: The full validation result document containing a
            'validationResult' dict keyed by rule identifiers.
    """
    validation_result = doc.get("validationResult")
    if not isinstance(validation_result, dict) or not validation_result:
        st.info("No validationResult available yet.")
        return

    items: List[Dict[str, Any]] = []
    for rule_key, item in validation_result.items():
        if not isinstance(item, dict):
            continue

        result_obj = item.get("result") if isinstance(item.get("result"), dict) else {}
        items.append(
            {
                "rule_key": rule_key,
                "alias": _safe_str(item.get("alias")) or rule_key,
                "rule": _safe_str(item.get("rule")),
                "score": result_obj.get("score"),
                "reason": _safe_str(result_obj.get("reason")),
                "evidence_markdown": _safe_str(result_obj.get("evidence_markdown")),
            }
        )

    if not items:
        st.info("No rule results available yet.")
        return

    st.subheader("Summary")
    summary_rows: List[Dict[str, Any]] = []
    for i, it in enumerate(items, start=1):
        summary_rows.append(
            {
                "#": i,
                "alias": it["alias"],
                "score(1-10)": it["score"],
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    st.subheader("Details")
    for i, it in enumerate(items, start=1):
        header = f"{i}. {it['alias']}"
        with st.expander(header, expanded=True):
            st.markdown("**Score (1-10)**")
            st.metric(label="", value=it["score"] if it["score"] is not None else "N/A", label_visibility="collapsed")

            st.markdown("**Reason**")
            st.write(it["reason"] if it["reason"] else "(no reason)")

            if it["rule"]:
                with st.expander("Rule text", expanded=False):
                    st.markdown(it["rule"])

            with st.expander("Evidence", expanded=False):
                if it["evidence_markdown"]:
                    st.markdown(it["evidence_markdown"])
                else:
                    st.write("(no evidence)")

            st.markdown("---")


# =========================
# UI
# =========================
st.title("Validation Result Query 📋")
st.write("- Retrieve and view validation results by entering a Validation ID.")

with st.expander("ℹ️ What this page shows", expanded=True):
    st.markdown(
        """
- **Metadata**: status, rule Id, extraction Ids, time elapsed, timestamps
- **ValidationResult**: per rule: alias / rule + score / reason / evidence
- **Download**: JSON only
"""
    )

col1, col2 = st.columns([3, 1])
with col1:
    validation_id = st.text_input("Validation ID", value="", label_visibility="collapsed")
with col2:
    fetch_button = st.button("Fetch Result", width="stretch")

doc: Optional[Dict[str, Any]] = None

if fetch_button:
    if not validation_id.strip():
        st.error("Please enter a Validation ID.")
    else:
        with st.spinner("Fetching validation result..."):
            doc = fetch_validation_result(validation_id.strip())
        st.session_state["last_validation_doc"] = doc

if "last_validation_doc" in st.session_state and st.session_state["last_validation_doc"]:
    doc = st.session_state["last_validation_doc"]

if doc:
    status_norm = normalize_status(doc.get("status"))
    shown_validation_id = doc.get("validationId") or doc.get("id") or validation_id.strip()

    st.markdown(
        f"### Validation ID: `{shown_validation_id}` - Status: {STATUS_EMOJI.get(status_norm, STATUS_EMOJI['unknown'])}"
    )

    st.subheader("Metadata")
    extraction_ids = doc.get("extractionIds", [])
    extraction_count = len(extraction_ids) if isinstance(extraction_ids, list) else "N/A"

    c1, c2, c3, c4 = st.columns([2, 2, 2, 2])
    with c1:
        st.write(f"**Rule ID:** {doc.get('ruleId', 'N/A')}")
    with c2:
        st.write(f"**Extraction IDs:** {extraction_count}")
    with c3:
        time_elapsed = doc.get("timeLapsed", None)
        if time_elapsed is not None:
            try:
                time_elapsed = f"{math.ceil(float(time_elapsed))}s"
            except (ValueError, TypeError):
                time_elapsed = "N/A"
        else:
            time_elapsed = "N/A"
        st.write(f"**Time Elapsed:** {time_elapsed}")
    with c4:
        st.write(f"**Last Modified (HK):** {doc.get('modifiedAtHKTime', 'N/A')}")

    with st.expander("Show Extraction IDs / External Data", expanded=False):
        st.markdown("**Extraction IDs**")
        if isinstance(extraction_ids, list) and extraction_ids:
            st.code("\n".join([str(x) for x in extraction_ids]), language="text")
        else:
            st.write("(none)")

        st.markdown("**External Data**")
        external_data = doc.get("externalData")
        if external_data:
            st.code(str(external_data), language="text")
        else:
            st.write("(none)")

    if doc.get("error"):
        st.subheader("Error")
        st.error(str(doc.get("error")))

    st.subheader("Validation Result")
    render_validation_result(doc)

    st.subheader("Download")
    st.download_button(
        label="📝 Download JSON",
        data=prepare_json_download(doc),
        file_name=f"validation_{shown_validation_id}.json",
        mime="application/json",
        width="stretch",
    )

render_footer()
