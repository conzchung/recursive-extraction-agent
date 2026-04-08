import streamlit as st
import requests
import time
import copy
from typing import List, Dict, Any

from shared_utils import (
    setup_page, render_footer, API_BASE_URL, HEADERS,
    fetch_rules, fetch_rule_by_id,
)

setup_page()


def save_rule(rule_id: str, rule_sets: List[Dict[str, str]]) -> Dict[str, Any]:
    """Save or update a validation rule via the backend API.

    Sends a PUT request with the rule set definitions. The backend handles
    both creation and update logic.

    Args:
        rule_id: The rule identifier to create or update.
        rule_sets: A list of dicts, each containing 'alias' and 'rule' keys.

    Returns:
        Dict[str, Any]: The API response body on success, or an empty dict
            on failure.
    """
    payload = {"ruleSets": rule_sets}
    try:
        r = requests.put(
            f"{API_BASE_URL}/validation/update_rule/{rule_id}",
            json=payload,
            headers=HEADERS,
            timeout=60,
        )
        if r.status_code == 200:
            return r.json()
        st.error(f"Error saving rule: {r.text}")
        return {}
    except Exception as e:
        st.error(f"Error saving rule: {e}")
        return {}


# Session state
if "rules" not in st.session_state:
    st.session_state.rules = fetch_rules()

if "active_rule_id" not in st.session_state:
    st.session_state.active_rule_id = None

if "rule_sets" not in st.session_state:
    st.session_state.rule_sets = [{"alias": "", "rule": ""}]

if "rule_sets_snapshot" not in st.session_state:
    st.session_state.rule_sets_snapshot = copy.deepcopy(st.session_state.rule_sets)

if "rule_sets_version" not in st.session_state:
    st.session_state.rule_sets_version = "default"


st.title("Validation Rule Builder")
st.write("- Create or update validation rules")

col1, col2 = st.columns([3, 1])
with col1:
    dropdown_options = ["Create New..."] + [
        f"{r.get('ruleId', '')} (Modified: {r.get('modifiedAtHKTime', 'N/A')})"
        for r in st.session_state.rules
        if r.get("ruleId")
    ]
    selected_option = st.selectbox(
        "Select Existing Rule",
        dropdown_options,
        label_visibility="collapsed",
    )

with col2:
    if st.button("Refresh Rules", use_container_width=True):
        st.session_state.rules = fetch_rules()
        st.success("Refresh done.")
        st.rerun()

selected_rule_id = None
if selected_option != "Create New...":
    selected_rule_id = selected_option.split(" (Modified: ")[0]

unsaved_changes = st.session_state.rule_sets != st.session_state.rule_sets_snapshot

if selected_rule_id != st.session_state.active_rule_id:
    if unsaved_changes and st.session_state.active_rule_id is not None:
        st.warning(
            f"Switched rule from '{st.session_state.active_rule_id}' with unsaved changes. Edits were discarded."
        )

    st.session_state.active_rule_id = selected_rule_id

    if selected_rule_id is None:
        st.session_state.rule_sets = [{"alias": "", "rule": ""}]
        st.session_state.rule_sets_version = f"new_{time.time()}"
    else:
        rule_doc = fetch_rule_by_id(selected_rule_id)
        rule_sets = rule_doc.get("ruleSets", [])
        st.session_state.rule_sets = rule_sets if isinstance(rule_sets, list) and rule_sets else [{"alias": "", "rule": ""}]
        st.session_state.rule_sets_version = f"v_{selected_rule_id}_{time.time()}"

    st.session_state.rule_sets_snapshot = copy.deepcopy(st.session_state.rule_sets)

if st.session_state.active_rule_id is None:
    rule_id = st.text_input("Rule ID", value="")
else:
    rule_id = st.text_input("Rule ID", value=st.session_state.active_rule_id, disabled=True)

st.subheader("Rule Sets")
version = st.session_state.rule_sets_version

for i, rs in enumerate(st.session_state.rule_sets):
    with st.container():
        c1, c2, c3 = st.columns([2, 6, 1])

        with c1:
            rs["alias"] = st.text_input(
                "Alias",
                value=rs.get("alias", ""),
                key=f"alias_{version}_{i}",
            )

        with c2:
            rs["rule"] = st.text_area(
                "Rule",
                value=rs.get("rule", ""),
                height=90,
                key=f"rule_{version}_{i}",
            )

        with c3:
            if st.button("Remove", key=f"remove_{version}_{i}"):
                st.session_state.rule_sets.pop(i)
                if not st.session_state.rule_sets:
                    st.session_state.rule_sets = [{"alias": "", "rule": ""}]
                st.rerun()

        st.markdown("---")

if st.button("Add Rule Set", use_container_width=True):
    st.session_state.rule_sets.append({"alias": "", "rule": ""})
    st.rerun()

st.markdown("---")

if st.button("Save Rule", use_container_width=True):
    cleaned_rule_id = (rule_id or "").strip()
    if not cleaned_rule_id:
        st.error("Rule ID is required.")
    else:
        cleaned_rule_sets: List[Dict[str, str]] = []
        for rs in st.session_state.rule_sets:
            alias = (rs.get("alias") or "").strip()
            rule_text = (rs.get("rule") or "").strip()
            if not alias or not rule_text:
                st.error("Each rule set must have both Alias and Rule filled.")
                cleaned_rule_sets = []
                break
            cleaned_rule_sets.append({"alias": alias, "rule": rule_text})

        if cleaned_rule_sets:
            existing_rule_sets: List[Dict[str, Any]] = []
            if st.session_state.active_rule_id is not None:
                current_doc = fetch_rule_by_id(st.session_state.active_rule_id)
                existing_rule_sets = current_doc.get("ruleSets", []) if isinstance(current_doc, dict) else []

            if existing_rule_sets == cleaned_rule_sets:
                st.warning("No changes detected. Rule not updated.")
            else:
                result = save_rule(cleaned_rule_id, cleaned_rule_sets)
                if result.get("status") == "success":
                    st.success(f"Rule '{cleaned_rule_id}' {result.get('action', 'saved')} successfully.")
                    st.session_state.rules = fetch_rules()
                    st.session_state.active_rule_id = cleaned_rule_id
                    st.session_state.rule_sets_snapshot = copy.deepcopy(st.session_state.rule_sets)
                else:
                    st.error("Failed to save rule.")

render_footer()
