import streamlit as st
import requests
import time
import json
import uuid
import copy
from typing import List, Dict
from shared_utils import (
    setup_page, render_footer, API_BASE_URL, HEADERS,
    fetch_configurations, fetch_config_by_id,
)

setup_page()


def save_configuration(config_id: str, fields_to_extract: List[Dict]) -> Dict:
    """Save or update an extraction configuration via the backend API.

    Sends a PUT request with the field definitions. The backend handles
    both creation and update logic.

    Args:
        config_id: The configuration identifier to create or update.
        fields_to_extract: A list of field dicts, each containing
            'columnName', 'dataType', and 'remarks'.

    Returns:
        Dict: The API response body on success, or an empty dict on failure.
    """
    payload = {"fieldsToExtract": fields_to_extract}
    try:
        response = requests.put(
            f"{API_BASE_URL}/extraction/update_configuration/{config_id}",
            json=payload,
            headers=HEADERS,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error saving configuration: {response.text}")
            return {}
    except Exception as e:
        st.error(f"Error saving configuration: {e}")
        return {}


# --- Session state initialization ---

# Cached configurations list
if "configs" not in st.session_state:
    st.session_state.configs = fetch_configurations()

# ID of configuration currently loaded in the editor
if "active_config_id" not in st.session_state:
    st.session_state.active_config_id = None

# Current editable fields
if "fields" not in st.session_state:
    st.session_state.fields = [
        {"columnName": "", "dataType": "Text", "remarks": ""}
    ]

# Snapshot of last loaded/saved fields for unsaved-change detection
if "fields_snapshot" not in st.session_state:
    st.session_state.fields_snapshot = copy.deepcopy(st.session_state.fields)

# Version token used in widget keys to force reset when switching configs
if "fields_version" not in st.session_state:
    st.session_state.fields_version = "default"


# --- UI: Title ---

st.title("🛠️ Extraction Configuration 🛠️")
st.write("- Create or update a configuration schema for document extraction.")


# --- Config selection & loading with unsaved-change warning ---

col1, col2 = st.columns([3, 1])
with col1:
    dropdown_options = ["Create New..."] + [
        f"{config['configId']} (Modified: {config.get('modifiedAtHKTime', 'N/A')})"
        for config in st.session_state.configs
    ]
    selected_option = st.selectbox(
        "Select Existing Configuration",
        dropdown_options,
        label_visibility="collapsed",
    )

with col2:
    if st.button("Refresh Configurations", use_container_width=True):
        st.session_state.configs = fetch_configurations()
        st.success("Refresh Done!")
        st.rerun()

# Convert selection to config ID or None
if selected_option != "Create New...":
    selected_config_id = selected_option.split(" (Modified: ")[0]
else:
    selected_config_id = None

# Detect unsaved changes on currently active config
unsaved_changes = (
    st.session_state.fields != st.session_state.fields_snapshot
)

# If selection changed, possibly discard unsaved changes and load new config
if selected_config_id != st.session_state.active_config_id:
    if unsaved_changes and st.session_state.active_config_id is not None:
        st.warning(
            f"Switched configuration from '{st.session_state.active_config_id}' "
            f"with unsaved changes. Previous edits were discarded."
        )

    st.session_state.active_config_id = selected_config_id

    if selected_config_id is None:
        # New configuration: reset fields
        st.session_state.fields = [
            {"columnName": "", "dataType": "Text", "remarks": ""}
        ]
        st.session_state.fields_version = f"new_{time.time()}"
    else:
        # Existing configuration: fetch from backend once
        config_data = fetch_config_by_id(selected_config_id)
        if config_data:
            st.session_state.fields = config_data.get("fieldsToExtract", [])
            st.session_state.fields_version = (
                f"v_{selected_config_id}_{time.time()}"
            )

    # Update snapshot after loading/resetting
    st.session_state.fields_snapshot = copy.deepcopy(
        st.session_state.fields
    )

# Config ID input
if st.session_state.active_config_id is None:
    # New config: editable ID
    config_id = st.text_input("Configuration ID", value="")
else:
    # Existing config: fixed ID
    config_id = st.text_input(
        "Configuration ID",
        value=st.session_state.active_config_id,
        disabled=True,
    )


# --- Fields to Extract - Dynamic Rows ---

st.subheader("📋 Fields to Extract")
version = st.session_state.get("fields_version", "default")

for i, field in enumerate(st.session_state.fields):
    with st.container():
        col1, col2, col3, col4 = st.columns([2, 1, 3, 1])
        with col1:
            field["columnName"] = st.text_input(
                "Column Name",
                value=field.get("columnName", ""),
                key=f"col_name_{version}_{i}",
            )
        with col2:
            current_type = field.get("dataType", "Text")
            field["dataType"] = st.selectbox(
                "Data Type",
                ["Text", "Table"],
                index=0 if current_type == "Text" else 1,
                key=f"data_type_{version}_{i}",
            )
        with col3:
            field["remarks"] = st.text_area(
                "Remarks",
                value=field.get("remarks", ""),
                height=50,
                key=f"remarks_{version}_{i}",
            )
        with col4:
            if st.button("Remove", key=f"remove_{version}_{i}"):
                st.session_state.fields.pop(i)
                st.rerun()
        st.markdown("---")

# Button to add new field
if st.button("Add New Field", use_container_width=True):
    st.session_state.fields.append(
        {"columnName": "", "dataType": "Text", "remarks": ""}
    )
    st.rerun()

st.markdown("---")


# --- Save button with change detection for updates ---

if st.button("Save Configuration", use_container_width=True):
    if not config_id:
        st.error("Please enter a Configuration ID.")
    elif not st.session_state.fields or any(
        not f["columnName"] for f in st.session_state.fields
    ):
        st.error("Please fill in all Column Names.")
    else:
        # Updating existing config
        if st.session_state.active_config_id is not None:
            existing_config = fetch_config_by_id(
                st.session_state.active_config_id
            )
            existing_fields = existing_config.get("fieldsToExtract", [])
            current_fields = st.session_state.fields

            if existing_fields == current_fields:
                st.warning(
                    "No changes detected. Configuration not updated."
                )
            else:
                result = save_configuration(
                    config_id, st.session_state.fields
                )
                if result.get("status") == "success":
                    st.success(
                        f"Configuration '{config_id}' "
                        f"{result.get('action', 'saved')} successfully!"
                    )
                    st.session_state.configs = fetch_configurations()
                    st.session_state.fields_snapshot = copy.deepcopy(
                        st.session_state.fields
                    )
                else:
                    st.error("Failed to save configuration.")
        else:
            # New configuration; backend handles overwrite if same ID exists
            result = save_configuration(config_id, st.session_state.fields)
            if result.get("status") == "success":
                st.success(
                    f"Configuration '{config_id}' "
                    f"{result.get('action', 'saved')} successfully!"
                )
                st.session_state.configs = fetch_configurations()
                st.session_state.active_config_id = config_id
                st.session_state.fields_snapshot = copy.deepcopy(
                    st.session_state.fields
                )
            else:
                st.error("Failed to save configuration.")


render_footer()
