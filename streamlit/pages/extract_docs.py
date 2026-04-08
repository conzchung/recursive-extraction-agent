import streamlit as st
import requests
import time
import pandas as pd
from typing import Dict
from requests.exceptions import RequestException

from shared_utils import (
    setup_page, render_footer, require_login, API_BASE_URL, HEADERS,
    fetch_configurations, fetch_config_by_id,
)

setup_page()
require_login("the extraction tool")


def upload_file(file) -> Dict:
    """Upload a document file to Azure Blob Storage via the backend API.

    Args:
        file: A Streamlit UploadedFile object from the file uploader widget.

    Returns:
        Dict: Response data containing 'blob_url' and 'filename', or an
            empty dict on failure.
    """
    try:
        if not file:
            st.error("No file selected for upload.")
            return {}
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(f"{API_BASE_URL}/extraction/upload_document", files=files, headers=HEADERS)
        if response.status_code == 200:
            return response.json().get("data", {})
        else:
            st.error(f"Error uploading file: {response.text}")
            return {}
    except Exception as e:
        st.error(f"Error uploading file: {e}")
        return {}

def extract_document(
    extraction_id: str,
    configuration_id: str,
    file_path: str,
    model: str = "GPT52",
    max_retries: int = 2,
    retry_delay: int = 5
) -> Dict:
    """Queue a document extraction job via the backend API with retry logic.

    Sends the extraction request and retries on network or server errors up to
    max_retries times. The backend runs the actual extraction asynchronously
    as a background task.

    Args:
        extraction_id: A user-defined identifier for this extraction job.
        configuration_id: The ID of the field configuration to use.
        file_path: The blob URL of the uploaded document.
        model: The model tier to use ('basic' or 'advanced').
        max_retries: Maximum number of retry attempts on failure.
        retry_delay: Seconds to wait between retries.

    Returns:
        Dict: The API response body on success, or an empty dict if all
            retries are exhausted.
    """
    payload = {
        "extraction_id": extraction_id,
        "configuration_id": configuration_id,
        "file_path": file_path,
        "model": model,
        "user_id": st.session_state.get("username", "unknown")
    }
    attempt = 0
    while attempt < max_retries:
        try:
            response = requests.post(
                f"{API_BASE_URL}/extraction/extract_document",
                json=payload,
                headers=HEADERS,
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Error during extraction (Attempt {attempt+1}/{max_retries}): {response.text}")
                attempt += 1
                if attempt < max_retries:
                    st.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
        except RequestException as e:
            st.error(f"Network error during extraction (Attempt {attempt+1}/{max_retries}): {e}")
            attempt += 1
            if attempt < max_retries:
                st.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    st.error("Extraction failed after maximum retries.")
    return {}

# Page UI
st.title("Document Extraction 📤")
st.write("- Upload a document and extract data based on a selected configuration.")

# Model guide
with st.expander("**Model Selection Guide 📊**", expanded=True):
    st.markdown("- **Basic**: Fast and cost-effective 💰, ideal for quick initial extractions 🚀.")
    st.markdown("- **Advanced**: Smarter with deep reasoning 🧠, perfect for complex documents 📚.")


# Extraction ID & model selection
col1, col2 = st.columns([4, 1])
with col1:
    extraction_id = st.text_input("Extraction ID", value="", label_visibility='visible')
with col2:
    model_options = ["Basic", "Advanced"]
    selected_model = st.selectbox("Select Model", model_options, index=1, label_visibility='visible')
    model_value = "basic" if selected_model == "Basic" else "advanced"

# Load configurations
if "configs" not in st.session_state:
    st.session_state.configs = fetch_configurations()

# Configuration dropdown
col1, col2 = st.columns([3, 1])
with col1:
    dropdown_options = ["Select a Configuration"] + [
        f"{config['configId']} (Modified: {config.get('modifiedAtHKTime', 'N/A')})"
        for config in st.session_state.configs
    ]
    selected_option = st.selectbox("Select Configuration", dropdown_options, label_visibility='collapsed')
with col2:
    if st.button("Refresh Configurations", width='stretch'):
        st.session_state.configs = fetch_configurations()
        st.success("Refresh Done!")
        st.rerun()

configuration_id = None
if selected_option != "Select a Configuration":
    configuration_id = selected_option.split(" (Modified: ")[0]
    config_data = fetch_config_by_id(configuration_id)
    if config_data:
        with st.expander(f"👀 Preview Fields for {configuration_id}", expanded=False):
            fields = config_data.get("fieldsToExtract", [])
            if fields:
                fields_df = pd.DataFrame(fields).rename(columns={
                    "columnName": "Column Name",
                    "dataType": "Data Type",
                    "remarks": "Remarks"
                })
                st.dataframe(fields_df, width='stretch', hide_index=True)
            else:
                st.write("No fields defined in this configuration.")

# Session state for blob_url
if "blob_url" not in st.session_state:
    st.session_state.blob_url = None

# File upload
uploaded_file = st.file_uploader("Upload Document", type=["pdf", "png", "jpg", "jpeg", "xlsx"], key="file_uploader")
if uploaded_file is not None:
    with st.spinner("Uploading file..."):
        upload_result = upload_file(uploaded_file)
        if upload_result:
            st.session_state.blob_url = upload_result.get("blob_url")
            st.success(f"File {uploaded_file.name} uploaded successfully!")
        else:
            st.error("Failed to upload file.")

# Extract & reset buttons
col1, col2, col3 = st.columns([2, 3, 1])
with col1:
    extract_clicked = st.button("Extract Data", width='stretch')
with col3:
    if st.button("Reset", width='stretch'):
        st.session_state.blob_url = None
        st.rerun()

# Handle extraction click without polling
if extract_clicked:
    if not extraction_id:
        st.error("Please enter an Extraction ID.")
    elif not configuration_id:
        st.error("Please select a Configuration.")
    elif not uploaded_file or not st.session_state.blob_url:
        st.error("Please upload a document.")
    else:
        st.info("Triggering extraction request...")
        result = extract_document(
            extraction_id=extraction_id,
            configuration_id=configuration_id,
            file_path=st.session_state.blob_url,
            model=model_value
        )
        if result and result.get("status") == "queued":
            st.success(f"✅ Extraction queued for ID: {extraction_id}")
            st.info("You can check your extraction status later in the 'Monitor Progress' page. Click Reset to start another extraction.")
        else:
            st.error("Failed to queue extraction task.")

st.markdown("<div style='margin-bottom: 10rem;'></div>", unsafe_allow_html=True)
render_footer()
