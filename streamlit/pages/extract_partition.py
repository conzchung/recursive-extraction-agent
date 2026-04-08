import streamlit as st
import requests
from typing import Dict
from requests.exceptions import RequestException

from shared_utils import setup_page, render_footer, API_BASE_URL, HEADERS

setup_page()


def upload_file(file) -> Dict:
    """Upload a file to the backend /upload_document endpoint.

    Args:
        file: A Streamlit UploadedFile object from the file uploader widget.

    Returns:
        Dict: Response data containing 'blob_url' and 'filename', or an
            empty dict on error.
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


def partition_document(file_path: str, model: str = "GPT5m") -> Dict:
    """Trigger document partitioning via the backend API.

    Sends the uploaded file's blob URL to the /partition_document endpoint,
    which splits a combined PDF into separate documents and returns a ZIP.

    Args:
        file_path: The blob URL of the previously uploaded file.
        model: The model tier to use for partitioning ('basic' or 'advanced').

    Returns:
        Dict: Response data containing 'zip_blob_url', or an empty dict
            on error.
    """
    try:
        if not file_path:
            st.error("No file path provided for partitioning.")
            return {}
        payload = {"file_path": file_path, "model": model}
        response = requests.post(f"{API_BASE_URL}/extraction/partition_document", json=payload, headers=HEADERS)
        if response.status_code == 200:
            return response.json().get("data", {})
        else:
            st.error(f"Error partitioning document: {response.text}")
            return {}
    except RequestException as e:
        st.error(f"Error partitioning document: {e}")
        return {}


def download_zip(zip_url: str):
    """Fetch a ZIP file from a URL and return its raw content.

    Used to download the partitioned-document ZIP from Azure Blob Storage
    so it can be offered as a Streamlit download button.

    Args:
        zip_url: The full URL of the ZIP file to download.

    Returns:
        bytes or None: The raw ZIP file content, or None on failure.
    """
    try:
        zip_response = requests.get(zip_url)
        if zip_response.status_code == 200:
            return zip_response.content
        else:
            st.error(f"Failed to fetch ZIP file: {zip_response.text}")
            return None
    except Exception as e:
        st.error(f"Error downloading ZIP file: {e}")
        return None


# Streamlit app for Document Upload and Partitioning
st.title("Document Partitioning 📂")
st.write("- Upload a document and partition it into multiple files based on its type.")

# Add descriptive text for model selection as an expander
with st.expander("**Model Selection Guide 📊**", expanded=True):
    st.markdown("- **Basic (GPT-5.4 mini)**: Fast and cost-effective 💰, ideal for quick initial extractions 🚀.")
    st.markdown("- **Intermediate (GPT-5.1)**: Balanced performance ⚖️, suitable for moderately complex documents 📄.")
    st.markdown("- **Advanced (GPT-5.2)**: Smarter with deep reasoning 🧠, perfect for complex documents 📚.")

# File upload section
st.subheader("Step 1: Upload Document 📤")

col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader("Choose a document to upload (PDF, XLSX, etc.)", type=["pdf", "xlsx", "xls"])
with col2:
    model_options = ["Basic (GPT-5.4 mini)", "Advanced (GPT-5.4)"]
    selected_model = st.selectbox("Select Model", model_options, index=1, label_visibility='visible')
    model_value = "basic" if selected_model == "Basic (GPT-5.4 mini)" else "advanced"

# State to store upload and partition results
if "upload_result" not in st.session_state:
    st.session_state.upload_result = {}
if "partition_result" not in st.session_state:
    st.session_state.partition_result = {}
if "zip_content" not in st.session_state:
    st.session_state.zip_content = None

# Button to trigger upload
if uploaded_file is not None:
    if st.button("Upload Document", width='stretch'):
        with st.spinner("Uploading document..."):
            st.session_state.upload_result = upload_file(uploaded_file)
            if st.session_state.upload_result:
                st.success(f"Document uploaded successfully!")
            else:
                st.error("Upload failed. Please try again.")

# Partition section (only shown if upload is successful)
if st.session_state.upload_result and "blob_url" in st.session_state.upload_result:
    st.markdown("---")
    st.subheader("Step 2: Partition Document ✂️")
    st.write(f"Uploaded File: {st.session_state.upload_result.get('filename', 'Unknown')}")

    if st.button("Partition Document", width='stretch'):
        with st.spinner("Partitioning document..."):
            file_path = st.session_state.upload_result.get("blob_url")
            st.session_state.partition_result = partition_document(file_path, model=model_value)
            if st.session_state.partition_result:
                zip_url = st.session_state.partition_result.get("zip_blob_url", "N/A")
                st.success(f"Document partitioned successfully!")
            else:
                st.error("Partitioning failed. Please try again.")

# Download and Reset Buttons (only shown if partitioning is successful)
if st.session_state.partition_result and "zip_blob_url" in st.session_state.partition_result:
    st.markdown("---")
    zip_url = st.session_state.partition_result.get("zip_blob_url", "")
    if zip_url:
        st.subheader("Step 3: Download Partitioned Document 💾")

        # Automatically fetch ZIP content if not already downloaded
        if not st.session_state.get("zip_content"):
            st.session_state.zip_content = download_zip(zip_url)

        # Create three columns for layout
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            if st.session_state.get("zip_content"):
                st.download_button(
                    label="Download Partitioned ZIP",
                    data=st.session_state.zip_content,
                    file_name="partitioned_documents.zip",
                    mime="application/zip",
                    key="download_zip_button",
                    width='stretch'
                )
            else:
                st.error("Failed to fetch ZIP file. Please try again.")
                if st.button("Retry Fetching ZIP", width='stretch'):
                    st.session_state.zip_content = download_zip(zip_url)
                    st.rerun()

        with col3:
            if st.button("Reset", width='stretch'):
                # Clear session state to reset the process
                st.session_state.upload_result = {}
                st.session_state.partition_result = {}
                st.session_state.zip_content = None
                st.rerun()

        st.markdown("---")

# Display instructions if no upload yet
if not st.session_state.upload_result:
    st.info("Please upload a document to begin the partitioning process.")
elif not st.session_state.partition_result:
    st.info("Click 'Partition Document' to split the uploaded file into partitions.")

render_footer()
