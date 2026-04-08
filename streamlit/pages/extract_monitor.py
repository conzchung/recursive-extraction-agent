import streamlit as st
import math

from shared_utils import (
    setup_page, render_footer, require_login, STATUS_EMOJI,
    fetch_user_extractions, fetch_extraction_result,
    prepare_json_download, prepare_excel_download, prepare_csv_download,
)

setup_page()
require_login("extraction monitoring")

username = st.session_state["username"]

# =========================
# UI
# =========================
st.title("Extraction Results Monitor 📊")
st.write("- View and track the status of your extractions, including results when available.")

# Instructions expander
with st.expander("ℹ️ Instructions", expanded=True):
    st.markdown("""
    1. **Status Mapping**
       - 🟢 **Succeeded**: Extraction completed successfully and results are ready.
       - 🔴 **Failed**: Extraction did not complete due to an error.
       - 🔄 **Processing**: Extraction is currently running.
       - ⏳ **Queued**: Extraction is waiting to start.

    2. **Record Limiting**
       - Use the **Records to fetch** dropdown to choose how many recent extractions to load (10–50).

    3. **Refreshing Data**
       - Click **🔄 Refresh** at the top to fetch the latest extraction statuses.

    4. **Downloading Results**
       - click **Prepare Download**. You'll see three download buttons:
         📦 CSV (ZIP) — 📝 JSON — 📊 Excel
    """)

# Records-per-fetch selector + refresh
col_limit, col_refresh = st.columns([1, 3], vertical_alignment="bottom")
with col_limit:
    record_limit = st.selectbox(
        "Records to fetch",
        options=[10, 20, 30, 40, 50],
        index=2,
        key="record_limit",
    )
with col_refresh:
    refresh_clicked = st.button("🔄 Refresh", use_container_width=True)

# Load records into session_state once
if "records" not in st.session_state:
    with st.spinner("Loading extractions..."):
        st.session_state.records = fetch_user_extractions(username, record_limit)

if refresh_clicked:
    with st.spinner("Fetching latest extractions..."):
        st.session_state.records = fetch_user_extractions(username, record_limit)
    st.rerun()

st.markdown("---")

records = st.session_state.records
if not records:
    st.info("No extractions found yet.")
    st.stop()

# Sort records
records.sort(key=lambda x: x.get("modifiedAtIsoTime", ""), reverse=True)

# Pagination setup
ITEMS_PER_PAGE = 10
if "page_number" not in st.session_state:
    st.session_state.page_number = 0

start_idx = st.session_state.page_number * ITEMS_PER_PAGE
end_idx = start_idx + ITEMS_PER_PAGE
page_records = records[start_idx:end_idx]

with st.spinner("Rendering extraction records..."):
    for rec in page_records:
        status = rec.get("status", "unknown")
        status_text = STATUS_EMOJI.get(status, f"❓ {status.capitalize()}")
        st.markdown(f"##### Extraction ID: `{rec.get('extractionId', '')}` - Status: {status_text}")

        config_id = rec.get('configId', 'N/A')
        model = rec.get('model', 'N/A')

        time_elapsed = rec.get('timeLapsed', None)
        if time_elapsed is not None:
            try:
                time_elapsed = f"{math.ceil(float(time_elapsed))}s"
            except (ValueError, TypeError):
                time_elapsed = "N/A"
        else:
            time_elapsed = "N/A"

        modified_time = rec.get('modifiedAtHKTime', 'N/A')

        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        col1.write(f"**Configuration ID:** {config_id}")
        col2.write(f"**Model:** {model}")
        col3.write(f"**Time Elapsed:** {time_elapsed}")
        col4.write(f"**Last Modified:** {modified_time}")

        if status.lower() == "succeeded":
            with st.expander("📂 Download Results"):
                extraction_id = rec['extractionId']
                state_key = f"download_ready_{extraction_id}"

                if state_key not in st.session_state:
                    st.session_state[state_key] = False

                if not st.session_state[state_key]:
                    if st.button(f"Prepare Download for {extraction_id}", key=f"btn_{extraction_id}"):
                        with st.spinner("Fetching and preparing download..."):
                            # Lazy fetch full result from detail endpoint
                            full_result = fetch_extraction_result(extraction_id)

                            if full_result:
                                extraction_result_raw = full_result.get("extractionResult", {})

                                if isinstance(extraction_result_raw.get("extraction_progress"), dict):
                                    extraction_result = extraction_result_raw["extraction_progress"]
                                else:
                                    extraction_result = extraction_result_raw

                                st.session_state[f"csv_data_{extraction_id}"] = prepare_csv_download(extraction_result)
                                st.session_state[f"json_data_{extraction_id}"] = prepare_json_download(extraction_result)
                                st.session_state[f"excel_data_{extraction_id}"] = prepare_excel_download(extraction_result)
                                st.session_state[state_key] = True
                            else:
                                st.error("Failed to fetch extraction result.")

                        if st.session_state[state_key]:
                            st.rerun()
                else:
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.download_button(
                            label="📦 Download as CSV (ZIP)",
                            data=st.session_state[f"csv_data_{extraction_id}"],
                            file_name=f"extraction_{extraction_id}.zip",
                            mime="application/zip",
                            width='stretch'
                        )

                    with col2:
                        st.download_button(
                            label="📝 Download as JSON",
                            data=st.session_state[f"json_data_{extraction_id}"],
                            file_name=f"extraction_{extraction_id}.json",
                            mime="application/json",
                            width='stretch'
                        )

                    with col3:
                        st.download_button(
                            label="📊 Download as Excel",
                            data=st.session_state[f"excel_data_{extraction_id}"],
                            file_name=f"extraction_{extraction_id}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            width='stretch'
                        )

        st.markdown("---")

# Pagination controls
max_page = max(0, math.ceil(len(records) / ITEMS_PER_PAGE) - 1)
_, prev_col, mid_col, next_col, _ = st.columns([1, 1, 1, 1, 1])
with prev_col:
    if st.session_state.page_number > 0:
        if st.button("⏪ Previous", width='stretch'):
            st.session_state.page_number -= 1
            st.rerun()
with mid_col:
    st.markdown(f"<div style='text-align:center'>Page {st.session_state.page_number + 1} / {max_page + 1}</div>", unsafe_allow_html=True)
with next_col:
    if end_idx < len(records):
        if st.button("Next ⏩", width='stretch'):
            st.session_state.page_number += 1
            st.rerun()


render_footer()
