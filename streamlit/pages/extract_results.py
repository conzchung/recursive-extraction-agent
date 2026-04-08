import streamlit as st
import pandas as pd

from shared_utils import (
    setup_page, render_footer,
    fetch_extraction_result,
    prepare_json_download, prepare_excel_download, prepare_csv_download,
)

setup_page()


# Streamlit app for Page 5
st.title("Extraction Results Query 🔍")
st.write("- Retrieve and view previously extracted data by entering an Extraction ID.")

# Input for Extraction ID and Fetch Button on the same row
col1, col2 = st.columns([3, 1])  # Adjust column widths to give more space to the input field
with col1:
    extraction_id = st.text_input("Extraction ID", value="", label_visibility='collapsed')
with col2:
    fetch_button = st.button("Fetch Results", width='stretch')

# Query Button Logic
if fetch_button:
    if not extraction_id:
        st.error("Please enter an Extraction ID.")
    else:
        with st.spinner("Fetching extraction results...", width='stretch'):
            result = fetch_extraction_result(extraction_id)
            if result:
                # Extract relevant data from result
                # Get the extractionResult dict (or empty dict if missing)
                extraction_result_raw = result.get("extractionResult", {})

                # If extraction_progress exists and is a dict, use it; otherwise use extractionResult itself
                if isinstance(extraction_result_raw.get("extraction_progress"), dict):
                    extraction_result = extraction_result_raw["extraction_progress"]
                else:
                    extraction_result = extraction_result_raw

                time_lapsed = result.get("timeLapsed", 0)
                token_usage = result.get("tokenUsage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
                fields_to_extract = result.get("fieldsToExtract", [])

                # Show success message with timeLapsed
                st.success("Results retrieved successfully!")

                # Display Fields to Extract in a table within an expander
                with st.expander("👀 Preview Fields to Extract", expanded=False):
                    if fields_to_extract:
                        st.write("Fields Used for Extraction")
                        fields_df = pd.DataFrame(fields_to_extract)
                        # Rename columns for better readability
                        fields_df = fields_df.rename(columns={
                            "columnName": "Column Name",
                            "dataType": "Data Type",
                            "remarks": "Remarks"
                        })
                        st.dataframe(fields_df, width='stretch', hide_index=True)
                    else:
                        st.write("No fields to extract available.")

                # Display Extraction Result
                st.subheader("✅ Extraction Results")
                if extraction_result and isinstance(extraction_result, dict):
                    for key, value in extraction_result.items():

                        MAX_PREVIEW_ROWS = 200

                        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                            st.markdown(f"**{key}:**")
                            df = pd.DataFrame(value)
                            if len(df) > MAX_PREVIEW_ROWS:
                                st.warning(f"Showing first {MAX_PREVIEW_ROWS} rows of {len(df)} total.")
                                st.dataframe(df.head(MAX_PREVIEW_ROWS), width='stretch', hide_index=True)
                            else:
                                st.dataframe(df, width='stretch', hide_index=True)
                        else:
                            # Display as key-value pair if not a list of dictionaries
                            st.markdown(f"**{key}:** {value}")
                else:
                    st.write("No extraction results available.")

                st.markdown("---")

                # Display Token Usage in 3 columns using st.metric
                st.subheader("💰 Token Usage")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Prompt Tokens", value=token_usage.get("prompt_tokens", 0))
                with col2:
                    st.metric(label="Completion Tokens", value=token_usage.get("completion_tokens", 0))
                with col3:
                    st.metric(label="Total Tokens", value=token_usage.get("total_tokens", 0))

                st.markdown("---")

                # Provide download options for results
                st.subheader("💾 Download Results")
                col1, col2, col3 = st.columns(3)
                with col1:
                    csv_data = prepare_csv_download(extraction_result)
                    st.download_button(
                        label="Download as CSV (ZIP)",
                        data=csv_data,
                        file_name=f"extraction_{extraction_id}.zip",
                        mime="application/zip",
                        width='stretch'
                    )
                with col2:
                    json_data = prepare_json_download(extraction_result)
                    st.download_button(
                        label="Download as JSON",
                        data=json_data,
                        file_name=f"extraction_{extraction_id}.json",
                        mime="application/json",
                        width='stretch'
                    )
                with col3:
                    excel_data = prepare_excel_download(extraction_result)
                    st.download_button(
                        label="Download as Excel",
                        data=excel_data,
                        file_name=f"extraction_{extraction_id}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        width='stretch'
                    )
            else:
                st.error("Failed to retrieve extraction results. Check the error messages above.")
                st.info("Please retry the extraction on the '📤 Document Extraction' page. If the failure persists, contact support for assistance.")


render_footer()
