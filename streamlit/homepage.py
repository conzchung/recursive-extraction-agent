import streamlit as st
import datetime

st.set_page_config(
    page_title="AI Extraction & Validation",
    page_icon="🐱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 0.5rem !important;
        margin-top: 0rem !important;
    }
    div[data-testid="stDecoration"] {visibility: hidden; height: 0%; position: fixed;}
    div[data-testid="stStatusWidget"] {visibility: hidden; height: 0%; position: fixed;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align: center;'>💡 Welcome to AI Extraction &amp; Validation! 📄</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<h3 style='text-align: center;'>Turn messy documents into structured data, then validate results with evidence-backed checks.</h3>",
    unsafe_allow_html=True,
)

with st.expander("🔔 **New Updates** 🔔", expanded=True):
    st.markdown("### 🛡️ Validation workflow is now available")
    st.markdown(
        """
- **Rule Setup (New)**: create/manage validation rules before running **Content Validation**.
- **Content Validation**: validate extraction results with per-rule **score (1–10)**, **reason**, and **evidence (markdown)**.
- **Submission protection**: content validation is **blocked** unless all selected extraction IDs are **🟢 succeeded**.
- **Validation Results (Monitor & Query)**: combined page to track jobs at a glance and fetch one result by **Validation ID** (JSON download supported).
"""
    )

    st.markdown("---")
    st.markdown("### 🤖 Extraction model choices")
    st.markdown("- **Basic (GPT-5.4-mini)**: fast and cost-effective, best for quick or simpler documents.")
    st.markdown("- **Advanced (GPT-5.4)**: stronger reasoning for complex documents (slower and higher cost).")

    st.markdown("---")
    st.markdown("### 🧰 Document preprocessing upgrades")
    st.markdown("- **Auto-orientation**: rotates pages upright to improve readability and extraction quality.")
    st.markdown("- **Smart cropping**: removes unnecessary margins/whitespace so the model focuses on content.")
    st.markdown("- **Quality enhancement**: cleans and sharpens pages to improve text clarity.")

st.header("What is AI Extraction & Validation? 🚀")
st.markdown(
    """
A workflow app for:
- **Extraction**: convert unstructured documents into structured outputs using configured fields.
- **Validation**: score and explain quality/compliance checks with evidence so results are reviewable and auditable.

**Outcome**: less manual review work, faster turnaround, and more confidence in extracted data.
"""
)

st.markdown("---")

st.header("🌟 Explore Our Features 🌟")
st.markdown("Use the tabs below to understand each page, then open pages from the **nevigation bar** to run workflows.")

tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    [
        "✂️ Document Partitioning",
        "🛠️ Configuration Setup",
        "📤 Document Extraction",
        "📊 Extraction Results (Monitor & Query)",
        "🧾 Rule Setup",
        "🛡️ Content Validation",
        "📈 Validation Results (Monitor & Query)",
    ]
)

with tab0:
    st.markdown(
        """
- **Purpose**: split one combined document into separate files (e.g., multi-month statements or multiple banks).
- **How to use**:
  - Open **Document Partitioning** from the **nevigation bar**.
  - Upload a combined PDF.
  - Click **Split Document**.
  - Download a ZIP of separated documents.
- **Best for**: organizing bulk inputs before extraction.
"""
    )

with tab1:
    st.markdown(
        """
- **Purpose**: define and manage field configurations for extraction.
- **How to use**:
  - Open **Configuration Setup** from the **nevigation bar**.
  - Create a **Config ID** and define fields (name/date/amount/etc.).
  - Save and reuse configurations later.
- **Best for**: consistent extraction across repeated document types.
"""
    )

with tab2:
    st.markdown(
        """
- **Purpose**: run extraction using a saved configuration.
- **How to use**:
  - Open **Document Extraction** from the **nevigation bar**.
  - Enter an **Extraction ID**.
  - Select a **Configuration** and preview fields.
  - Upload a document and click **Extract Data**.
- **Best for**: turning unstructured documents into structured results.
"""
    )

with tab3:
    st.markdown(
        """
- **Purpose**: one place to **monitor** extraction jobs and **query** a specific Extraction ID.
- **What it includes**:
  - Recent extraction list with statuses: 🟢 succeeded / 🔴 failed / 🔄 processing / ⏳ queued
  - Search / fetch by **Extraction ID**
  - Downloads when results are ready
- **How to use**:
  - Open **Extraction Results** from the **nevigation bar**.
  - Use the monitor section to track recent jobs.
  - Use the query section when a specific Extraction ID is known.
- **Best for**: reducing page switching for monitoring + lookup.
"""
    )

with tab4:
    st.markdown(
        """
- **Purpose**: create and manage validation rule sets used by content validation.
- **How to use**:
  - Open **Rule Setup** from the **nevigation bar**.
  - Create/edit rule sets (each rule: alias + rule text).
  - Save rules, then select them later in **Content Validation**.
- **Best for**: maintaining reusable rule libraries and keeping validation consistent.
"""
    )

with tab5:
    st.markdown(
        """
- **Purpose**: validate one or more extraction results against a selected rule set.
- **How to use**:
  - Open **Content Validation** from the **nevigation bar**.
  - Select a **Rule** and provide up to **5 extraction IDs**.
  - Click **Check Status** to confirm readiness.
  - Click **Queue Validation** to submit.
- **Important**:
  - Submission is **blocked** unless all selected extraction IDs are **🟢 succeeded**.
- **Best for**: quality/compliance checks with scored, explainable outputs.
"""
    )

with tab6:
    st.markdown(
        """
- **Purpose**: one place to **monitor** validation jobs and **query** a specific Validation ID.
- **What it includes**:
  - Recent validation list with statuses: 🟢 succeeded / 🔴 failed / 🔄 processing / ⏳ queued
  - Search / fetch by **Validation ID**
  - Result display with a **score threshold slider (default 6.0)** to auto-expand evidence for low scores
  - **JSON download** for sharing/auditing
- **How to use**:
  - Open **Validation Results** from the **nevigation bar**.
  - Use the monitor section to track recent validations.
  - Use the query section when a specific Validation ID is known.
- **Best for**: reducing page switching for progress tracking + deep-dive review.
"""
    )

st.markdown("---")

st.subheader("Get Started Now! 🚀")
st.markdown(
    """
- Choose a page from the **nevigation bar** to begin: configure → extract → monitor/review → set up rules → validate → monitor/review.
"""
)

# st.markdown("---")

current_year = datetime.datetime.now().year
st.markdown(
    f"""
    <hr style="margin-top: 2rem; margin-bottom: 0.5rem;">
    <div style="text-align: center; color: gray; font-size: 0.85rem;">
        © {current_year} AI Extraction & Validation
    </div>
    """,
    unsafe_allow_html=True,
)