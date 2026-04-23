import os
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv
from layer1 import DocumentProcessor
from layer2 import MedicalKnowledgeBase       # NEW
from prompt_builder import build_final_prompt  # NEW

load_dotenv()

st.set_page_config(page_title="Clinical RAG Assistant", page_icon="🏥")

# ── Session State ──────────────────────────────────────────────────────────────
if "processor" not in st.session_state:
    st.session_state.processor = DocumentProcessor()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "kb" not in st.session_state:
    st.session_state.kb = None

# ── Medical KB: load ONCE, cache across all sessions ──────────────────────────
@st.cache_resource(show_spinner="Loading medical knowledge base...")
def load_medical_kb():
    med_kb = MedicalKnowledgeBase()
    med_kb.build()
    return med_kb

med_kb = load_medical_kb()  # NEW

# ── OpenRouter Client ─────────────────────────────────────────────────────────
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Document Center")

    # NEW: show medical KB status
    n_chunks = len(med_kb.chunks)
    if n_chunks > 0:
        st.success(f"✅ Medical KB: {n_chunks} chunks ready")
    else:
        st.warning("⚠️ No files found in refrences/")

    # NEW: doctor prompt input
    st.divider()
    doctor_prompt = st.text_area(
        "Doctor / System Instruction (optional):",
        placeholder="e.g. Focus on renal dosing. Patient is CKD stage 3.",
        height=100,
    )

    st.divider()

    # PDF ONLY — changed from ["pdf", "txt"]
    uploaded_file = st.file_uploader("Upload Patient PDF", type=["pdf"])
    if uploaded_file:
        if st.session_state.kb is None:
            with st.spinner("Indexing document..."):
                st.session_state.kb = st.session_state.processor.process_file(uploaded_file)
                st.success("Document ready!")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.session_state.kb = None
        st.rerun()

st.title("Clinical Decision Support Assistant")

# ── Chat History ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── User Interaction ──────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a clinical question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # ── Retrieval ─────────────────────────────────────────────────────────────
    pdf_chunks = []
    med_chunks = []

    with st.status("Searching knowledge sources...", expanded=False):
        # Layer 1: user PDF (same as before, just renamed variable)
        if st.session_state.kb:
            pdf_text_chunks = st.session_state.processor.hybrid_search_and_rerank(
                prompt, st.session_state.kb, top_k=3
            )
            pdf_chunks = [{"text": c, "source": "Patient PDF"} for c in pdf_text_chunks]
            st.write(f"PDF: {len(pdf_chunks)} chunks found")

        # Layer 2: medical papers (NEW)
        med_chunks = med_kb.search(prompt, top_k=3)
        st.write(f"Medical KB: {len(med_chunks)} chunks found")

    # ── Prompt Construction ───────────────────────────────────────────────────
    system_prompt = build_final_prompt(
        doctor_prompt=doctor_prompt,
        pdf_chunks=pdf_chunks,
        med_chunks=med_chunks,
    )

    # ── LLM Call (identical to your original) ─────────────────────────────────
    with st.chat_message("assistant"):
        response_placeholder = st.empty()

        response = client.chat.completions.create(
            model="tencent/hy3-preview:free",
            messages=[
                {"role": "system", "content": system_prompt},
                *st.session_state.messages
            ]
        )

        full_response = response.choices[0].message.content
        response_placeholder.write(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})