import streamlit as st
from rag_pipeline import RAGPipeline
from agent import run_agent

st.set_page_config(page_title="Agentic RAG Chatbot")

st.title("Agentic RAG Chatbot")

if "rag" not in st.session_state:
    st.session_state["rag"] = RAGPipeline()

if "history" not in st.session_state:
    st.session_state["history"] = []

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    chunks = st.session_state["rag"].load_and_split("uploaded.pdf")
    st.session_state["rag"].build_vectorstore(chunks)

    st.success("PDF processed!")

query = st.text_input("Ask a question")

llm_choice = st.selectbox("Choose LLM", ["claude", "gemini"])

if st.button("Send") and query:
    answer, context = run_agent(
        query,
        rag=st.session_state["rag"],
        llm=llm_choice
    )

    st.session_state["history"].append((query, answer, context))

st.markdown("---")

for q, a, ctx in reversed(st.session_state["history"]):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")

    if ctx:
        with st.expander("Context"):
            st.write(ctx)

    st.markdown("---")