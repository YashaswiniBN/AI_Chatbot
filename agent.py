from langgraph.graph import StateGraph, END
from rag_pipeline import RAGPipeline
from llm_utils import claude_generate, gemini_generate

# Decision Node
def decide_node(state):
    query = state["query"]

    if "pdf" in query.lower() or "document" in query.lower():
        state["next"] = "retrieve"
    elif len(query.split()) < 3:
        state["next"] = "clarify"
    else:
        state["next"] = "answer"

    return state

# Retrieve Node
def retrieve_node(state):
    rag = state["rag"]
    context = rag.get_relevant_context(state["query"])
    state["context"] = context
    return state

# Answer Node
def answer_node(state):
    context = state.get("context", "")
    query = state["query"]
    llm_choice = state.get("llm", "claude")

    if llm_choice == "claude":
        answer = claude_generate(context, query)
    else:
        answer = gemini_generate(context, query)

    state["answer"] = answer
    return state

# Clarify Node
def clarify_node(state):
    state["answer"] = "Can you clarify your question?"
    return state

# Build Graph
graph = StateGraph(dict)

graph.add_node("decide", decide_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("answer", answer_node)
graph.add_node("clarify", clarify_node)

# FIXED routing
graph.add_conditional_edges(
    "decide",
    lambda state: state["next"]
)

graph.add_edge("retrieve", "answer")
graph.add_edge("clarify", END)
graph.add_edge("answer", END)

graph.set_entry_point("decide")

compiled_graph = graph.compile()

# Agent runner
def run_agent(query, rag, llm="claude"):
    state = {
        "query": query,
        "rag": rag,
        "llm": llm
    }

    result = compiled_graph.invoke(state)

    return result["answer"], result.get("context")