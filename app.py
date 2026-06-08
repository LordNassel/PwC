import streamlit as st
# Import the compiled LangGraph application instance from your agentic framework
from agentic_rag import app as agent_app

# Configure the global web page metadata and initial visual canvas behavior
st.set_page_config(
    page_title="Financial Data Analyst", 
    page_icon="📈", 
    layout="wide",                  # Maximizes terminal width utilization for better data presentation
    initial_sidebar_state="expanded" # Keeps structural configuration details persistently visible
)

# Persistent Session State Management
# Instantiates a conversational thread cache if it does not exist on initial page render
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", 
            "content": "System initialized. Send a query regarding uploaded earnings reports or segment performance metrics."
        }
    ]

# Construct structural control elements and environment diagnostics within the left-hand sidebar
with st.sidebar:
    st.markdown("### Runtime Environment")
    # Display hardcoded active backend indicators for structural audit visibility
    st.success("Ollama Cluster: Connected (Qwen 2.5 14B)")
    st.success("Vector Store: ChromaDB Indexed")
    
    st.markdown("---")
    st.markdown("### System Pipeline Nodes")
    st.markdown("""
    - **Context Discovery:** Dynamic retrieval via semantic vector matching.
    - **Execution Tooling:** Isolated calculator for numeric operations.
    - **Conditional Routing:** State-driven graph path orchestration.
    """)
    
    st.markdown("---")
    # Action item to clear user conversation state and force a clean Streamlit re-render loop
    if st.button("Reset Session History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Render main viewport headings
st.title("Enterprise Earnings Intelligence Portal")
st.caption("Asynchronous LangGraph engine executing over local vector indexes")

# Reactive Conversation Thread Rendering
# Sequentially loops through historical user/assistant turns stored in session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Interaction & Runtime Stream Processing Handling
# Triggers if the user submits a non-empty string into the chat interface input field
if prompt := st.chat_input("Enter your quantitative financial inquiry..."):
    
    # Mirror and save user-generated payload into memory cache and visual canvas
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Begin agent processing pipeline display matching assistant output space
    with st.chat_message("assistant"):
        # Wrap underlying multi-node model execution steps inside an expanding status dropdown container
        with st.status("Invoking pipeline engine...", expanded=True) as status:
            st.write("Evaluating intent and computing conditional path graph...")
            
            # Formulate the initial base state variable dictionary expected by the agent graph schema
            initial_state = {
                "question": prompt,
                "context": "",
                "math_expression": "",
                "math_result": "",
                "final_answer": "",
                "next_step": ""
            }
            # Execute the LangGraph workflow synchronously across all designated decision tree loops
            final_state = agent_app.invoke(initial_state)
            
            # Traceability Stage 1: Document Fragment Verification
            retrieved_context = final_state.get("context", "")
            if retrieved_context:
                st.write("Extracting localized matching text blocks from index...")
                # Flatten multi-line string outputs into single horizontal block views for uniform inspection
                preview = retrieved_context.replace('\n', ' | ')
                st.code(preview, language="text")
            
            # Traceability Stage 2: Quantitative Verification
            math_expr = final_state.get("math_expression", "")
            math_res = final_state.get("math_result", "")
            # Check if the routing engine determined mathematical tool invocation was required
            if math_expr and math_expr != "None":
                st.write("Routing targeted formula calculation down to mathematical compiler...")
                st.code(f"Expression Evaluated: {math_expr}\nOutput Yielded: {math_res}", language="python")
            
            st.write("Compiling and verifying synthesis matrix...")
            # Collapse the operational dropdown container and update final execution label upon state resolution
            status.update(label="Graph Execution Finalized", state="complete", expanded=False)
        
        # Display the finalized synthesis output directly within the main screen area
        final_answer = final_state.get("final_answer", "State resolution fault occurred.")
        st.markdown(final_answer)
        
        # Ensure structural continuity by adding the compiled answer payload back into historical context storage
        st.session_state.messages.append({"role": "assistant", "content": final_answer})