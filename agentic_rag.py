import os
import logging
from typing import TypedDict
from langchain_core.documents import Document
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Configure global system logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
# ==========================================
# 1. Inference & Embeddings Initialization
# ==========================================
# Dynamically resolve the Ollama service address
ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Pass the base_url parameters explicitly
embeddings = OllamaEmbeddings(
    base_url=ollama_base_url,
    model="bge-large"
)

llm = ChatOllama(
    base_url=ollama_base_url,
    model="qwen2.5:14b",
    temperature=0
)
# ==========================================
# 2. Document Loading & Text Chunking Pipeline
# ==========================================
real_documents = []
data_directory = "./data"

# Iterate over files inside the target data repository directory
if os.path.exists(data_directory):
    for filename in os.listdir(data_directory):
        file_path = os.path.join(data_directory, filename)
        try:
            # Handle standard flat-text files (.txt)
            if filename.endswith(".txt"):
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        real_documents.append(Document(page_content=content, metadata={"source": filename}))
                        logging.info(f"Loaded TXT asset: {filename}")
            
            # Handle standard unstructured portable document formats (.pdf) via PyPDFLoader
            elif filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                pdf_docs = loader.load()
                real_documents.extend(pdf_docs)
                logging.info(f"Loaded PDF asset: {filename}")
                
        except Exception as e:
            logging.error(f"Error reading file {filename}: {e}")

# System fallback token if directory is unpopulated or missing
if not real_documents:
    logging.warning("Data directory missing or empty. Using empty repository token.")
    real_documents = [Document(page_content="System Repository initialization token. Document database is currently empty.")]

# Initialize chunk splitter configuration maximizing table/list layout cohesion
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        
    chunk_overlap=150,     
    separators=["\n\n", "\n", " ", ""]
)
split_docs = text_splitter.split_documents(real_documents)

# ==========================================
# 3. Vector Database Indexing (ChromaDB)
# ==========================================
try:
    # Build or replace vector store memory collections using the specified embedding logic
    vectorstore = Chroma.from_documents(documents=split_docs, embedding=embeddings)
    # Expose the vector database as a top-k chunk document retriever context
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
except Exception as e:
    logging.error(f"Vector store indexing failed: {e}")
    retriever = None

# ==========================================
# 4. Isolated Execution Tooling
# ==========================================
@tool
def calculate_math(expression: str) -> str:
    """Evaluates basic mathematical expressions securely to avoid execution exploits."""
    # Sanitize inputs by filtering out characters that are not math operators or numbers
    clean_expr = "".join(c for c in expression if c in "0123456789+-*/(). ")
    if not clean_expr.strip():
        return "Error: No valid mathematical expression found."
    try:
        # Override builtins dictionary scope to block runtime arbitrary code injection attacks
        allowed_names = {"__builtins__": None}
        return str(eval(clean_expr, allowed_names, {}))
    except ZeroDivisionError:
        return "Error: Division by zero is undefined."
    except Exception as e:
        return f"Error calculating: {e}"

# ==========================================
# 5. Core Agent & Sub-Graph State Schemas
# ==========================================
class AgentState(TypedDict):
    """Global data state dictionary shared across primary workflow agent nodes."""
    question: str         # Original incoming prompt string
    context: str          # Collected matching text blocks retrieved from ChromaDB
    math_expression: str  # Structural formula isolated for math execution
    math_result: str      # Raw evaluation result provided by math compiler tool
    final_answer: str     # Synthetic conversational answer payload
    next_step: str        # Contextual string flag defining downstream node navigation

class RAGState(TypedDict):
    """Isolated local schema dictionary used exclusively inside the RAG sub-graph loop."""
    query: str
    retrieved_docs: str

# ==========================================
# 6. RAG Sub-Graph Definition
# ==========================================
def retrieve_docs(state: RAGState):
    """Executes search inquiries safely against the indexed Chroma Vector base."""
    try:
        if not retriever:
            raise ValueError("Uninitialized retriever connection context.")
        docs = retriever.invoke(state.get("query", ""))
        # Join documents using a standard visual string layout separator
        return {"retrieved_docs": "\n---\n".join([d.page_content for d in docs])}
    except Exception as e:
        logging.error(f"Vector store retrieval exception: {e}")
        return {"retrieved_docs": "Error: Context data could not be pulled from vector store."}

# Build and compile the isolated sub-graph app interface
rag_graph = StateGraph(RAGState)
rag_graph.add_node("retrieve", retrieve_docs)
rag_graph.set_entry_point("retrieve")
rag_graph.add_edge("retrieve", END)
rag_app = rag_graph.compile()

# ==========================================
# 7. Main Agent Application Graph Nodes
# ==========================================
def router_node(state: AgentState):
    """Determines execution routing paths based on historical node passes."""
    question = str(state.get("question", "")).strip()
    context = str(state.get("context", "")).strip()
    math_result = str(state.get("math_result", "")).strip()
    next_step = str(state.get("next_step", "")).strip()

    # Step A: Clean historical scratchpad values if this is the initial graph entry point
    if next_step not in ["from_rag", "math_node"]:
        return {
            "context": "",
            "math_expression": "",
            "math_result": "",
            "next_step": "rag_node"
        }

    # Step B: Scan incoming question against vocabulary keywords to invoke math tooling
    math_keywords = ["compare", "difference", "sum", "subtract", "plus", "calculate", "growth", "grow", "increase", "decrease", "+", "-"]
    if any(k in question.lower() for k in math_keywords) and not math_result:
        return {"next_step": "math_node"}
        
    # Step C: Fall through default fallback option directly into text resolution synthesis
    return {"next_step": "synthesize_node"}

def rag_executor_node(state: AgentState):
    """Interfaces with the secondary compiled RAG sub-graph to discover ground facts."""
    try:
        question = state.get("question", "")
        # Call the sub-graph as a standard executable step passing the source query parameter
        result = rag_app.invoke({"query": question})
        return {"context": result.get("retrieved_docs", ""), "next_step": "from_rag"}
    except Exception as e:
        logging.error(f"RAG step execution fault: {e}")
        return {"context": "Error: Vector database connectivity dropped.", "next_step": "from_rag"}

def math_executor_node(state: AgentState):
    """Isolates numerical values from content fragments and invokes the math execution tool."""
    context = state.get("context", "")
    question = state.get("question", "")
    
    try:
        # Prompt structural construction forcing the LLM into generating raw expression code only
        prompt = f"Based on this context:\n{context}\n\nWhat is the mathematical expression to answer: '{question}'? Respond ONLY with numbers and operators (e.g., 25000 - 20000). Do not include any text, reasoning, or formatting markdown wrapper tags."
        expression = llm.invoke(prompt).content.strip()
        
        # Strip conversational prefix leakages if the LLM fails to comply with standard output instructions
        if any(w in expression.lower() for w in ["sure", "based", "the", "expression"]):
            expression = "".join(c for c in expression if c in "0123456789+-*/(). ")
            
        # Trigger safe numeric calculator function
        result = calculate_math.invoke(expression)
        return {"math_expression": expression, "math_result": result, "next_step": "synthesize_node"}
    except Exception as e:
        logging.error(f"Math calculation generation failure: {e}")
        return {"math_expression": "None", "math_result": f"Error formatting calculation expression: {e}", "next_step": "synthesize_node"}

def synthesizer_node(state: AgentState):
    """Combines text contexts and calculation tools into a definitive structural answer."""
    question = state.get("question", "").strip()
    context = state.get("context", "None")
    math_result = state.get("math_result", "None")
    
    # Catch downstream runtime connection errors instantly to ensure system continuity
    if "Error:" in context or "Error:" in math_result:
        return {"final_answer": "System Error: The request could not be processed due to internal data asset timeouts.", "next_step": "formatter_node"}

    try:
        # Strict guidelines to force deterministic extraction answers without speculative assumptions
        prompt = f"""
        You are an expert, strict financial analyst. Answer the user's question using ONLY the provided document context fragments and specific tool calculation results.
        
        CRITICAL RULES:
        1. Base your response solely on the factual data provided below.
        2. If the user input is a single company keyword or name fragment, simply list the raw facts available for that company in the context without creating speculative comparisons.
        3. If the context does not contain the specific information required to accurately answer the question, state exactly: "I do not have the data required to answer this question." Do not extrapolate or assume.
        
        Question: {question}
        Context: {context}
        Math Result: {math_result}
        """
        answer = llm.invoke(prompt).content.strip()
        return {"final_answer": answer, "next_step": "formatter_node"}
    except Exception as e:
        logging.error(f"LLM compilation timeout anomaly: {e}")
        return {"final_answer": "Error: Synthesis module encountered a timeout processing this query.", "next_step": "formatter_node"}

def formatter_node(state: AgentState):
    """Appends quantitative execution trace summaries onto the completed output message."""
    final_ans = state.get("final_answer", "Error resolving state data.")
    math_expr = str(state.get("math_expression", "")).strip()
    math_res = str(state.get("math_result", "")).strip()
    
    formatted_output = f"### Final Answer\n{final_ans}\n\n"
    
    # Check if a quantitative mathematical path trace occurred to cleanly append metadata blocks
    if math_expr and math_res and math_expr.lower() != "none" and math_res.lower() != "none" and math_expr != "":
        formatted_output += f"**Metadata:**\n- Math Used: {math_expr} = {math_res}\n"
    else:
        formatted_output += f"**Metadata:**\n- Math Used: None\n"
        
    return {"final_answer": formatted_output}

# ==========================================
# 8. Operational Graph Routing Logic
# ==========================================
def route_step(state: AgentState):
    """Acts as the dynamic transition edge evaluator mapping out current active targets."""
    return state.get("next_step", "synthesize_node")

# ==========================================
# 9. Graph Architecture Construction & Compilation
# ==========================================
workflow = StateGraph(AgentState)

# Append tracking processing nodes to construction space
workflow.add_node("router_node", router_node)
workflow.add_node("rag_node", rag_executor_node)
workflow.add_node("math_node", math_executor_node)
workflow.add_node("synthesize_node", synthesizer_node)
workflow.add_node("formatter_node", formatter_node)

# Map foundational graph entry rules
workflow.set_entry_point("router_node")

# Dynamic execution edge switching using the condition evaluation helper function
workflow.add_conditional_edges(
    "router_node",
    route_step,
    {
        "rag_node": "rag_node",
        "math_node": "math_node",
        "synthesize_node": "synthesize_node"
    }
)

# Static relationship edge declarations mapping out linear completion sequences
workflow.add_edge("rag_node", "router_node")       # Returns back up to evaluate math needs after pull tasks
workflow.add_edge("math_node", "synthesize_node")  # Paths straight forward to synthesis once computed
workflow.add_edge("synthesize_node", "formatter_node")
workflow.add_edge("formatter_node", END)          # Terminates thread execution safely

# Compile operational blueprint configuration into a deployable application package
app = workflow.compile()