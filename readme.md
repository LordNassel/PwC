# Financial Agentic RAG Platform

## 1. Executive Summary & Problem Definition
**Domain Focus:** Corporate Earnings Report and Performance Transcript Analytics

Traditional vector-search Retrieval-Augmented Generation (RAG) architectures face inherent structural limits when applied to complex financial analysis. While baseline semantic lookups successfully isolate discrete textual statements (e.g., retrieving "In Q3, segment revenue reached 25,000 million dollars"), they fail when queries require cross-document comparison, chronological delta tracking, or compound quantitative synthesis. 

This repository implements an **Agentic RAG** platform designed to bridge this capability gap. By leveraging state-driven graph orchestration, the system dynamically extracts localized numeric parameters across separate corporate reporting boundaries (such as sequential quarterly or annual text files). It then channels these parameters into a sandboxed code execution tool to evaluate mathematical operations. Combining targeted vector retrieval with deterministic compiler validation ensures precise financial calculations with zero risk of numerical hallucination.

---

## 2. System Architecture & Topology Design
The platform architecture is built on a state-encapsulated directed graph using `LangGraph`. The workflow is organized into five explicit main processing nodes, supplemented by an isolated context retrieval subgraph.

* **Router Node:** Evaluates incoming query syntax and state telemetry to govern conditional path assignments. It manages the runtime routing between context acquisition, programmatic compilation, and final output handoff.
* **RAG Executor Node:** Routes targeted string queries into the modular document retrieval subgraph.
* **Math Executor Node:** Formulates sanitized raw algorithmic syntax strings from retrieved text parameters via LLM structural mapping and dispatches them to the verification compiler tool.
* **Synthesizer Node:** Aggregates multi-source context frames and execution tool outputs to compile final technical assessments in natural language.
* **Formatter Node:** Applies UI-compatible text layouts and appends the system execution metadata matrix.

### State Leak Guardrail & Modular Isolation
To prevent context mixing and cross-query state contamination commonly found in persistent conversational loops, the Router Node includes a strict isolation guard. If an execution transition originates outside known retrieval states, the system flushes all lingering database reference objects and mathematical variables from the state thread before triggering downstream components. 

The core document discovery process is separated into an isolated sub-graph (`rag_graph`). This subgraph compiles independently and is invoked natively by the primary supervisor nodes, keeping the architecture modular and maintainable.

---

## 3. Core Model Infrastructure & Trade-offs
The runtime environment is designed for completely local operation, bypassing external API endpoints to guarantee corporate data perimeter security and remove variable cloud service costs.

* **Inference Engine:** `qwen2.5:14b` served via an internal Ollama server instance. Qwen 2.5 was chosen over generic architectures due to its high accuracy in structured table parsing, strict adherence to system prompt boundaries, and native token formatting during numeric extraction loops.
* **Vector Embeddings Engine:** `bge-large` handled directly through the local model service layers. BGE-Large provides high dimensional accuracy when indexing mathematical records and dense business terminology compared to standard lightweight text encoders.

### Architecture Trade-offs
Processing a 14-billion parameter model and large embedding spaces locally increases token turnaround latency under high multi-threaded access. To preserve performance and structural stability without relying entirely on slow LLM classification passes, the routing system utilizes deterministic keyword and formula detection matrices to secure execution pathways.

---

## 4. Verification & Operational Telemetry

### Functional Validation Matrix
The workflow routing and guardrail mechanisms were verified using a 10-tier technical test matrix. The scenarios evaluated direct text extraction, sequential trend calculation, multi-entity formula synthesis, keyword fragment processing, and out-of-domain bound inspections.

The platform achieved **100% routing and execution precision** across all test categories:
* Quantitative questions cleanly routed into the code compiler tool.
* Incomplete phrase fragments generated structured item summaries instead of speculative calculations.
* Out-of-domain queries and missing corporate entities successfully triggered the data-absent fallback string (*"I do not have the data required to answer this question."*), preventing hallucinated responses.

### Concurrency Stress Testing
The infrastructure container was profiled under a concurrent access load executing 50 total query requests distributed across 4 parallel processing worker threads.

```text
Operational Performance Metrics:
--------------------------------------------------
Total Processed Transactions : 50
System Transaction Success   : 100.0% (50/50)
Total Wall-Clock Time        : 250.40 seconds
Calculated System Throughput : 0.20 req/sec
Latency Standard Dev (σ)     : 5.96s (Jitter Metric)
--------------------------------------------------
Minimum Turnaround Latency   : 17.63s
Median Turnaround (p50)      : 17.80s
Average Transaction Latency  : 19.50s
90th Percentile Latency (p90): 18.09s
95th Percentile Latency (p95): 38.74s
Maximum Turnaround Latency   : 45.55s
--------------------------------------------------

---

# 5. Deployment & Installation Guide

This platform is containerized using Docker to ensure dependency consistency and ease of deployment. Follow these steps to spin up the local infrastructure.

## Prerequisites

- Docker installed and running
- Docker Compose (V2 recommended)
- At least **16GB of system RAM** (due to the 14B parameter model requirements)

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. Verify Data Directory

Ensure your target financial documents (e.g., `.txt` or `.pdf` earnings reports) are placed inside the `./data` directory located at the root of the project.

### 3. Boot the Infrastructure

Build and start the application and database containers using Docker Compose. The custom `rag_network` will be created automatically to prevent port conflicts on port `11434`.

```bash
# Ensure any conflicting processes (like local Ollama services)
# are stopped on port 11434

sudo docker-compose up --build -d
```

### 4. Download the Neural Weights (First Run Only)

The inference engine and vector database require specific model weights. Pull them into the active Ollama container:

```bash
sudo docker exec -it ollama_server ollama pull bge-large
sudo docker exec -it ollama_server ollama pull qwen2.5:14b
```

### 5. Restart the Application Container

To ensure the application indexes your `./data` files correctly with the newly downloaded embedding model, restart the Streamlit container:

```bash
sudo docker restart agentic_streamlit
```

## Accessing the Platform

### Web Interface

Open your browser and navigate to:

```text
http://localhost:8501
```

### Run the Test Suite

To execute the validation matrix and stress test, run the following command:

```bash
sudo docker exec -it agentic_streamlit python evaluate.py
```

## Troubleshooting

### Port 11434 Conflict

If the `ollama_server` container fails to start with an **"address already in use"** error, ensure no native Ollama instances are running on your host machine:

```bash
sudo systemctl stop ollama
```

The included `docker-compose.yml` is configured to communicate internally over the Docker bridge network, helping mitigate host-level port conflicts.

### EOF / Timeout Errors

If the application crashes with an unexpected EOF (`status 500`), your machine may be out of memory, or the model file may be corrupted.

Potential solutions:

1. Delete and re-pull the affected model.
2. Verify sufficient RAM and swap space are available.
3. Switch to the lighter `qwen2.5:7b` model in `agentic_rag.py` if hardware resources are limited.