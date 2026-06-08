# Financial Agentic RAG Platform

## Overview

The Financial Agentic RAG Platform is a fully local, graph-orchestrated Retrieval-Augmented Generation (RAG) system designed for corporate earnings report and performance transcript analysis. Unlike traditional semantic retrieval systems that can only retrieve isolated pieces of information, this platform supports cross-document comparisons, temporal analysis, and deterministic financial calculations through an agent-driven execution pipeline.

By combining vector retrieval, state-based workflow orchestration, and compiler-validated mathematical execution, the platform enables accurate financial reasoning while eliminating numerical hallucinations.

---

# 1. Executive Summary & Problem Definition

**Domain Focus:** Corporate Earnings Report and Performance Transcript Analytics

Traditional vector-search Retrieval-Augmented Generation (RAG) architectures face inherent structural limitations when applied to complex financial analysis. While baseline semantic retrieval systems can successfully isolate individual statements (e.g., retrieving *"In Q3, segment revenue reached 25,000 million dollars"*), they struggle when queries require cross-document comparison, chronological trend tracking, or multi-step quantitative reasoning.

This repository implements an **Agentic RAG** platform designed to address these limitations. Using state-driven graph orchestration, the system dynamically extracts financial parameters across multiple reporting periods and routes them through a sandboxed computation layer for deterministic mathematical evaluation.

By combining targeted vector retrieval with compiler-validated calculations, the platform delivers precise financial insights while eliminating numerical hallucinations.

---

# 2. System Architecture & Topology Design

The platform is implemented as a state-managed directed graph using **LangGraph**. The workflow consists of five primary processing nodes supported by an isolated retrieval subgraph.

## Core Processing Nodes

### Router Node

Analyzes query structure and state telemetry to determine execution pathways. It governs routing between retrieval, computation, and response synthesis.

### RAG Executor Node

Dispatches retrieval requests into the dedicated document discovery subgraph and returns relevant contextual information.

### Math Executor Node

Transforms retrieved numerical information into validated computational expressions using structured LLM extraction and executes them through a sandboxed compiler tool.

### Synthesizer Node

Combines retrieved context and calculation outputs into a coherent analytical response.

### Formatter Node

Formats final responses for the user interface and appends execution metadata.

## State Isolation & Guardrails

To prevent context leakage and state contamination across conversations, the Router Node enforces strict execution isolation. Whenever an execution path originates outside known retrieval states, residual database references and computational variables are purged before downstream processing begins.

The document retrieval pipeline is encapsulated within an isolated subgraph named `rag_graph`. This modular separation improves maintainability, scalability, and future extensibility.

---

# 3. Core Model Infrastructure & Trade-offs

The platform is designed for fully local execution, eliminating external API dependencies to maximize data privacy and remove recurring cloud inference costs.

## Models

### Inference Engine

- **Model:** `qwen2.5:14b`
- **Runtime:** Ollama

Qwen 2.5 was selected due to its strong performance in:

- Structured document analysis
- Financial table interpretation
- Numerical extraction
- Prompt adherence
- Long-context processing

### Embedding Engine

- **Model:** `bge-large`

BGE-Large provides high-quality semantic representations for financial terminology, business language, and numerical records, resulting in improved retrieval accuracy compared to lightweight embedding models.

## Architectural Trade-offs

Running a 14-billion-parameter model locally introduces increased latency, particularly under concurrent workloads. To reduce unnecessary model utilization, the routing layer incorporates deterministic keyword detection and formula classification logic, minimizing reliance on expensive LLM-based decision-making.

This hybrid approach improves performance while maintaining routing accuracy.

---

# 4. Verification & Operational Telemetry

## Functional Validation Matrix

The platform was evaluated using a ten-category test suite covering:

- Direct information extraction
- Sequential trend analysis
- Multi-document calculations
- Formula synthesis
- Fragment-based queries
- Out-of-domain requests
- Missing-data scenarios
- Retrieval boundary conditions
- Router correctness
- State-isolation validation

The system achieved **100% routing and execution accuracy** across all validation scenarios.

### Validation Results

- Quantitative queries consistently routed to the computation engine.
- Incomplete query fragments generated structured summaries rather than speculative outputs.
- Out-of-domain and missing-data scenarios correctly returned:

> "I do not have the data required to answer this question."

thereby preventing hallucinated responses.

## Concurrency Stress Testing

The infrastructure was benchmarked using 50 requests distributed across 4 concurrent worker threads.

```text
Operational Performance Metrics
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
```

The results demonstrate stable execution under moderate concurrent load while maintaining perfect request completion rates.

---
## 5. Bottleneck Analysis & Optimization Recommendations

---

### Bottleneck Analysis

The primary system bottleneck is **local LLM inference latency via Ollama**.

The primary bottleneck is Local LLM Inference Latency via Ollama.
Because LLM generation is inherently sequential and compute-heavy, concurrent requests cause a queueing effect, spiking the Max Latency as threads wait for GPU/CPU availability.

### Optimization Suggestions

1. Semantic Caching: Implement a cache (e.g., Redis) in Node 1. If a user asks a
structurally identical or semantically similar question, return cached responses
to bypass LLM inference entirely.
2. Smaller / Quantized Models: Replace large models (e.g., Qwen2.5:14B) with lighter
 alternatives such as Phi-3-Mini (3.8B), or apply 4-bit quantization to reduce
ptoken generation latency and improve throughput under concurrency.
Request Batching: Batch multiple retrieval or generation requests where possible
to amortize inference overhead across tokens.
4. KV-Cache Optimization: Improve key-value cache reuse in the inference backend to
avoid recomputing attention states for shared prefixes.
5. Async Queue Control: Introduce a bounded asynchronous queue to smooth traffic
spikes and prevent GPU saturation during peak loads.

--- 
# 6. Deployment & Installation Guide

The platform is distributed as a Docker-based application stack to ensure reproducible deployments and dependency consistency across environments.

## Prerequisites

- Docker installed and running
- Docker Compose (V2 recommended)
- Minimum 16 GB RAM
- Linux, macOS, or Windows with Docker support

## Installation Steps

### 1. Clone the Repository

```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. Populate the Data Directory

Place your financial documents (`.txt`, `.pdf`, etc.) inside:

```text
./data
```

### 3. Start the Infrastructure

Build and launch all required services:

```bash
# Ensure conflicting Ollama instances are stopped first

sudo docker-compose up --build -d
```

The deployment automatically creates the custom Docker network used by the application stack.

### 4. Download Required Models

On first startup, download the required embedding and inference models into the Ollama container:

```bash
sudo docker exec -it ollama_server ollama pull bge-large
sudo docker exec -it ollama_server ollama pull qwen2.5:14b
```

### 5. Restart the Application Container

After downloading the models, restart the Streamlit application so indexing can begin:

```bash
sudo docker restart agentic_streamlit
```

## Accessing the Platform

### Web Interface

Open your browser and navigate to:

```text
http://localhost:8501
```

### Run the Evaluation Suite

To execute the validation matrix and concurrency benchmark:

```bash
sudo docker exec -it agentic_streamlit python evaluate.py
```

## Troubleshooting

### Port 11434 Already in Use

If the `ollama_server` container fails to start with an:

```text
address already in use
```

error, stop any host-level Ollama instances:

```bash
sudo systemctl stop ollama
```

The included Docker configuration is designed to communicate internally over a dedicated bridge network to minimize port conflicts.

### EOF / Status 500 Errors

Possible causes:

- Insufficient system memory
- Corrupted model weights
- Excessive concurrent workload

Recommended actions:

1. Remove and re-download the affected model.
2. Verify sufficient RAM and swap space are available.
3. Replace `qwen2.5:14b` with `qwen2.5:7b` if running on resource-constrained hardware.

---

# Technology Stack

| Component | Technology |
|------------|------------|
| Orchestration | LangGraph |
| LLM Runtime | Ollama |
| Inference Model | qwen2.5:14b |
| Embedding Model | bge-large |
| Frontend | Streamlit |
| Containerization | Docker |
| Validation Engine | Sandboxed Python Compiler |
| Retrieval Layer | Vector Search + Agentic Routing |

---

# Key Features

- Fully local deployment
- Zero external API dependency
- Agentic workflow orchestration
- Deterministic mathematical execution
- Financial document retrieval
- Cross-report comparison
- Numerical hallucination prevention
- State-isolated execution graph
- Dockerized infrastructure
- Automated evaluation framework
