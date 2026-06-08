import time
import statistics
import concurrent.futures
import logging
# Import the compiled LangGraph execution engine for testing
from agentic_rag import app as agent_app

# Suppress noisy HTTP connection logs triggered during high-volume parallel requests
logging.getLogger("httpx").setLevel(logging.WARNING)

# ==========================================
# 1. Functional Integrity Evaluation Suite
# ==========================================
# This matrix defines explicit behavioral expectations for different query archetypes.
# It ensures the Router Node makes the correct decisions regarding math execution and data rejection.
EVALUATION_SET = [
    {
        "id": 1, 
        "question": "What was Tesla's total revenue in Q3?", 
        "expect_math": False,     # Simple extraction: no math required
        "expect_rejection": False, # Data exists: do not reject
        "category": "RAG Extraction"
    },
    {
        "id": 2, 
        "question": "What is the difference in revenue between Tesla Q3 and Q2?", 
        "expect_math": True,      # Comparative trigger: math must be executed
        "expect_rejection": False,
        "category": "Quantitative Logic"
    },
    {
        "id": 3, 
        "question": "What was Apple's hardware revenue in 2023?", 
        "expect_math": False, 
        "expect_rejection": False,
        "category": "RAG Extraction"
    },
    {
        "id": 4, 
        "question": "Calculate the sum of Apple's hardware revenue in 2021 and 2023.", 
        "expect_math": True, 
        "expect_rejection": False,
        "category": "Quantitative Logic"
    },
    {
        "id": 5, 
        "question": "What was Microsoft total revenue was 143000 million dollars in 2020?", 
        "expect_math": False, 
        "expect_rejection": False,
        "category": "RAG Extraction"
    },
    {
        "id": 6, 
        "question": "How much did Microsoft's total revenue grow between 2020 and 2021?", 
        "expect_math": True, 
        "expect_rejection": False,
        "category": "Quantitative Logic"
    },
    {
        "id": 7, 
        "question": "Apple", 
        "expect_math": False, 
        "expect_rejection": False, # System should extract basic Apple facts without calculating
        "category": "Phrase Fragments"
    },
    {
        "id": 8, 
        "question": "Weather in Linz in 2023 June", 
        "expect_math": False, 
        "expect_rejection": True,  # Out of domain: system must explicitly reject
        "category": "Bound Inspections"
    },
    {
        "id": 9, 
        "question": "What was Nvidia's total revenue in 2025?", 
        "expect_math": False, 
        "expect_rejection": True,  # Non-existent chronological data: system must explicitly reject
        "category": "Bound Inspections"
    },
    {
        "id": 10, 
        "question": "Compare Microsoft 2021 total revenue to Apple 2023 hardware revenue.", 
        "expect_math": True,       # Cross-document calculation test
        "expect_rejection": False,
        "category": "Quantitative Logic"
    },
    {
        "id": 11, 
        "question": "What was Nvidia's Data Center revenue in Q1 2024?", 
        "expect_math": False, 
        "expect_rejection": False, # Deep nested segment lookup
        "category": "RAG Extraction"
    },
    {
        "id": 12, 
        "question": "What is the difference between Amazon's AWS segment sales and Nvidia's Data Center revenue in Q1 2024?", 
        "expect_math": True,       # Complex multi-segment cross-document calculation
        "expect_rejection": False,
        "category": "Cross-Document Logic"
    },
    {
        "id": 13, 
        "question": "Calculate the sum of Amazon's North America and International segment sales in Q1 2024.", 
        "expect_math": True, 
        "expect_rejection": False, # Inter-document segment calculation
        "category": "Quantitative Logic"
    }
]

def run_functional_evaluation():
    """Iterates through the EVALUATION_SET to map routing decisions against expected logic."""
    print("\n[INFO] Starting semantic and functional validation suite")
    print("-" * 60)
    
    passed_tests = 0
    category_stats = {}
    
    for item in EVALUATION_SET:
        # Initialize dictionary to track pass rates for different query categories
        cat = item["category"]
        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "passed": 0}
        category_stats[cat]["total"] += 1
        
        print(f"Test #{item['id']} [{cat}] | Query: '{item['question']}'")
        
        # Build the initial state schema expected by the LangGraph engine
        initial_state = {
            "question": item['question'], 
            "context": "", 
            "math_expression": "",
            "math_result": "", 
            "final_answer": "",
            "next_step": ""
        }
        
        try:
            # Execute the query through the full agent pipeline
            final_state = agent_app.invoke(initial_state)
            
            # Extract tracking variables from the returned final state dictionary
            math_expr = final_state.get("math_expression", "").strip()
            math_res = final_state.get("math_result", "").strip()
            final_ans = final_state.get("final_answer", "")
            
            # Boolean check: Did the agent actually utilize the math tooling path?
            math_used = bool(math_expr and math_res and math_expr.lower() != "none" and math_res.lower() != "none")
            # Boolean check: Did the agent trigger the hardcoded out-of-bounds rejection string?
            rejection_triggered = "I do not have the data required" in final_ans
            
            # Verify actual system behavior against expected test parameters
            math_correct = (math_used == item['expect_math'])
            rejection_correct = (rejection_triggered == item['expect_rejection'])
            
            if math_correct and rejection_correct:
                print("  -> Status: PASS")
                passed_tests += 1
                category_stats[cat]["passed"] += 1
            else:
                # Failure branch providing exact deviation context for debugging
                print("  -> Status: FAIL (Assertion Mismatch)")
                print(f"     Expected Math Flag: {item['expect_math']} | Actual: {math_used}")
                print(f"     Expected Rejection Flag: {item['expect_rejection']} | Actual: {rejection_triggered}")
                
        except Exception as e:
            print(f"  -> Status: ERROR (Graph Execution Fault): {e}")

    # Calculate and output total system precision metrics
    global_accuracy = (passed_tests / len(EVALUATION_SET)) * 100
    
    print("\nCategorical Accuracy Metrics:")
    print("-" * 40)
    for cat, stats in category_stats.items():
        cat_acc = (stats["passed"] / stats["total"]) * 100
        print(f" * {cat:<20}: {cat_acc:>5.1f}% ({stats['passed']}/{stats['total']})")
        
    print("-" * 40)
    print(f"Overall Test Suite Accuracy: {global_accuracy:.1f}%")
    print("-" * 40)


# ==========================================
# 2. Concurrency Workload Profiling
# ==========================================
NUM_REQUESTS = 50        # Total queries to execute
CONCURRENT_WORKERS = 4   # Parallel threads mimicking simultaneous active users

def single_query_worker(worker_id):
    """Executes a benchmark RAG request payload to profile operational throughput."""
    start_time = time.time()
    
    # Establish a baseline extraction query standard across all threads
    initial_state = {
        "question": "What was Apple's hardware revenue in 2023?", 
        "context": "", "math_expression": "", "math_result": "",
        "final_answer": "", "next_step": ""
    }
    
    success = False
    try:
        res = agent_app.invoke(initial_state)
        # Verify the pipeline actually returned a populated answer string
        if res.get("final_answer"):
            success = True
    except Exception as e:
        logging.error(f"Thread worker {worker_id} encountered execution exception: {e}")
        
    # Return elapsed time and success flag back to the thread pool collector
    return time.time() - start_time, success

def run_load_test():
    """Fires a burst of asynchronous requests to stress-test token generation latency."""
    print(f"\n[INFO] Initializing concurrency stress test (Requests: {NUM_REQUESTS}, Workers: {CONCURRENT_WORKERS})")
    print("-" * 60)
    
    latencies = []
    successes = 0
    wall_clock_start = time.time()
    
    # Instantiate the thread pool and map the worker function across the defined request count
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        futures = [executor.submit(single_query_worker, i) for i in range(NUM_REQUESTS)]
        
        # Resolve threads as they complete to prevent blocking the data collection
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            latency, success = future.result()
            latencies.append(latency)
            if success:
                successes += 1
            if i % 10 == 0:
                print(f"Progress: {i}/{NUM_REQUESTS} total transactions completed")

    # Finalize total system execution time
    wall_clock_end = time.time()
    total_elapsed_time = wall_clock_end - wall_clock_start
    
    # Calculate fundamental volume limits
    throughput_rps = NUM_REQUESTS / total_elapsed_time
    success_rate = (successes / NUM_REQUESTS) * 100
    
    # Compute statistical turnaround properties
    avg_latency = statistics.mean(latencies)
    median_latency = statistics.median(latencies)
    std_dev_latency = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    
    # Generate 100-tier distribution to identify extreme queueing delays (p90/p95 tails)
    pct_distribution = statistics.quantiles(latencies, n=100)
    p90_latency = pct_distribution[89]
    p95_latency = pct_distribution[94]
    
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    # Output the structured performance matrix
    print("\nOperational Telemetry Summary:")
    print("-" * 50)
    print(f"Total Processed Requests   : {NUM_REQUESTS}")
    print(f"Transaction Success Rate   : {success_rate:.1f}% ({successes}/{NUM_REQUESTS})")
    print(f"Total Execution Runtime    : {total_elapsed_time:.2f}s")
    print(f"Calculated Throughput      : {throughput_rps:.2f} req/sec")
    print(f"Latency Standard Dev (σ)   : {std_dev_latency:.2f}s (Jitter Metric)")
    print("-" * 50)
    print(f"Minimum Turnaround Latency : {min_latency:.2f}s")
    print(f"Median Turnaround (p50)    : {median_latency:.2f}s")
    print(f"Average Node Latency       : {avg_latency:.2f}s")

# This is the crucial trigger that was missing!
if __name__ == "__main__":
    run_functional_evaluation()
    run_load_test()