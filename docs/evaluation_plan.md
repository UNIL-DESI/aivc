# AIVC Evaluation Plan & Benchmarking Protocol (EASE 2027 / AAMAS 2027)

> [!IMPORTANT]
> **Official Strategic Decision**: AIVC evaluation will officially adopt Princeton NLP's open-source **SWE-bench-CL (Continual Learning)** benchmark via Docker. The evaluation is 100% automated (Pytest unit tests + non-regression + token efficiency reduction), completely abandoning human user study protocols (Amir) and MemGPT baselines.
> **Target Submissions**: EASE 2027 / AAMAS 2027.

## Overview

This document specifies the scientific evaluation strategy and automated benchmarking architecture for **AIVC** (AI Version Control). The evaluation framework is designed for **100% automated execution** using deterministic codebase validation and LLM-as-a-judge protocols, targeting publication at **EASE 2027** and **AAMAS 2027**.

---

## Target Conferences & Scope

* **EASE 2027 (Co-Primary Target)**: Empirical Assessment in Software Engineering. Empirical evaluation of agent memory systems on real software maintenance tasks.
* **AAMAS 2027 (Co-Primary Target)**: Autonomous Agents and Multiagent Systems. Focus on long-term memory architectures, state versioning, and multi-agent coordination.
* **ICLR / FAccT**: Secondary targets for representation learning, context retrieval, and governance.

---

## Benchmarking Protocol Comparison: Automated (Henri) vs. Human User Study (Amir)

| Axis | Amir's Proposal (Deprecated) | Henri's Protocol (Adopted - SWE-bench-CL via Docker) |
| :--- | :--- | :--- |
| **Methodology** | Human User Study (N=15-30 developers) & MemGPT | 100% Automated SWE-bench-CL + Pytest + LLM-as-a-Judge |
| **Inference Infrastructure** | Interactive developer sessions | Parallel execution via Docker + API execution |
| **Validation Metrics** | Qualitative surveys (SUS, NASA-TLX) | Pytest unit test pass rate, non-regression rate, token savings |
| **Reproducibility** | Subjective, high variance | 100% deterministic, seedable, reproducible |
| **Throughput** | Weeks of manual testing | Automated continuous evaluation pipeline |

---

## Benchmark Baselines & SOTA Literature

1. **SWE-bench-CL (Continual Learning - Princeton NLP)**: Primary benchmark framework running via Docker for sequential code maintenance.
2. **LoCoMo**: Long-Context Memory Benchmark for evaluating multi-turn retrieval.
3. **MemBench**: Long-term memory persistence and freshness in LLM agents.
4. **MemoryAgentBench**: Sequential memory update precision.
5. **ContextBench**: Structural memory vs flat RAG performance.

> [!NOTE]
> MemGPT and human user study protocols are officially abandoned in favor of the deterministic SWE-bench-CL pipeline.

---

## Automated 3-Tier Evaluation Architecture

### Tier 1: Micro-benchmarking (Component Level)
- **File Association Precision & Recall**: Accuracy of `read_files` and `edited_files` tracking.
- **Version Lineage Querying**: Precision of historical state reconstruction across commit trees.
- **Noise Injection Robustness**: Resistance against distractors in memory recall queries.

### Tier 2: Trajectory & Token Efficiency (Workflow Level)
- **Exploration Tool Call Reduction**: Percentage reduction in redundant `grep_search`, `view_file`, and `list_dir` actions.
- **Token Consumption**: Total prompt + output token count comparison per resolved issue.
- **Wall-Clock Speedup**: Latency reduction achieved by pre-populating relevant agent context.

### Tier 3: Continual SWE-bench (System Level - 50 Iterations)
Sequential evaluation across **50 consecutive repository issues** comparing four system variants:
1. **AIVC**: Active long-term memory with file associations & version lineage.
2. **Zero-Memory Ephemeral Agent**: Fresh state per task (no memory).
3. **Naive RAG Agent**: Unstructured vector search over flat code snippets.
4. **Unstructured Flat Memory**: Raw append-only log without file association index.

#### Key Metrics
- **Cumulative Pass@1 Rate**
- **Catastrophic Forgetting Rate**
- **Context Accumulation & Drift Rate**

---

## Action Plan

1. Curate a 50-task sequential SWE-bench repository dataset.
2. Implement automated evaluation harness in `tests/eval_harness.py`.
3. Execute Tier 1 & Tier 2 micro-benchmarks.
4. Run 50-iteration Continual SWE-bench benchmark using Together AI API.
5. Generate statistical figures and prepare paper submission for AAMAS.
