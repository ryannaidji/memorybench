# MemoryBench

**Durable customer memory for support conversations, with an LLM eval regression gate.**

MemoryBench reads a customer-support conversation, extracts the facts worth remembering
(contact preference, plan, open issues, preferences, profile), stores them, and recalls the
right ones in the next conversation. A labelled golden set measures extraction and retrieval
quality, and GitHub Actions runs a real LLM evaluation on every push: **the build fails if quality drops**.

Everything runs on free, local, open-weight models through [Ollama](https://ollama.com). The client
speaks the OpenAI-compatible API, so swapping in another provider is a config change, not a code change.

![demo](docs/demo.png)

## Results

Golden set: 60 synthetic support conversations (20 customers × 3), 100 labelled facts, 100 retrieval queries.
Measured on a laptop CPU (Intel, 8 cores, no GPU), temperature 0, embeddings `nomic-embed-text`.

| Model | Prompt | Precision | Recall | F1 | Retrieval recall@5 | MRR | Latency p50 / p95 |
|---|---|---|---|---|---|---|---|
| qwen2.5:3b | v1 | 0.30 | 0.65 | **0.41** | 0.63 | 0.46 | 18.9 s / 90.8 s |
| qwen2.5:3b | v2 | 0.84 | 0.60 | **0.70** | 0.63 | 0.52 | 10.2 s / 14.9 s |
| qwen2.5:1.5b (base) | v2 | 0.56 | 0.44 | **0.49** | 0.47 | 0.38 | 5.9 s / 13.1 s |
| **qwen2.5:1.5b + LoRA (fine-tuned)** | v2 | 0.92 | 0.93 | **0.93** | 0.93 | 0.75 | 6.4 s / 8.5 s |
| qwen2.5:1.5b * | v2 | 0.37 | 0.28 | **0.32** | 0.32 | 0.22 | 6.9 s / 22.8 s |

\* first 15 conversations only, run before the placeholder filter. Too weak to be a useful gate, so CI uses qwen2.5:3b. The v1 row was also measured before the placeholder filter was added.

## Fine-tuning: a 1.5B model beats the 3B one

A LoRA adapter (r=16) on `Qwen2.5-1.5B-Instruct`, trained with Unsloth on a **free Colab T4 GPU** in **3.7 minutes**
on 360 synthetic conversations ([`finetune/finetune_colab.ipynb`](finetune/finetune_colab.ipynb)), exported to GGUF q4_k_m
and served by Ollama ([`finetune/Modelfile`](finetune/Modelfile)).

- Training data is generated separately (`evals/gen_train.py`, seed 11, different customers) and **one phrasing template per
  category is held out**: 18 of the 100 eval facts use phrasings the model never saw.
- Same 1.5B model, same prompt: **F1 0.49 → 0.93**. It also beats the 3B prompt-only model (0.70) while being faster (p50 6.4 s vs 10.2 s).

Recall split by whether the phrasing appeared in training:

| Model | Seen phrasings (82 facts) | Unseen phrasings (18 facts) |
|---|---|---|
| qwen2.5:1.5b base | 0.45 | 0.39 |
| qwen2.5:3b, prompt v2 | 0.62 | 0.50 |
| qwen2.5:1.5b + LoRA | 1.00 | **0.61** |

Honest reading: perfect recall on seen phrasings is partly template memorisation (the data is synthetic); the unseen column
is the meaningful number, and the fine-tuned model still generalises better than both prompt-only baselines.
Lesson learned: the exported GGUF had no chat template; served raw, outputs were garbage until the Qwen chat template was
added to the Modelfile.

**What changed between v1 and v2.** Error analysis of the v1 run showed the model over-extracting
(218 facts for 100 gold): invented details ("Technology company, specializes in e-commerce"), one sentence
split into three facts, language preferences filed as contact preferences, and "Bye" stored as a fact.
Prompt v2 restricts extraction to explicit customer statements, one fact per sentence, sharper category
definitions and one few-shot example; a post-filter drops placeholder values such as "Not specified".

Raw outputs, including every predicted fact per conversation, are in [`evals/results/`](evals/results).

## How it works

```mermaid
flowchart LR
    C[Conversation] --> X[Extractor<br/>Ollama · JSON-schema output]
    X --> V{Pydantic validation<br/>dedupe · placeholder filter}
    V --> S[(SQLite fact store<br/>+ embeddings)]
    Q[Question] --> E[Embed query] --> R[Cosine top-k] 
    S --> R --> A[Relevant facts]
    G[Golden set] --> EV[Eval harness<br/>P / R / F1 · recall@5 · MRR · latency]
    X -.-> EV
    EV --> CI{CI gate<br/>thresholds-ci.json}
```

- **Structured output**: the model decodes under the Pydantic JSON schema (`FactList`), then the result is validated again;
  invalid JSON is retried once and counted as a parse error.
- **Matching for evals**: a predicted fact counts as correct when the category matches and the value is semantically
  equivalent to the gold fact (embedding cosine ≥ 0.80, each gold fact matched at most once).
- **Retrieval is evaluated end to end**: queries run against the facts the extractor actually stored, not the gold facts.

## Streaming ingestion (v0.3)

Conversations can also arrive as events on a Kafka-API topic and get processed by a group of consumers that write to PostgreSQL.

```mermaid
flowchart LR
    P[Producer<br/>golden set / load generator] -- "key = customer_id<br/>headers: trace_id, produced_at" --> T[(Redpanda topic<br/>conversations · 3 partitions)]
    T --> C1[Consumer c0]
    T --> C2[Consumer c1]
    T --> C3[Consumer c2]
    C1 & C2 & C3 -- "extract() via Ollama" --> PG[(PostgreSQL 16 + pgvector<br/>facts · processed_events)]
    C1 & C2 & C3 -- "malformed after retries" --> D[(conversations.dlq)]
    PG -- "commit offset AFTER DB txn" --> T
```

- **Partitioning and ordering**: events are keyed by `customer_id`, so each customer's conversations stay ordered within one partition. The consumers share one consumer group (`memorybench-extractors`), and you can run 1 to K of them.
- **At-least-once delivery**: auto-commit is off. A consumer commits the offset synchronously only after the Postgres transaction commits, so a crash before the commit means the message gets redelivered.
- **Idempotency**: the facts and a `processed_events` row (PRIMARY KEY `event_id`) are written in a single transaction. A redelivered event is skipped before the LLM runs. If the same conversation comes back under a new event id, the `UNIQUE (customer_id, category, value)` constraint on facts absorbs the duplicates.
- **Dead-letter topic**: a payload that still fails JSON/Pydantic parsing after 3 attempts goes to `conversations.dlq`, with the error and its source partition and offset as headers, and then its offset is committed. A transient LLM failure works differently: the consumer retries it 3 times, then crashes without committing, so the message is redelivered rather than dead-lettered.
- **Store**: `PostgresFactStore` ([`memorybench/pgstore.py`](memorybench/pgstore.py)) has the same `add / list_facts / recall` interface as the SQLite store. Recall is plain SQL: `ORDER BY embedding <=> query::vector` (pgvector cosine distance). The schema lives in [`memorybench/migrations/`](memorybench/migrations) and is applied once, under an advisory lock. SQLite stays the default; set `STORE=postgres` to switch.
- **Observability**: logs are JSON lines ([`memorybench/logjson.py`](memorybench/logjson.py), stdlib only). The producer creates a `trace_id`, sends it as a Kafka header, and every consumer log line for that message carries it. The id is also stored in `processed_events.trace_id`.

### Run it

```bash
docker compose up -d redpanda postgres        # Redpanda --smp 1 --memory 512M; Postgres on host port 55432
STORE=postgres python -m memorybench.stream.consumer --id c0 &   # start 1..K of these
python -m memorybench.stream.producer --source golden --limit 30
python -m memorybench.stream.producer --source load --n 200 --run r2   # load generator
python -m evals.bench_stream scale --consumers 1 2 3 --n 30
python -m evals.bench_stream idempotency --n 15
python -m evals.bench_stream failure --n 20 --kill-after 6
PG_DSN=postgresql://memorybench:memorybench@localhost:55432/memorybench KAFKA_BOOTSTRAP=localhost:19092 pytest -m integration
```

### Measured results

Setup: one laptop (CPU only, Intel, 8 cores), model `memorybench-qwen1.5b-ft`, 3 partitions, golden conversations, each consumer a separate process. End-to-end latency runs from the producer timestamp to the DB commit, and every event is published at t=0, so latency includes time spent waiting in the queue.
Raw output: [`evals/results/stream/`](evals/results/stream). That folder has `scale.json`, `idempotency.json` and `failure.json`, plus the per-consumer JSON logs.

| Consumers | Conversations | Throughput (conv/min) | E2E latency p50 / p95 | Extraction p50 | Messages per consumer |
|---|---|---|---|---|---|
| 1 | 30 | **7.98** | 119.5 s / 218.0 s | 7.6 s | 30 |
| 2 | 30 | **7.35** | 126.2 s / 233.6 s | 10.0 s | 24 / 6 |
| 3 | 30 | **4.50** | 229.7 s / 392.8 s | 27.7 s | 12 / 12 / 6 |

**Adding consumers did not help on one machine; it hurt.** All of them call the same local Ollama server on the same CPU, so the LLM is the bottleneck. Extraction p50 went from 7.6 s with 1 consumer to 27.7 s with 3, because concurrent requests compete for the same cores. On top of that, the keys are skewed: 30 conversations from 10 customers hashed into partitions of 12/12/6. The consumer group and partitioning work as designed. To scale out, each consumer needs its own inference capacity, such as more machines or a GPU server with batching. Adding processes on one CPU does not do it.

| Test | Result |
|---|---|
| Idempotency (15 conversations) | First pass: 15 events, **25 facts**. Replaying the same 15 events: 15 `duplicate_skipped`, still **25 facts**. Replaying with new event ids (re-extracted): 30 events, still **25 facts** |
| Failure (2 consumers, 20 events + 2 poison messages, `c0` hard-killed after 6 events processed) | 20 produced, **20 processed** (c0: 3, c1: 17 after the rebalance), **0 lost**, 2 sent to the DLQ |

During the failure run, no message happened to be in the window between "DB committed" and "offset committed" when c0 was killed, so 0 redeliveries were observed. That path is covered by the redelivery unit test and by the Kafka integration test, which replays events.

### Limitations

- Everything runs on one laptop: a single Redpanda broker with replication factor 1, and one Postgres instance. Nothing here measures multi-node behavior.
- The data is synthetic (the golden set). The runs are short (15 to 30 conversations), so the throughput figures are rough and noisy.
- The fact-level idempotency relies on deterministic extraction (temperature 0). If a re-extraction phrases a fact differently, it is stored as a new fact. The event ledger is the real guard.
- There is no OpenTelemetry exporter; the trace id is propagated by hand through headers and logs.
- CI runs the integration tests against PostgreSQL and pgvector as a GitHub Actions service container, and Redpanda started with `docker run`, because a service container cannot take Redpanda's start flags. Those tests use a fake extractor and fake embeddings.

## Eval gate in CI

`.github/workflows/ci.yml` runs on every push:
1. `unit-tests`: pytest with a fake LLM and fake embeddings (no model needed).
   `integration-tests`: Postgres/pgvector + Redpanda, real SQL and a real broker, fake extractor.
2. `eval-gate`: installs Ollama on the free GitHub runner, pulls `qwen2.5:3b` and `nomic-embed-text`, runs the eval on the
   first 15 conversations and fails if any metric is below [`evals/thresholds-ci.json`](evals/thresholds-ci.json).
   The eval JSON is uploaded as a build artifact.

## Run it

```bash
ollama pull qwen2.5:3b && ollama pull nomic-embed-text
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # (bin/ on Linux/macOS)

pytest -q                                                    # unit tests
python -m evals.run_eval --model qwen2.5:3b --out evals/results/run.json
python -m evals.run_eval --model qwen2.5:3b --limit 15 --gate evals/thresholds-ci.json
uvicorn memorybench.api:app --reload                         # API on :8000
pip install streamlit && streamlit run demo/app.py           # demo UI
docker compose up                                            # API + Ollama
```

API:

| Method | Path | |
|---|---|---|
| POST | `/conversations` | extract and store facts from `{customer_id, turns:[{role,text}]}` |
| GET | `/customers/{id}/facts` | all stored facts |
| GET | `/customers/{id}/recall?q=...&k=5` | top-k facts for a question |

## Limitations

- The golden set is synthetic (template-generated, seed 7, see `evals/gen_golden.py`). It tests the pipeline and catches
  regressions; it does not measure performance on real customer data.
- Semantic matching at a 0.80 threshold can accept a near-miss or reject a correct paraphrase.
- The default store is SQLite + NumPy cosine search, sized for a demo; `STORE=postgres` uses PostgreSQL + pgvector (v0.3).
- Prompt-only small models miss many facts (3B recall 0.60); the fine-tuned 1.5B fixes most of it on this distribution, but its training data shares templates with the eval set (see the seen/unseen split).

## Stack

Python 3.11 · Unsloth / LoRA (PEFT) · Hugging Face · FastAPI · Pydantic · OpenAI-compatible client · Ollama (qwen2.5) · nomic-embed-text · NumPy · SQLite · PostgreSQL + pgvector · Kafka API (Redpanda, confluent-kafka) ·
pytest · GitHub Actions · Docker Compose · Streamlit
