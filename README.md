# Masters Thesis – Codebase Overview

Three components make up this project: endpoint anticipation model training/evaluation, the Unmute live voice server, and a RAG pipeline for FIT knowledge retrieval.

---

## `endpoint_anticipation/`

Trains and evaluates turn-endpoint anticipation models on conversational speech datasets (SpokenWOZ, Switchboard).

**Configs** – `configs/infer.yaml` controls inference: dataset paths, sampling rate, channel handling, model checkpoint location, threshold sweep range (`threshold_range`), and W&B logging. Training configs live under `configs/forecasting/` and follow the pattern `<model_type>/<run_name>.yaml`.

**Implementations** – Two feature encoder backbones are supported:
- *Mimi-based*: uses Kyutai's Mimi codec at 12.5 Hz as the acoustic feature extractor, fed into a Transformer or LSTM head. Run names follow the `*_mimi_12.5hz_*` convention.
- *NeMo-based*: uses NVIDIA NeMo features as the encoder backbone. Downloaded via `scripts/download_nemo.py` / `download_nemo.sh`.

**Metrics** – `run.py` drives the full train/infer loop via `load_run` and `load_data`. Inference sweeps a threshold range and computes turn-prediction accuracy with a configurable collar (`infer_accuracy_collar_frames`). Results are synced to W&B via `scripts/sync_to_wandb.py`. Evaluation merged across splits with `scripts/merge.py`.

---

## `unmute_server/`

A full-duplex voice assistant server forked from Kyutai Unmute, extended with endpoint anticipation, RAG, and offline/online evaluation tooling.

**Speculation** – `unmute/unmute_handler_speculative.py` implements speculative generation: when the anticipator model (port 8093) exceeds `ANTICIPATE_THRESHOLD` (default 0.5), LLM+TTS generation starts early and is buffered. If real VAD fires within the 960 ms anticipation window the buffer is committed; otherwise it is discarded. Speculative+RAG variants are in `unmute_handler_speculative_rag.py` and `unmute_handler_speculative_rag_online.py`.

**RAG integration** – `unmute/unmute_handler_rag.py` and `main_websocket_rag.py` wire a FAISS/Neo4j retrieval backend into the response pipeline. `unmute/main_websocket_speculative_rag.py` combines both.

**Offline evaluation** – `dockerless/build_eval_response_jsons.py` converts FIT RAG benchmark results into the `response_jsons/` format consumed by `eval_rag.py`. `dockerless/eval_latency_fit.py` measures TOR (turn output rate) and first-word latency against ground-truth input end times. 

**Key dockerless files** – `dockerless/` contains the full no-Docker deployment stack: `install.sh` + `setup_rust.sh` for environment setup, and `start_*.sh` scripts that launch each microservice independently (STT, LLM, TTS, endpointer, anticipator, RAG, backend, frontend, tunnel).

**`infer*` files** – Scripts for offline batch inference:
- `infer_ep.py` – runs the LSTM endpoint anticipation model over audio folders, scoring with jiwer/WER and saving per-file predictions.
- `infer_fit_voice.sh` / `infer_fit_voice_speculative.sh` / `infer_fit_voice_baseline.sh` – shell wrappers that invoke the voice inference pipeline on FIT benchmark audio under different conditions (standard, speculative, baseline).
- `infer_fdb.sh` – inference on the Full-Duplex-Bench dataset.
- `infer_humdial.sh` – inference on the HumDial multilingual dataset.
- `infer_better_setup.sh` – convenience wrapper with improved GPU allocation.

---

## `rag/`

A knowledge-graph + vector RAG pipeline over FIT university data (courses, personnel, publications, projects, groups).

**Data collection** – `scripts/scrape_FIT_data.py` crawls the FIT website using BeautifulSoup, guided by `data_configs/fit.yaml`, and saves extracted entities as JSON files. `scripts/normalise_text.py` canonicalises entity text before indexing.

**Building the RAG index** – `scripts/build_vector_chunks.py` converts scraped JSON into overlapping text chunks with metadata. `scripts/build_vector_index.py` embeds chunks and builds a FAISS index. `scripts/embed_into_neo4j.py` ingests entities and relationships into a Neo4j knowledge graph. `utils/kg_contract.py` defines the canonical entity schema and name normalisation used throughout.

**Retrieval** – `scripts/retrieve_from_question.py` and `scripts/query_vector_index.py` run top-k vector search over the FAISS index. `scripts/infer_basic_ctx_faiss.py` performs end-to-end question answering using retrieved context. `scripts/generate_from_question_and_retrieved.py` handles LLM generation given a question and retrieved passages.

**Benchmarks and metrics** – `scripts/generate_test_set.py` and `scripts/generate_test_set_from_llm.py` create evaluation question sets. `scripts/evaluate_retrieval_hit_rates.py` computes per-category retrieval hit rates (overall, courses, groups, personnel, projects, publications, longform). `scripts/evaluate_vector_benchmark_results.py` aggregates benchmark result files. `scripts/eval_rag.py` scores full RAG pipeline outputs (generation quality). `scripts/score_rag.py` provides summary scoring utilities. `scripts/validate_kg_contract.py` checks that all stored entities conform to the KG schema.
