You are helping me develop a small local AI chatbot called KayenAI.

## Project goal

Build a local conversational AI system around a ~500M parameter decoder-only Transformer.

The system must run on a computer with 8 GB RAM as the primary target.

The project should prioritize:

* low RAM usage (target: <6 GB peak during inference)
* CPU-friendly inference (no CUDA requirement)
* modular C/C++ architecture with clear component boundaries
* streaming datasets (never assume full dataset fits in RAM)
* checkpointing and resume capability
* deterministic and testable components
* simple deployment (single binary for inference)

## Model Architecture

**Target:** approximately 500M parameters (~2 GB FP32 weights, ~1 GB FP16)

**Decoder-only Transformer with:**

* **Embedding Layer:** Token embeddings + positional encoding (RoPE)
* **Attention Mechanism:**
  - Multi-head self-attention with grouped query attention (GQA) for efficiency
  - Attention compute: O(n²) per layer, mitigated by KV cache in inference
  - Optional flash attention optimization
* **Normalization:** RMSNorm (before/after residuals per layer)
* **MLP:** SwiGLU activation (gating mechanism for better expressiveness)
* **Layer Configuration:**
  - Hidden size: 1024
  - Number of layers: 16-24 (configurable)
  - Attention heads: 8-16 (configurable)
  - Head dimension: 128
  - MLP intermediate: 4x hidden size (4096)
  - Context length: 4096 tokens (configurable)
* **Precision:**
  - Training: FP32 default, FP16/BF16 supported (with loss scaling)
  - Inference: FP32 primary, INT8/INT4 optional for further compression
* **BPE Tokenizer:**
  - Vocabulary size: ~32K tokens
  - Merges learned from training corpus
  - Support for special tokens: `<bos>`, `<eos>`, `<pad>`, `<unk>`

**Training Configuration:**
- Conversational/instruction-following style
- Learning patterns:
  - "I want to build..."
  - "I need help with..."
  - "I was doing X and Y happened..."
  - Questions and follow-up questions
  - Multi-turn conversation context
  - Requests for explanations
  - Requests involving previous context

## Context & Memory System

**Short-term Memory (Current Conversation):**
- Conversation manager maintains complete turn history in model context
- Dynamic context packing: fit as many complete turns as possible within token limit
- Strategy:
  1. Always include current user query (full message)
  2. Include assistant response to previous turn
  3. Backfill earlier turns (oldest first) until token budget exhausted
  4. Maintain conversation continuity without hard cutoffs

**Long-term Memory (Persistent):**
- Separate embeddings index (FAISS or similar) for memory retrieval
- Memory retrieval pipeline:
  1. User message → embed
  2. Vector search top-K relevant memories
  3. Rerank by recency and relevance
  4. Insert top-N into context before forward pass
- Do NOT blindly insert all stored memories into every prompt
- Configurable memory insertion (disabled by default, enable per-conversation)
- Memory storage:
  - Each turn creates memories for important facts, user preferences, context
  - Automatic: extract entities, goals, constraints
  - Manual: user-provided annotations
  - Decay: old memories weighted lower, can be pruned

**Context Builder:**
- Constructs model input from:
  - System prompt (static)
  - Retrieved memories (optional, top-N)
  - Current conversation turns (dynamic)
  - Status/metadata (tokens used, memory count)
- Token accounting: track exact usage before inference

## Tool System (Structured Tool Dispatch)

**Principle:** Tools are NOT directly controlled by the model. The model generates structured tool requests only.

**Permission Model:**
The C++ runtime validates before executing anything. Support permission states:
- `deny` - tool permanently blocked
- `allow-once` - single execution, then revert to deny
- `allow-session` - enabled for current conversation only
- `permanent` - configured in settings (survives session restarts)

**Validation Flow:**
```
Model Output (structured JSON tool request)
    ↓
Permission Manager (check state)
    ├─ If denied: return denial + context to model
    └─ If allowed:
        ↓
        Tool Executor (runtime safety checks)
            ├─ Validate arguments
            ├─ Check resource limits
            └─ Execute or reject
```

**The model must never bypass the permission manager.** Rejection must be safe and non-fatal.

**Initial Tools:**

1. **web_search** (requires permission)
   - Input: query (string)
   - Output: {title, URL, snippet} array
   - Limits: max 5 results per request, timeouts at 10s
   - Returns structured results only

2. **file_read** (optional, requires permission)
   - Input: path (string)
   - Output: file contents as string
   - Limits: max 1 MB per file, no traversal above project root
   - Validates path safety

3. **file_write** (optional, requires permission)
   - Input: path, contents
   - Output: success/error
   - Limits: no overwrite of system files, max 10 MB
   - Requires explicit user confirmation for new files

4. **command_execution** (optional, dangerous, requires explicit permission)
   - Input: command string
   - Output: stdout/stderr as text
   - Limits: 30s timeout, restricted shell (no sudo, rm -rf, etc.)
   - Requires allow-once permission per command

**Tool Request Format (JSON):**
```json
{
  "type": "tool_request",
  "tool": "web_search",
  "arguments": {"query": "how to install rust"},
  "id": "req_uuid"
}
```

**Tool Response (back to model as context):**
```json
{
  "type": "tool_response",
  "tool": "web_search",
  "request_id": "req_uuid",
  "status": "success",
  "result": [{"title": "...", "url": "...", "snippet": "..."}]
}
```

## Inference Engine

Implement the inference engine in C++ (no Python required for normal inference).

**Architecture Components:**

```
Input Tokens
    ↓
Embedding Layer (token + position)
    ↓
Transformer Blocks (×N layers)
    ├─ Multi-Head Attention (with KV cache)
    ├─ Feed-Forward (SwiGLU)
    └─ RMSNorm + Residuals
    ↓
Output Logits
    ↓
Sampler (temperature, top-k, top-p)
    ↓
Next Token
```

**KV Cache Management:**
- Allocate once for max context length
- Reuse across inference steps
- Dynamic shape: grows with sequence length
- Circular buffer pattern for efficiency

**Memory Budgets (8 GB target machine):**
- Model weights: ~1-2 GB (FP32)
- KV cache: ~1 GB (FP16 for 4K context)
- Activations: ~0.5-1 GB (per batch)
- Embeddings index: ~1 GB
- Overhead & OS: ~2-3 GB
- **Peak inference: <6 GB RAM**

**Inference API:**
```cpp
// Initialize once
Transformer model = load_checkpoint("model.bin");
KVCache cache(max_context=4096);

// Per generation step
vector<int> tokens = {<input_tokens>};
Tensor logits = model.forward(tokens, cache);
int next_token = sample(logits, temperature=0.7);
```

**Component Separation:**
- `src/tensor/` - Tensor ops and memory management
- `src/tokenizer/` - BPE tokenizer
- `src/model/` - Transformer architecture
- `src/attention/` - Attention + KV cache
- `src/sampler/` - Token sampling strategies
- `src/inference/` - Main inference loop
- `src/conversation/` - Context/conversation manager
- `src/embeddings/` - Embedding model + vector search
- `src/tools/` - Tool dispatcher
- `src/permissions/` - Permission manager
- `src/cli/` - Command-line interface
- `src/gui/` - GUI (separate from inference)

**Python Usage (Training/Preprocessing Only):**
- `training/` - Training loop, optimization
- `data/` - Dataset loading, preprocessing, streaming
- Separate from inference binary

## Memory & Dataset Constraints

The machine has **8 GB RAM as primary target.**

**Never assume the entire dataset can be loaded into RAM.**

**Training Dataset Requirements:**
- Quality prioritized over size (aim for curated, high-quality data)
- Expected dataset: ~1-10 GB (can be larger with streaming)
- Format: JSONL (one example per line) or Parquet for efficient loading
- Example structure:
  ```json
  {"instruction": "...", "input": "...", "output": "...", "metadata": {...}}
  ```

**Streaming & Sharding:**
- Implement data loaders that fetch batches from disk on demand
- Support disk-mapped datasets (mmap) for deterministic sampling
- Shard datasets across multiple files for parallel loading
- Support dataset versioning (checkpointing data state)

**Preprocessing Pipeline:**
- Tokenization in batches (convert text → tokens)
- Packing strategy: combine multiple examples into single sequence (with separator)
- Validation: ensure no OOM during preprocessing
- Caching: precomputed token sequences stored efficiently

**Training Memory Budget:**
- Batch size: typically 4-8 (fit in 8 GB)
- Gradient accumulation: simulate larger batches without extra memory
- Checkpointing: recompute activations during backprop to save memory
- Mixed precision: FP16 forward/backward, FP32 model parameters

**Checkpoint/Resume:**
- Save every N steps:
  - Model weights (FP32 master copy)
  - Optimizer state (Adam moments)
  - Training metadata (step, loss, dataset offset)
- Resume: restore exact state to continue training deterministically
- Support recovery from interrupted training (power loss, timeout, etc.)

**Avoid:**
- Loading entire dataset at startup
- Unnecessary copies of tensors/strings
- Keeping full gradient history in memory
- Non-streaming evaluation on large test sets

## User Interface

**GUI Design Principles:**
- Separate from inference engine (communicate via API/IPC)
- Real-time updates without blocking inference
- Responsive, minimal resource usage
- Local deployment (no network calls to render UI)

**GUI Components:**
1. **Conversation Panel:**
   - Turn-by-turn history with user/assistant distinction
   - Streaming response display (token-by-token as generated)
   - Context scrollback
   
2. **Generation Status:**
   - Current token index / max tokens
   - Generation speed (tokens/sec)
   - Temperature and sampling parameters
   - Stop reason (end-of-sequence, max tokens, etc.)

3. **Context Information:**
   - Total tokens used (prompt + generated)
   - Memory usage (model, activations, cache)
   - Time per step
   - Available context window remaining

4. **Tool Integration:**
   - Permission prompt overlay (user approval for tools)
   - Tool request/response log
   - Status: pending, denied, executing, completed
   - Tool output formatted in conversation

5. **Memory/Retrieval:**
   - Retrieved memories display (if enabled)
   - Memory count and relevance scores
   - Option to manually annotate/save memories

6. **Model Settings:**
   - Temperature slider (0.0 - 2.0)
   - Top-k and top-p controls
   - Max tokens to generate
   - System prompt editor

**CLI Interface:**
- Lightweight, single-binary deployment
- Usage: `./kayenai --prompt "Your question" [options]`
- Options:
  - `--model` - checkpoint path
  - `--context` - context length
  - `--temperature` - sampling temperature
  - `--top-k`, `--top-p` - sampling controls
  - `--stream` - stream output vs. wait
  - `--memory` - enable/disable long-term memory
  - `--tools` - enable specific tools
  - Interactive REPL mode if no prompt provided

## Engineering Rules & Best Practices

**Code Principles:**
- Prefer simple implementations over unnecessary frameworks
- Use RAII (Resource Acquisition Is Initialization) for memory safety
- Avoid global mutable state; use clear ownership semantics
- Modern C++17: smart pointers (`unique_ptr`, `shared_ptr`), `optional`, `variant`
- No silent replacement of components with external AI runtimes

**Testing Strategy:**
Unit tests required for:
- Tokenizer (correct BPE merging, edge cases)
- Tensor operations (shape, dtype, memory layout)
- Attention mechanisms (correctness against reference, KV cache consistency)
- KV cache (reuse, cleanup, overflow handling)
- Vector search (retrieval accuracy, index building)
- Context construction (token counting, turn packing)
- Tool permissions (all 4 states, bypass attempts)
- Tool dispatch (safe rejection, error recovery)
- Serialization (checkpoints roundtrip correctly)

**Determinism & Reproducibility:**
- Fixed random seeds for initialization
- Bitwise deterministic inference (no floating-point accumulation order changes)
- Unit tests with fixed inputs produce fixed outputs
- Training resumable from any checkpoint with identical loss curves

**Build & Dependencies:**
- Build system: CMake (C++17 minimum)
- Dependencies: minimal (no TensorFlow, PyTorch in C++ code)
- Single binary output for inference
- Optional Python tools for training (separate build)
- Support both debug (-g) and optimized (-O2/-O3) builds

**File Organization:**
```
src/
  tensor/      → Tensor class, operations, memory management
  tokenizer/   → BPE tokenizer
  model/       → Model loading, config
  attention/   → Attention, RoPE, KV cache
  sampler/     → Sampling strategies
  inference/   → Main inference loop
  conversation/ → Context manager, turn history
  embeddings/  → Embedding model, vector search
  tools/       → Tool dispatcher
  permissions/ → Permission manager
  cli/         → Command-line interface
  gui/         → GUI (separate, optional)
training/      → Training scripts (Python)
data/          → Preprocessing utilities
tests/         → Unit tests
models/        → Checkpoint storage
build/         → Build artifacts (gitignored)
CMakeLists.txt
README.md
prompt.md
```

**Naming Conventions:**
- Classes: PascalCase (e.g., `Tensor`, `Tokenizer`, `TransformerBlock`)
- Functions: snake_case (e.g., `compute_attention`, `load_checkpoint`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_CONTEXT_LENGTH`)
- Private members: `_leading_underscore`
- Namespaces: `kayenai::tensor`, `kayenai::model`, etc.

**Error Handling:**
- Use exceptions for non-recoverable errors
- Return `std::optional<T>` or `std::variant<T, Error>` for expected failures
- All exceptions must have clear error messages (no silent fails)
- Fail fast: detect corrupted checkpoints immediately

## Development Roadmap

**Build Incrementally (Validated at Each Stage):**

1. ✅ **Tensor Library** - Memory-efficient operations (DONE)
   - Shape tracking, strides, multiple dtypes
   - Element-wise ops, matmul, reductions
   - Unit tests for all operations

2. **Tokenizer (BPE)** - Text → Tokens
   - BPE merge learning from corpus
   - Encode/decode with special tokens
   - Deterministic merging
   - Unit tests: edge cases, special chars, reproducibility

3. **Transformer Forward Pass** - Model architecture
   - RoPE positional encoding
   - Multi-head attention (simplified, then GQA)
   - RMSNorm layers
   - SwiGLU MLP
   - Residual connections
   - Verify forward pass correctness vs. reference

4. **Training Loop** - Iterative optimization
   - Adam optimizer
   - Loss computation (cross-entropy)
   - Gradient accumulation for smaller batches
   - Learning rate scheduling (cosine annealing)
   - Checkpoint saving/loading

5. **Checkpoint Format** - Persistence
   - Save model weights (FP32 master)
   - Save optimizer state
   - Save training metadata
   - Efficient binary format (no JSON overhead)
   - Resume from checkpoint with deterministic continuation

6. **KV Cache** - Efficient inference
   - Allocate once, reuse across steps
   - Update mechanism (append new key/value)
   - Prevent memory leaks
   - Benchmark: memory usage, speed improvement

7. **Inference Engine** - Text generation
   - Token-by-token generation
   - Integrate tokenizer + model + KV cache
   - Sampler integration
   - Clean inference API

8. **Conversation Manager** - Context handling
   - Track turn history
   - Dynamic context packing (fit maximum turns)
   - Token accounting
   - Clear on new conversation

9. **Embeddings & Vector Index** - Long-term memory
   - Simple embedding model (TinyBERT or similar)
   - Vector search (FAISS or in-memory)
   - Memory storage format
   - Retrieval pipeline

10. **Tool Dispatcher** - Safe tool execution
    - Structured tool request parsing (JSON)
    - Tool registry and execution
    - Resource limits + timeouts
    - Error recovery

11. **Permission Manager** - Access control
    - 4 permission states: deny, allow-once, allow-session, permanent
    - User interaction for approval
    - Configuration storage
    - Non-bypassable enforcement

12. **Web Search Tool** - External search
    - Query formatting
    - Result aggregation
    - Structured output
    - Integration with context builder

13. **CLI Interface** - Command-line usage
    - Single-binary deployment
    - Argument parsing
    - Interactive REPL mode
    - Streaming output

14. **GUI** - Interactive interface (optional, lower priority)
    - Chat interface
    - Real-time status display
    - Tool/permission management
    - Separate process from inference

**Validation Criteria:**
- Each component passes unit tests before integration
- Incremental tests verify integration (e.g., tokenizer + model forward pass)
- Deterministic outputs for reproducibility
- No regressions in memory or latency

## Performance Targets

**Inference (8 GB RAM target machine):**
- Time-to-first-token: <500ms
- Tokens/sec: 5-10 (CPU-friendly speed)
- Peak memory: <6 GB total
- Model weights: 1-2 GB (FP32), 0.5-1 GB (FP16)

**Training:**
- Batch size: 4-8 (fits in 8 GB)
- Steps/hour: ~500-1000 (depends on data quality, not quantity)
- Convergence: 10K-100K steps to reasonable loss (quality-dependent)

**Memory Profiling:**
- Per-component tracking (model, cache, embeddings, buffer pools)
- No memory leaks during long inference sessions
- Graceful OOM handling (errors instead of crashes)
