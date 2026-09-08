# Copilot Instructions for KayenAI

## Project Overview

KayenAI is a **local conversational AI system** built around a ~500M parameter decoder-only Transformer. It prioritizes low RAM usage (8 GB target), CPU-friendly inference, and a modular C/C++ architecture. The model is conversational/instruction-following and runs without requiring external APIs.

See `prompt.md` for the full specification.

## Architecture

The project consists of several interconnected components that should be developed incrementally:

### Core Pipeline
1. **Tokenizer** (BPE) → Input tokens
2. **Tensor Library** → Memory-efficient operations (FP16/BF16 support)
3. **Transformer Model** → Decoder-only with RoPE, RMSNorm, SwiGLU, GQA
4. **KV Cache** → Efficient attention computation
5. **Sampler** → Token generation strategies

### System Components
- **Inference Engine** (C++ primary runtime, no Python required for inference)
- **Training Loop** (Python allowed for training utilities, dataset preprocessing)
- **Checkpoint Format** → Resume training and inference from checkpoints
- **Conversation Manager** → Keeps complete conversation in model context
- **Embeddings & Vector Index** → Long-term memory via vector search
- **Tool Dispatcher** → Validates and routes tool requests from model
- **Permission Manager** → Explicit permission states (deny, allow-once, allow-session, permanent)
- **GUI** → Interactive chat interface separate from inference engine
- **CLI** → Command-line interface for inference

### Tool System
- Tools are **NOT directly controlled by the model** - model generates structured requests only
- C++ runtime validates before execution
- Initial tools: `web_search`, `file_read`, `file_write`, `command_execution` (all permission-controlled)
- Tool results returned to model as context for answer generation

## Key Architectural Patterns

### Memory Constraints
- Assume 8 GB RAM at all times
- Datasets larger than RAM: use streaming/sharding, memory mapping, small batches
- Avoid unnecessary tensor/string copies
- Support gradient accumulation and checkpointing

### Modularity & Testing
- Clear ownership semantics, minimize global state
- Use RAII and modern C++
- **Unit test these components:**
  - Tokenizer
  - Tensors
  - Attention & KV cache
  - Vector search
  - Context construction
  - Tool permissions & dispatch
  - Serialization

### Model Architecture Details
- **Decoder-only Transformer** (~500M parameters)
- **Positional Encoding:** RoPE (Rotary Position Embeddings)
- **Normalization:** RMSNorm
- **MLP:** SwiGLU activation
- **Attention:** GQA (Grouped Query Attention) where practical
- **Training:** FP16/BF16 support
- **Inference:** INT8/INT4 quantization optional

### Training & Data
- Quality prioritized over dataset size (target flexible)
- Support streaming large datasets without full RAM load
- Include data preprocessing utilities in Python
- Implement checkpoint/resume from interrupted training

## Build, Test & Development

### Language Split
- **C++:** Inference engine, model, tensors, tokenizer, KV cache, vector search, tool dispatcher, permission manager
- **Python:** Training utilities, dataset preprocessing, experiments (not required for inference)
- **GUI:** Separate from inference (language TBD)

### No Existing Build System Yet
- Establish Makefile or CMake conventions early
- Include targets for: `build`, `test`, `clean`
- Build individual components independently

### Testing Strategy
- Unit tests for critical components (list above)
- Deterministic components verified with unit tests
- Training steps validate against fixed data samples

## Development Order

Build incrementally in this sequence (from `prompt.md`):
1. Tokenizer (BPE)
2. Tensor library
3. Transformer forward pass
4. Training loop
5. Checkpoint format
6. Inference engine
7. Conversation manager
8. Embeddings/vector memory
9. Tool dispatcher
10. Permission system
11. Web search tool
12. GUI

## Code Conventions

### Style & Quality
- Prefer simple implementations over unnecessary frameworks
- Avoid silently replacing requested components with external AI runtimes
- Explain architecture briefly before producing code
- All functionality must be controlled and understood by developers

### File Organization
- Logically group components (e.g., `src/tensor/`, `src/model/`, `src/tools/`)
- Keep headers clean; implementation in .cpp files
- Separate inference API from CLI/GUI

### Naming
- Clear, descriptive names for types and functions
- Modern C++: use `std::unique_ptr`, `std::optional`, RAII patterns
- Prefix internal/private members appropriately

## Dependencies & Tools

### Required Tools
- C++ compiler (C++17 or later)
- CMake or Make (TBD)
- Git

### Python (Training/Preprocessing Only)
- NumPy/PyTorch for data processing
- Specific versions TBD (add to requirements.txt when established)

### Avoid
- Heavy frameworks (TensorFlow, PyTorch) in C++ inference (implement core math in-house)
- External AI runtimes for model execution (build from scratch)

## Permission & Tool System Reference

Tools require explicit permission states. The model **cannot bypass** the permission manager.

**Permission States:**
- `deny` - tool blocked
- `allow-once` - single execution
- `allow-session` - enabled for current conversation
- `permanent` - configured in settings

**Validation Flow:**
```
Model Output (structured tool request)
  → Permission Manager (check state)
    → If allowed: Tool Executor
      → Runtime Validation (safety checks)
        → Execute or Reject
    → If denied: Return denial + context to model
```

## Quick Reference: Workspace Structure (Target)

```
KayenAI/
├── src/
│   ├── tokenizer/       # BPE tokenizer
│   ├── tensor/          # Tensor operations & memory management
│   ├── model/           # Transformer architecture
│   ├── attention/       # Attention mechanisms, KV cache
│   ├── sampler/         # Token sampling strategies
│   ├── inference/       # Main inference engine
│   ├── conversation/    # Context manager & conversation history
│   ├── embeddings/      # Embedding model & vector search
│   ├── tools/           # Tool dispatcher & execution
│   ├── permissions/     # Permission manager
│   ├── cli/             # Command-line interface
│   └── gui/             # GUI (separate from inference)
├── training/            # Training utilities (Python)
├── data/                # Dataset preprocessing
├── tests/               # Unit tests
├── models/              # Pretrained checkpoints
├── CMakeLists.txt       # Build configuration
├── prompt.md            # Project specification
└── README.md            # Usage & setup
```

## When Creating New Components

1. **Define Clear Boundaries** - What inputs/outputs, what it owns, what it doesn't
2. **Plan Unit Tests First** - What cases must pass?
3. **Document Assumptions** - RAM usage, precision, performance targets
4. **Incremental Integration** - Test component in isolation before adding to pipeline
5. **No Frameworks Until Proven Necessary** - Start simple, add abstractions if repeated patterns emerge

## Context for Common Tasks

### Adding a New Tool
- Extend `ToolRequest` struct with new tool type
- Add to `PermissionManager` validation
- Implement tool executor in `ToolDispatcher`
- Add tests for permission states & rejection paths
- Model receives tool result as context, not tool capability

### Expanding Attention Mechanisms
- Implement in `src/attention/` with unit tests
- Benchmark against baseline (measure RAM & latency)
- Update `Transformer` to support variant via config
- Ensure INT8/INT4 quantization remains compatible

### Dataset Pipeline Changes
- Streaming in `data/` must never assume full RAM load
- Checkpoint at logical boundaries during training
- Test preprocessing separately from training loop
- Document max batch size for 8 GB machine

---

**Last Updated:** Initial specification from prompt.md
**Architecture Version:** 1.0 (pre-development)
