You are helping me develop a small local AI chatbot called KayenAI.

## Project goal

Build a local conversational AI system around a ~500M parameter decoder-only Transformer.

The system must run on a computer with 8 GB RAM.

The project should prioritize:

* low RAM usage
* CPU-friendly inference
* modular C/C++ architecture
* streaming datasets
* checkpointing
* deterministic and testable components
* simple deployment

## Model

Target approximately 500M parameters.

Use a decoder-only Transformer with:

* RoPE
* RMSNorm
* SwiGLU
* GQA where practical
* BPE tokenizer
* configurable context length
* FP16/BF16 training where supported
* INT8/INT4 quantized inference as an optimization

The model must be trained as a conversational/instruction-following chatbot.

It should learn interactions such as:

* "I want to build..."
* "I need..."
* "I was doing X and Y happened..."
* questions
* follow-up questions
* multi-turn conversations
* requests for explanations
* requests involving previous context

## Context system

Implement a conversation manager that keeps the complete usable current conversation inside the model context.

Implement long-term memory separately using embeddings and vector search.

Do not blindly insert all stored memories into every prompt.

Instead:

user message
→ embedding
→ vector search
→ retrieve relevant memories
→ context builder
→ Transformer

## Tools

Tools must NOT be directly controlled by the model.

The model may generate a structured tool request.

The C++ runtime must validate the request and check permissions before executing anything.

Initial tools:

* web_search
* optional file_read
* optional file_write
* optional command execution

Every tool must have an explicit permission state.

Default state must be disabled.

The runtime must be able to reject a tool request without crashing or corrupting the conversation.

Support:

* deny
* allow once
* allow for current conversation
* permanently enabled/disabled configuration

The model must never bypass the permission manager.

## Web search

Web search must be implemented as an external tool rather than being encoded into the model weights.

The tool returns structured results containing at least:

* title
* URL
* text/snippet

The model then receives the results as tool output and generates the final answer.

## Inference engine

Implement the inference engine in C++.

Do not require Python for normal inference.

Separate the code into components such as:

* tensor
* tokenizer
* model
* transformer
* attention
* RoPE
* KV cache
* sampler
* embeddings
* vector index
* conversation manager
* context builder
* tool dispatcher
* permission manager
* inference API
* CLI/GUI

Python may be used for training utilities, dataset preprocessing and experiments, but the final inference runtime must be C++.

## Memory constraints

The machine has 8 GB RAM.

Never assume that the entire dataset can be loaded into RAM.

Training must support:

* streaming/sharded datasets
* memory mapping where appropriate
* small batches
* gradient accumulation
* checkpointing
* resuming interrupted training

Avoid unnecessary copies of tensors and strings.

## Dataset

Dataset size is flexible.

Quality is more important than reaching 500 GB.

Implement preprocessing capable of handling datasets much larger than available RAM.

## GUI

Create a simple interactive chat interface.

The UI should expose:

* conversation
* generation status
* context/token information
* enabled tools
* tool requests
* permission prompts
* retrieved memories
* model settings

Keep the UI separate from the inference engine.

## Engineering rules

Prefer simple implementations over unnecessary frameworks.

Use RAII and modern C++.

Avoid global mutable state.

Use clear ownership semantics.

Write unit tests for:

* tokenizer
* tensors
* attention
* KV cache
* vector search
* context construction
* tool permissions
* tool dispatch
* serialization

Do not implement the entire system at once.

Build it incrementally:

1. tokenizer
2. tensor library
3. Transformer forward pass
4. training loop
5. checkpoint format
6. inference engine
7. conversation manager
8. embeddings/vector memory
9. tool dispatcher
10. permission system
11. web search
12. GUI

For every implementation step, explain the architecture briefly and then produce the necessary files/code.

Do not silently replace a requested component with an external AI runtime.

The goal is to understand and control the complete system.
