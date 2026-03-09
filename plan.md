# LanguageModelKit Execution Tracker

## Completed Baseline
- [x] SwiftPM package scaffold with the six public products and the internal `LanguageModelStructuredCore` target.
- [x] Provider-neutral runtime contracts, registry, prompt rendering, token estimation surface, and shared HTTP/SSE helpers.
- [x] Portable structured-output contracts with `OutputSchema`, Codable round-trips, and `StructuredOutput.codable`.
- [x] Provider-neutral context-management layer with persistence seams, diagnostics, budgeting, bridging, and optional structured generation.
- [x] Initial Apple, OpenAI, and vLLM adapters for text generation plus first-pass structured output support.
- [x] Initial test coverage across runtime, structured output, context management, adapters, and env-gated integration smoke tests.

## SDK Exceptions
- [x] Apple adapter keeps `unsupportedCapability` for top-level nullable schemas.
- [x] Apple adapter keeps `unsupportedCapability` for string `minLength` and `maxLength`.
- [x] README and tests document these Apple-specific exceptions explicitly.

## Remaining Work

### Structured Session and Runtime Parity
- [x] Add a package-only live structured-session hook in `LanguageModelStructuredCore`.
- [x] Adopt the hook in Apple, OpenAI, and vLLM inference sessions where the adapter can perform stateful structured generation.
- [x] Update context management so `runtime.inference == (runtime.structuredOutput ?? runtime.inference)` reuses the live session for structured replies.
- [x] Keep cross-endpoint structured generation stateless through `StructuredOutputBackend`.

### Budgeting, Diagnostics, and Compaction
- [x] Budget structured requests against the active structured endpoint, not always the inference endpoint.
- [x] Replace portable-schema enum encoding in budgeting with stable rendered schema text.
- [x] Extend `CompactionReport` to persist requested mode, effective mode, and downgrade reason.
- [x] Persist structured-summary downgrade state through reply metadata, diagnostics, and rehydration.
- [x] Refactor compaction to use reducer types instead of only hardcoded branching.
- [x] Make `chunkTargetTokens`, `chunkSummaryTargetTokens`, and `maxMergeDepth` active in structured summary compaction.
- [x] Add chunked and hierarchical structured summary reduction before sliding-tail and emergency fallback.

### Adapter Cleanup
- [x] Reject unknown Apple `modelID` values during availability/session creation instead of silently falling back.
- [x] Implement vLLM `guidedDecoding=grammar` using a portable-schema-to-grammar translator.
- [x] Keep deterministic `unsupportedCapability` failures for portable schema features the grammar translator cannot lower.
- [x] Harden OpenAI completion and streaming error mapping to canonical `RuntimeError` cases.
- [x] Harden vLLM completion and streaming error mapping to canonical `RuntimeError` cases.

### Tests and Docs
- [x] Expand runtime tests for exact-estimator fallback, context-window fallback, transport mapping, and SSE error propagation.
- [x] Expand Apple tests for unknown models and documented unsupported schema cases.
- [x] Expand OpenAI tests for strict-schema rejection, refusal/content-filter mapping, and streaming behavior.
- [x] Expand vLLM tests for grammar mode, unsupported grammar shapes, guided-decoding rejection, and streaming behavior.
- [x] Expand context tests for rehydration, retrieval, blob spilling, file-backed stores, manual compaction, downgrade persistence, same-endpoint structured reuse, cross-endpoint stateless structured calls, structured budgeting, and streaming persistence.
- [x] Expand integration smoke tests for live structured output and streaming where configuration is available.
- [x] Add a top-level README describing the package layers, endpoint options, adapter limits, live structured reuse semantics, and Apple SDK exceptions.

## Checklist
- [x] Pass 1: structured-session/runtime parity
- [x] Pass 2: context/budgeting fixes
- [x] Pass 3: compaction and diagnostics hardening
- [x] Pass 4: adapter cleanup
- [x] Pass 5: tests and documentation
- [x] Final verification with `swift test`

## Verification
| Area | Status | Notes |
| --- | --- | --- |
| Runtime | Done | Transport mapping, SSE parsing, and budgeting fallbacks are covered. |
| Structured Output | Done | Portable schema round-trips and adapter-specific lowering failures are covered. |
| Context Management | Done | Rehydration, retrieval, blob spill, downgrade persistence, same-endpoint reuse, and streaming persistence are covered. |
| Apple Adapter | Done | Unknown-model rejection and documented SDK exceptions are covered. |
| OpenAI Adapter | Done | Strict schema rejection, refusal mapping, and streaming behavior are covered. |
| vLLM Adapter | Done | Grammar translation, deterministic unsupported shapes, refusal mapping, and streaming behavior are covered. |
| Integration | Done | Env-gated text, streaming, and structured-output smoke tests are in place. |
