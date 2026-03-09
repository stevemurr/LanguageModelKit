# LanguageModelKit

`LanguageModelKit` is a provider-neutral Swift package for text generation, structured generation, and stateful thread management across Apple Foundation Models, OpenAI-compatible APIs, and vLLM.

## Package Layout

- `LanguageModelRuntime`: shared runtime contracts, prompt rendering, token estimation seams, and HTTP/SSE helpers.
- `LanguageModelStructuredOutput`: portable structured-output API built on top of `LanguageModelStructuredCore`.
- `LanguageModelContextManagement`: logical thread management, persistence, budgeting, retrieval, bridge retries, and compaction.
- `LanguageModelApple`: Apple Foundation Models adapter.
- `LanguageModelOpenAI`: OpenAI Chat Completions adapter.
- `LanguageModelVLLM`: vLLM OpenAI-compatible adapter.

`LanguageModelStructuredCore` is an internal support target that owns the portable schema and structured-generation contracts shared by context management and adapter implementations.

## Endpoint Configuration

Each backend is selected with a `ModelEndpoint`.

```swift
let endpoint = ModelEndpoint(
    backendID: "openai",
    modelID: "gpt-4.1-mini",
    options: [
        "baseURL": "https://api.openai.com",
        "apiKey": "<token>"
    ]
)
```

### Apple

- `backendID`: `apple`
- `modelID`: `default`, `general`, or `contentTagging`
- `options["guardrails"]`: `default` or `permissiveContentTransformations`
- `options["adapter"]`: optional Apple adapter escape hatch for future/private model routing

Unknown Apple `modelID` values are rejected during availability and session creation.

### OpenAI

- `backendID`: `openai`
- `options["baseURL"]`: required; `/v1/chat/completions` is appended if needed
- `options["apiKey"]`: required
- `options["organization"]`: optional `OpenAI-Organization` header value
- `options["strictStructuredOutputs"]`: optional boolean-like string; defaults to `true`

Structured requests use Chat Completions `response_format.type = json_schema`.

### vLLM

- `backendID`: `vllm`
- `options["baseURL"]`: required; `/v1/chat/completions` is appended if needed
- `options["apiKey"]`: optional bearer token
- `options["guidedDecoding"]`: `auto`, `json`, or `grammar`

`auto` and `json` send `guided_json`. `grammar` sends `guided_grammar`.

## Structured Session Reuse

`LanguageModelContextManagement` distinguishes between two structured-generation paths:

- If `runtime.structuredOutput` is `nil` or exactly equal to `runtime.inference`, structured replies reuse the live inference session when the adapter supports package-only `LiveStructuredGenerationSession`.
- If `runtime.structuredOutput` points at a different endpoint, structured generation is executed statelessly through the configured `StructuredOutputBackend`.

That keeps same-backend structured calls on the active logical thread while preserving explicit cross-backend routing.

## Context Management Notes

- Text budgeting uses `runtime.inference`.
- Structured budgeting uses `runtime.structuredOutput ?? runtime.inference`.
- Compaction can downgrade from `structuredSummary`/`hybrid` to `slidingWindow`, and the downgrade reason is persisted in `CompactionReport`.
- Structured summary compaction uses chunked summaries and bounded recursive merges controlled by `chunkTargetTokens`, `chunkSummaryTargetTokens`, and `maxMergeDepth`.

## Adapter Limits

### Apple SDK Exceptions

The Apple adapter intentionally returns `unsupportedCapability` for these portable schema shapes because the current `FoundationModels` surface does not expose a clean translation target:

- top-level optional schemas
- string `minLength`
- string `maxLength`

These are documented exceptions rather than silent fallbacks.

### vLLM Grammar Mode

`guidedDecoding = grammar` supports the portable subset implemented by the translator:

- objects with fixed property order
- optional properties
- arrays
- booleans
- unconstrained strings
- unconstrained integers and numbers
- enums

It intentionally rejects shapes that cannot be lowered deterministically, including constrained strings, constrained integers, and constrained numbers.

## Testing

Run the full suite with:

```bash
swift test
```

Live integration smoke tests are env-gated. Supported environment variables:

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `VLLM_BASE_URL`
- `VLLM_API_KEY`
- `VLLM_MODEL`
