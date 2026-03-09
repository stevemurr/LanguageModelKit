# Spec: Modular Cross-Provider Runtime, Structured Output, and Context Management Library

## Summary
Build a new SwiftPM package from scratch that provides three independently adoptable layers:
- model runtime access
- structured output
- context management

The package must support Apple Foundation Models, OpenAI-compatible APIs, and vLLM. The core must be provider-agnostic. Context management must be optional. Structured output must be optional. Consumers must be able to use only the runtime layer, only runtime + structured output, only runtime + context management, or all three together.

V1 design goals:
- provider-neutral core contracts
- Apple/OpenAI/vLLM adapters implemented in the initial build
- no provider-neutral tool calling in v1
- final structured results only in the portable API
- no mid-thread model switching in v1

## Package Layout
Create one Swift package with these products and targets:

```text
Package.swift

Products
- LanguageModelRuntime
- LanguageModelStructuredOutput
- LanguageModelContextManagement
- LanguageModelApple
- LanguageModelOpenAI
- LanguageModelVLLM

Targets
- Sources/LanguageModelRuntime
- Sources/LanguageModelStructuredOutput
- Sources/LanguageModelContextManagement
- Sources/LanguageModelApple
- Sources/LanguageModelOpenAI
- Sources/LanguageModelVLLM

Tests
- Tests/LanguageModelRuntimeTests
- Tests/LanguageModelStructuredOutputTests
- Tests/LanguageModelContextManagementTests
- Tests/LanguageModelAppleTests
- Tests/LanguageModelOpenAITests
- Tests/LanguageModelVLLMTests
- Tests/LanguageModelIntegrationTests
```

Target dependencies:
- `LanguageModelStructuredOutput` depends on `LanguageModelRuntime`
- `LanguageModelContextManagement` depends on `LanguageModelRuntime`
- `LanguageModelApple` depends on `LanguageModelRuntime` and `LanguageModelStructuredOutput`
- `LanguageModelOpenAI` depends on `LanguageModelRuntime` and `LanguageModelStructuredOutput`
- `LanguageModelVLLM` depends on `LanguageModelRuntime` and `LanguageModelStructuredOutput`

Do not make `LanguageModelContextManagement` depend on `LanguageModelStructuredOutput` at the target level. Structured compaction support must be injected via capabilities and optional services.

## Public Core Contracts

### Runtime Layer
Expose these core types in `LanguageModelRuntime`:

```swift
public struct ModelEndpoint: Codable, Sendable, Equatable {
    public var backendID: String
    public var modelID: String
    public var options: [String: String]
    public var contextWindowOverride: Int?

    public init(
        backendID: String,
        modelID: String,
        options: [String: String] = [:],
        contextWindowOverride: Int? = nil
    )
}

public struct RuntimeCapabilities: Sendable, Equatable {
    public var supportsTextGeneration: Bool
    public var supportsTextStreaming: Bool
    public var supportsStructuredOutput: Bool
    public var supportsExactTokenEstimation: Bool
    public var supportsLocaleHints: Bool

    public init(...)
}

public struct RuntimeAvailability: Sendable, Equatable {
    public enum Status: Sendable, Equatable {
        case available
        case unavailable(reason: String)
    }

    public var status: Status
    public var capabilities: RuntimeCapabilities

    public init(status: Status, capabilities: RuntimeCapabilities)
}

public enum RuntimeError: Error, Sendable, Equatable {
    case unavailable(String)
    case unsupportedCapability(String)
    case unsupportedLocale(String)
    case contextOverflow(String)
    case refusal(String)
    case generationFailed(String)
    case transportFailed(String)
}
```

```swift
public protocol InferenceBackend: Sendable {
    var backendID: String { get }

    func availability(for endpoint: ModelEndpoint) async -> RuntimeAvailability
    func makeSession(
        endpoint: ModelEndpoint,
        instructions: String?,
        locale: Locale?
    ) async throws -> any InferenceSession
    func contextWindowTokens(for endpoint: ModelEndpoint) async -> Int?
    func exactTokenEstimator(for endpoint: ModelEndpoint) async -> (any TokenEstimating)?
}

public protocol InferenceSession: Sendable {
    func generateText(
        prompt: String,
        options: TextGenerationOptions
    ) async throws -> TextGenerationResult

    func streamText(
        prompt: String,
        options: TextGenerationOptions
    ) async -> AsyncThrowingStream<TextStreamEvent, Error>
}

public protocol TokenEstimating: Sendable {
    func estimate(
        prompt: RenderedPrompt,
        reservedOutputTokens: Int
    ) async -> TokenEstimate?
}
```

```swift
public struct TextGenerationOptions: Sendable, Equatable {
    public var maximumResponseTokens: Int?
    public var temperature: Double?
    public var deterministic: Bool

    public init(...)
}

public struct TextGenerationResult: Sendable, Equatable {
    public var text: String

    public init(text: String)
}

public enum TextStreamEvent: Sendable, Equatable {
    case partial(String)
    case completed(TextGenerationResult)
}
```

Add a registry:

```swift
public actor RuntimeRegistry {
    public init()
    public func register(_ backend: any InferenceBackend) async
    public func backend(for backendID: String) async throws -> any InferenceBackend
}
```

### Structured Output Layer
Expose these types in `LanguageModelStructuredOutput`:

```swift
public enum OutputSchema: Sendable, Equatable {
    case string(StringConstraints = .init())
    case integer(NumberConstraints<Int> = .init())
    case number(NumberConstraints<Double> = .init())
    case boolean
    case array(ArrayConstraints)
    case object(ObjectSchema)
    case enumeration(EnumSchema)
    case optional(OutputSchema)
}
```

```swift
public struct StringConstraints: Sendable, Equatable {
    public var regex: String?
    public var minLength: Int?
    public var maxLength: Int?
}

public struct NumberConstraints<T: Sendable & Equatable>: Sendable, Equatable {
    public var minimum: T?
    public var maximum: T?
}

public struct ArrayConstraints: Sendable, Equatable {
    public var item: OutputSchema
    public var minimumCount: Int?
    public var maximumCount: Int?
}

public struct ObjectSchema: Sendable, Equatable {
    public struct Property: Sendable, Equatable {
        public var name: String
        public var description: String?
        public var schema: OutputSchema
        public var isOptional: Bool
    }

    public var name: String
    public var description: String?
    public var properties: [Property]
}

public struct EnumSchema: Sendable, Equatable {
    public var name: String?
    public var cases: [String]
}
```

```swift
public struct StructuredOutputSpec<Value>: Sendable {
    public let schema: OutputSchema
    public let decode: @Sendable (Data) throws -> Value
    public let transcriptRenderer: @Sendable (Value) -> String

    public init(
        schema: OutputSchema,
        decode: @escaping @Sendable (Data) throws -> Value,
        transcriptRenderer: @escaping @Sendable (Value) -> String
    )
}
```

```swift
public protocol StructuredOutputBackend: Sendable {
    var backendID: String { get }

    func generateStructured<Value>(
        endpoint: ModelEndpoint,
        instructions: String?,
        locale: Locale?,
        prompt: String,
        spec: StructuredOutputSpec<Value>,
        options: StructuredGenerationOptions
    ) async throws -> StructuredGenerationResult<Value>
}
```

```swift
public struct StructuredGenerationOptions: Sendable, Equatable {
    public var maximumResponseTokens: Int?
    public var deterministic: Bool

    public init(...)
}

public struct StructuredGenerationResult<Value>: Sendable {
    public let value: Value
    public let transcriptText: String

    public init(value: Value, transcriptText: String)
}
```

Provide a built-in JSON helper:

```swift
public enum StructuredOutput {
    public static func codable<Value: Codable & Sendable>(
        _ type: Value.Type,
        schema: OutputSchema,
        renderTranscript: @escaping @Sendable (Value) -> String = { value in
            String(data: try! JSONEncoder().encode(value), encoding: .utf8) ?? "{}"
        }
    ) -> StructuredOutputSpec<Value>
}
```

V1 schema rules:
- support objects, arrays, strings, integers, numbers, booleans, enums, nullable/optional
- support regex, min/max, min/max count
- do not support arbitrary recursive schemas
- do not support heterogeneous unions other than optional/null
- do not support provider-specific guide features in the portable core

### Context Management Layer
Expose these types in `LanguageModelContextManagement`:

```swift
public struct ThreadRuntimeConfiguration: Codable, Sendable, Equatable {
    public var inference: ModelEndpoint
    public var structuredOutput: ModelEndpoint?

    public init(inference: ModelEndpoint, structuredOutput: ModelEndpoint? = nil)
}
```

```swift
public struct ContextManagerConfiguration: Sendable {
    public var runtimeRegistry: RuntimeRegistry
    public var structuredBackends: [String: any StructuredOutputBackend]
    public var budget: BudgetPolicy
    public var compaction: CompactionPolicy
    public var memory: MemoryPolicy
    public var persistence: PersistencePolicy
    public var diagnostics: DiagnosticsPolicy

    public init(...)
}
```

```swift
public struct ThreadConfiguration: Sendable {
    public var runtime: ThreadRuntimeConfiguration
    public var instructions: String?
    public var locale: Locale?

    public init(
        runtime: ThreadRuntimeConfiguration,
        instructions: String? = nil,
        locale: Locale? = nil
    )
}
```

```swift
public actor ContextManager {
    public init(configuration: ContextManagerConfiguration)

    public func session(
        id: String = UUID().uuidString,
        configuration: ThreadConfiguration
    ) async throws -> ContextSession

    public func availability(for endpoint: ModelEndpoint) async -> RuntimeAvailability
}
```

```swift
public struct ContextSession: Sendable {
    public let id: String

    public func respond(_ prompt: String) async throws -> String
    public func reply(to prompt: String) async throws -> TextReply
    public func stream(_ prompt: String) -> AsyncThrowingStream<TextStreamEvent, Error>

    public func generate<Value>(
        _ prompt: String,
        spec: StructuredOutputSpec<Value>
    ) async throws -> Value

    public func reply<Value>(
        to prompt: String,
        spec: StructuredOutputSpec<Value>
    ) async throws -> StructuredReply<Value>

    public var inspection: SessionInspection { get }
    public var maintenance: SessionMaintenance { get }
}
```

Portable structured output in context management is opt-in:
- if the caller never invokes structured APIs, structured backends are not required
- if the caller invokes structured APIs and no backend is configured, fail with `unsupportedCapability`
- if `runtime.structuredOutput` is nil, use `runtime.inference` for structured output if a structured backend exists for that backend ID

Preserve provider-neutral persistence seams:
- `ThreadStore`
- `MemoryStore`
- `BlobStore`
- `Retriever`

Persist thread state using provider-neutral configuration:
- thread ID
- instructions
- locale identifier
- `ThreadRuntimeConfiguration`
- active window index
- normalized turns
- diagnostics snapshots

## Adapter Implementations

### Apple Adapter
Implement `LanguageModelApple` with:
- `AppleInferenceBackend: InferenceBackend`
- `AppleStructuredOutputBackend: StructuredOutputBackend`

Behavior:
- use `LanguageModelSession` for text generation and text streaming
- translate `OutputSchema` into Apple `GenerationSchema`
- decode structured results through Apple native schema-guided generation
- map Apple availability, locale failures, refusal, overflow, and generation errors into `RuntimeError`
- expose exact token estimation if Apple runtime supports it, otherwise return nil
- use deterministic generation for structured compaction and structured output where possible

Add Apple-only convenience compatibility helpers:
- `GenerableStructuredOutputSpec<Value: Generable>`
- overloads to create `StructuredOutputSpec` from `Value.generationSchema`
- these helpers live only in `LanguageModelApple`
- do not expose `Generable` in the portable products

### OpenAI Adapter
Implement `LanguageModelOpenAI` with:
- `OpenAIInferenceBackend: InferenceBackend`
- `OpenAIStructuredOutputBackend: StructuredOutputBackend`

Behavior:
- target OpenAI-compatible `/v1/chat/completions`
- configuration is supplied through `ModelEndpoint.options`
- support text generation and SSE text streaming
- structured generation must use strict JSON-schema-style response format when available
- if the configured server/model does not support strict schema output, return `unsupportedCapability`
- decode final JSON bytes with the supplied `StructuredOutputSpec`

Required endpoint options:
- `baseURL`
- `apiKey`
- optional `organization`
- optional adapter-specific flags as strings

### vLLM Adapter
Implement `LanguageModelVLLM` with:
- `VLLMInferenceBackend: InferenceBackend`
- `VLLMStructuredOutputBackend: StructuredOutputBackend`

Behavior:
- use OpenAI-compatible transport for text generation
- structured generation must use vLLM guided decoding payloads
- translate `OutputSchema` to the supported vLLM guided JSON or grammar form
- if the requested schema cannot be expressed in vLLM guided decoding, return `unsupportedCapability`
- decode final JSON bytes with the supplied `StructuredOutputSpec`

Required endpoint options:
- `baseURL`
- optional `apiKey`
- optional guided decoding mode override

## Context Management Behavior
Port the current logical-thread design into provider-neutral form.

Required components:
- normalized turn model
- durable memory records
- blob spilling for oversized turns
- retrieval hook
- budget reporting
- compaction reporting
- bridge reporting
- rehydration from persisted thread state

Compaction modes:
- `slidingWindow`
- `structuredSummary`
- `hybrid`

Rules:
- sliding-window compaction is always available
- structured-summary compaction is available only when a structured backend is configured and supports the needed schema subset
- if structured-summary is configured but unavailable for a thread, downgrade automatically to sliding-window-only behavior
- log the downgrade in diagnostics
- never silently pretend structured compaction happened when it did not

Budgeting rules:
- use exact token estimation when the inference backend supplies it
- otherwise use heuristic estimation
- the context manager must not assume one provider tokenizer or transcript format
- the only required runtime budget input is context-window size
- if a backend cannot provide context-window size and no override is set, use `BudgetPolicy.defaultContextWindowTokens`

Bridging rules:
- one logical thread may span many runtime sessions
- one inference model is fixed per thread in v1
- no mid-thread inference backend change
- no mid-thread structured backend change
- bridge by rendering instructions + durable memory + recent tail into a new session seed
- overflow retries are bounded by `BudgetPolicy.maxBridgeRetries`

Structured request rules inside context management:
- if structured backend equals inference backend, use the same logical thread state and persist transcript text
- if structured backend differs from inference backend, render a fresh request snapshot from thread state and current prompt, call the structured backend statelessly, then persist the returned transcript text into the same logical thread
- canonical persisted transcript for structured responses is `spec.transcriptRenderer(value)`

## Implementation Notes
Use these internal seams:
- `InferenceBackend`
- `InferenceSession`
- `StructuredOutputBackend`
- `TokenEstimating`
- `ContextReducer`
- `ThreadStore`
- `MemoryStore`
- `BlobStore`
- `Retriever`

Do not reintroduce Apple-only types into core modules.
Do not put HTTP client code in the context management target.
Do not make the context layer depend directly on adapter targets.

Recommended internal file grouping:
- runtime types, backend registry, transport-neutral errors in `LanguageModelRuntime`
- schema types and translators in `LanguageModelStructuredOutput`
- compactor, bridge seed builder, budgeting, persistence, diagnostics in `LanguageModelContextManagement`

## Test Plan

### Runtime Tests
- backend registry resolves by `backendID`
- runtime-only text generation works with fake backends
- capability and availability mapping is deterministic
- transport failures are mapped to canonical runtime errors

### Structured Output Tests
- `OutputSchema` encodes/decodes correctly
- `StructuredOutputSpec` decodes valid JSON and rejects invalid JSON
- canonical transcript rendering is stable
- schema translation succeeds for supported Apple, OpenAI, and vLLM shapes
- unsupported schema features fail with `unsupportedCapability`

### Context Management Tests
- open thread, persist thread, rehydrate thread
- estimate budget with exact estimator and heuristic fallback
- respond text persists user and assistant turns
- overflow triggers bridge and retry
- retrieval and blob spilling work
- compaction reports reducers applied
- structured-summary compaction downgrades cleanly when structured output is unavailable
- structured responses persist rendered transcript text
- structured backend different from inference backend uses stateless structured call and still persists into the same thread

### Adapter Tests
- Apple adapter text and structured generation with fake or gated live tests
- OpenAI adapter text, streaming, and strict structured output request formation
- vLLM adapter text and guided decoding request formation
- error mapping for each adapter

### Integration Tests
- Apple live tests gated by platform availability
- OpenAI-compatible live tests gated by env vars:
  - `OPENAI_BASE_URL`
  - `OPENAI_API_KEY`
  - `OPENAI_MODEL`
- vLLM live tests gated by env vars:
  - `VLLM_BASE_URL`
  - `VLLM_MODEL`
  - optional `VLLM_API_KEY`
- all live tests skip cleanly when configuration is missing

## Acceptance Criteria
- A consumer can adopt `LanguageModelRuntime` alone and perform text generation without pulling in context management or structured output.
- A consumer can adopt runtime + structured output without pulling in context management.
- A consumer can adopt runtime + context management and use only plain text.
- A consumer can adopt all three layers and choose one backend for inference and a different backend for structured output.
- Apple-specific `Generable` usage is possible only by importing `LanguageModelApple`.
- No portable target imports `FoundationModels`.
- No portable target exposes provider-specific structured-output or tool types.
- All three adapters are implemented in v1.
- Tool calling is not implemented in the portable design.

## Assumptions and Defaults
- Swift 6.2 is the language baseline.
- Platform support is whatever is required by the Apple adapter; non-Apple targets should remain as portable as possible within Swift Foundation networking constraints.
- The portable structured-output API returns final values only in v1.
- Tool calling is deferred.
- Mid-thread backend switching is deferred.
- The portable schema subset is intentionally limited to what can be mapped cleanly across Apple, OpenAI-compatible endpoints, and vLLM.

