import Foundation
import XCTest
@testable import LanguageModelContextManagement
@testable import LanguageModelStructuredOutput

final class ContextManagementXCTests: XCTestCase {
    func testTextReplyPersistsTurns() async throws {
        let registry = RuntimeRegistry()
        let backend = FakeContextBackend(
            state: FakeContextBackendState(
                scripts: [.init(textResponses: [.success("Hello back")])]
            ),
            estimator: FixedTokenEstimator()
        )
        await registry.register(backend)

        let manager = ContextManager(
            configuration: ContextManagerConfiguration(
                runtimeRegistry: registry,
                budget: BudgetPolicy(exactCountingPreferred: true),
                persistence: PersistencePolicy(
                    threads: InMemoryThreadStore(),
                    memories: InMemoryMemoryStore(),
                    blobs: InMemoryBlobStore()
                ),
                diagnostics: DiagnosticsPolicy(isEnabled: false, logToOSLog: false)
            )
        )

        let session = try await manager.session(
            id: "thread-text",
            configuration: ThreadConfiguration(
                runtime: ThreadRuntimeConfiguration(
                    inference: ModelEndpoint(backendID: backend.backendID, modelID: "demo")
                ),
                instructions: "Reply plainly."
            )
        )

        let reply = try await session.reply(to: "Hello")
        let history = try await session.inspection.history()

        XCTAssertEqual(reply.text, "Hello back")
        XCTAssertEqual(reply.metadata.budget.accuracy, .exact)
        XCTAssertEqual(history.map(\.text), ["Hello", "Hello back"])
    }

    func testOverflowTriggersBridge() async throws {
        let state = FakeContextBackendState(
            scripts: [
                .init(textResponses: [.failure(.contextOverflow("overflow"))]),
                .init(textResponses: [.success("Recovered")])
            ]
        )
        let registry = RuntimeRegistry()
        let backend = FakeContextBackend(state: state)
        await registry.register(backend)

        let manager = ContextManager(
            configuration: ContextManagerConfiguration(
                runtimeRegistry: registry,
                persistence: PersistencePolicy(
                    threads: InMemoryThreadStore(),
                    memories: InMemoryMemoryStore(),
                    blobs: InMemoryBlobStore()
                ),
                diagnostics: DiagnosticsPolicy(isEnabled: false, logToOSLog: false)
            )
        )

        let session = try await manager.session(
            id: "thread-bridge",
            configuration: ThreadConfiguration(
                runtime: ThreadRuntimeConfiguration(
                    inference: ModelEndpoint(backendID: backend.backendID, modelID: "demo")
                ),
                instructions: "Keep context."
            )
        )

        let reply = try await session.reply(to: "Continue")
        let diagnostics = await session.inspection.diagnostics()
        let seeds = await state.recordedSeeds()

        XCTAssertEqual(reply.text, "Recovered")
        XCTAssertEqual(diagnostics?.windowIndex, 1)
        XCTAssertEqual(diagnostics?.lastBridge?.reason, "exceededContextWindowSize")
        XCTAssertEqual(seeds.count, 2)
    }

    func testStructuredBackendPersistence() async throws {
        let registry = RuntimeRegistry()
        let inferenceBackend = FakeContextBackend(
            backendID: "fake-inference",
            state: FakeContextBackendState(
                scripts: [.init(textResponses: [.success("unused")])]
            )
        )
        await registry.register(inferenceBackend)

        let structuredState = FakeStructuredBackendState(payload: #"{"title":"done"}"#)
        let structuredBackend = FakeStructuredBackend(
            backendID: "fake-structured",
            state: structuredState
        )

        let manager = ContextManager(
            configuration: ContextManagerConfiguration(
                runtimeRegistry: registry,
                structuredBackends: ["fake-structured": structuredBackend],
                persistence: PersistencePolicy(
                    threads: InMemoryThreadStore(),
                    memories: InMemoryMemoryStore(),
                    blobs: InMemoryBlobStore()
                ),
                diagnostics: DiagnosticsPolicy(isEnabled: false, logToOSLog: false)
            )
        )

        let session = try await manager.session(
            id: "thread-structured",
            configuration: ThreadConfiguration(
                runtime: ThreadRuntimeConfiguration(
                    inference: ModelEndpoint(backendID: "fake-inference", modelID: "text"),
                    structuredOutput: ModelEndpoint(backendID: "fake-structured", modelID: "json")
                ),
                instructions: "Produce structured output."
            )
        )

        let spec = StructuredOutput.codable(
            StructuredPayload.self,
            schema: .object(
                ObjectSchema(
                    name: "StructuredPayload",
                    properties: [.init(name: "title", schema: .string())]
                )
            ),
            renderTranscript: { value in
                "Rendered: \(value.title)"
            }
        )

        let reply = try await session.reply(to: "Generate", spec: spec)
        let history = try await session.inspection.history()
        let endpoints = await structuredState.recordedEndpointIDs()

        XCTAssertEqual(reply.value, StructuredPayload(title: "done"))
        XCTAssertEqual(history.last?.text, "Rendered: done")
        XCTAssertEqual(endpoints, ["fake-structured"])
    }
}

private struct StructuredPayload: Codable, Sendable, Equatable {
    let title: String
}

private struct FakeContextBackend: InferenceBackend {
    let backendID: String
    let state: FakeContextBackendState
    let estimator: (any TokenEstimating)?

    init(
        backendID: String = "fake-context",
        state: FakeContextBackendState,
        estimator: (any TokenEstimating)? = nil
    ) {
        self.backendID = backendID
        self.state = state
        self.estimator = estimator
    }

    func availability(for endpoint: ModelEndpoint) async -> RuntimeAvailability {
        _ = endpoint
        return RuntimeAvailability(
            status: .available,
            capabilities: RuntimeCapabilities(
                supportsTextGeneration: true,
                supportsTextStreaming: true,
                supportsStructuredOutput: false,
                supportsExactTokenEstimation: estimator != nil,
                supportsLocaleHints: false
            )
        )
    }

    func makeSession(
        endpoint: ModelEndpoint,
        instructions: String?,
        locale: Locale?
    ) async throws -> any InferenceSession {
        _ = endpoint
        _ = locale
        return await state.makeSession(instructions: instructions)
    }

    func contextWindowTokens(for endpoint: ModelEndpoint) async -> Int? {
        endpoint.contextWindowOverride ?? 4096
    }

    func exactTokenEstimator(for endpoint: ModelEndpoint) async -> (any TokenEstimating)? {
        _ = endpoint
        return estimator
    }
}

private actor FakeContextBackendState {
    private var scripts: [FakeSessionScript]
    private var seeds: [String?] = []

    init(scripts: [FakeSessionScript]) {
        self.scripts = scripts
    }

    func makeSession(instructions: String?) -> FakeContextSession {
        seeds.append(instructions)
        let script = scripts.isEmpty ? FakeSessionScript() : scripts.removeFirst()
        return FakeContextSession(script: script)
    }

    func recordedSeeds() -> [String?] {
        seeds
    }
}

private struct FakeSessionScript: Sendable {
    var textResponses: [Result<String, RuntimeError>] = []
}

private actor FakeContextSession: InferenceSession {
    private var script: FakeSessionScript

    init(script: FakeSessionScript) {
        self.script = script
    }

    func generateText(
        prompt: String,
        options: TextGenerationOptions
    ) async throws -> TextGenerationResult {
        _ = prompt
        _ = options
        guard script.textResponses.isEmpty == false else {
            return TextGenerationResult(text: "default")
        }
        let next = script.textResponses.removeFirst()
        switch next {
        case .success(let text):
            return TextGenerationResult(text: text)
        case .failure(let error):
            throw error
        }
    }

    func streamText(
        prompt: String,
        options: TextGenerationOptions
    ) async -> AsyncThrowingStream<TextStreamEvent, Error> {
        _ = prompt
        _ = options
        return AsyncThrowingStream { continuation in
            continuation.yield(.completed(.init(text: "unused")))
            continuation.finish()
        }
    }
}

private actor FakeStructuredBackendState {
    private let payload: String
    private var endpointIDs: [String] = []

    init(payload: String) {
        self.payload = payload
    }

    func record(endpoint: ModelEndpoint) {
        endpointIDs.append(endpoint.backendID)
    }

    func renderedPayload() -> String {
        payload
    }

    func recordedEndpointIDs() -> [String] {
        endpointIDs
    }
}

private struct FakeStructuredBackend: StructuredOutputBackend {
    let backendID: String
    let state: FakeStructuredBackendState

    func generateStructured<Value: Sendable>(
        endpoint: ModelEndpoint,
        instructions: String?,
        locale: Locale?,
        prompt: String,
        spec: StructuredOutputSpec<Value>,
        options: StructuredGenerationOptions
    ) async throws -> StructuredGenerationResult<Value> {
        _ = instructions
        _ = locale
        _ = prompt
        _ = options
        await state.record(endpoint: endpoint)
        let payload = await state.renderedPayload()
        let value = try spec.decode(Data(payload.utf8))
        return StructuredGenerationResult(
            value: value,
            transcriptText: spec.transcriptRenderer(value)
        )
    }
}

private struct FixedTokenEstimator: TokenEstimating {
    func estimate(
        prompt: RenderedPrompt,
        reservedOutputTokens: Int
    ) async -> TokenEstimate? {
        _ = prompt
        _ = reservedOutputTokens
        return TokenEstimate(
            inputTokens: 42,
            breakdown: [.instructions: 12, .currentPrompt: 30]
        )
    }
}
