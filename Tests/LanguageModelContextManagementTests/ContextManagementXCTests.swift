import Foundation
import XCTest
@testable import LanguageModelContextManagement
@testable import LanguageModelStructuredCore
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
        let liveStructuredCalls = await inferenceBackend.state.liveStructuredCallCount()

        XCTAssertEqual(reply.value, StructuredPayload(title: "done"))
        XCTAssertEqual(history.last?.text, "Rendered: done")
        XCTAssertEqual(endpoints, ["fake-structured"])
        XCTAssertEqual(liveStructuredCalls, 0)
    }

    func testSameEndpointStructuredReplyReusesLiveSession() async throws {
        let registry = RuntimeRegistry()
        let backend = FakeContextBackend(
            backendID: "fake-live-structured",
            state: FakeContextBackendState(
                scripts: [.init(structuredPayload: #"{"title":"live"}"#)]
            )
        )
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
            id: "thread-live-structured",
            configuration: ThreadConfiguration(
                runtime: ThreadRuntimeConfiguration(
                    inference: ModelEndpoint(backendID: backend.backendID, modelID: "demo")
                ),
                instructions: "Return a structured value."
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
        let liveStructuredCalls = await backend.state.liveStructuredCallCount()

        XCTAssertEqual(reply.value, StructuredPayload(title: "live"))
        XCTAssertEqual(history.last?.text, "Rendered: live")
        XCTAssertEqual(liveStructuredCalls, 1)
    }

    func testStructuredBudgetUsesStructuredEndpointOverride() async throws {
        let registry = RuntimeRegistry()
        let inferenceBackend = FakeContextBackend(
            backendID: "fake-inference",
            state: FakeContextBackendState(scripts: [.init(textResponses: [.success("unused")])])
        )
        await registry.register(inferenceBackend)

        let structuredBackend = FakeStructuredBackend(
            backendID: "fake-structured",
            state: FakeStructuredBackendState(payload: #"{"title":"done"}"#)
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
            id: "thread-structured-budget",
            configuration: ThreadConfiguration(
                runtime: ThreadRuntimeConfiguration(
                    inference: ModelEndpoint(backendID: "fake-inference", modelID: "text", contextWindowOverride: 4096),
                    structuredOutput: ModelEndpoint(backendID: "fake-structured", modelID: "json", contextWindowOverride: 128)
                ),
                instructions: "Return a structured value."
            )
        )
        let spec = StructuredOutput.codable(
            StructuredPayload.self,
            schema: .object(
                ObjectSchema(
                    name: "StructuredPayload",
                    properties: [.init(name: "title", schema: .string())]
                )
            )
        )

        let reply = try await session.reply(to: "Generate", spec: spec)

        XCTAssertEqual(reply.metadata.budget.contextWindowTokens, 128)
    }

    func testBudgetFallsBackToApproximateWhenExactEstimatorIsUnavailable() async throws {
        let registry = RuntimeRegistry()
        let backend = FakeContextBackend(
            state: FakeContextBackendState(
                scripts: [.init(textResponses: [.success("Fallback")])]
            )
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
            id: "thread-budget-fallback",
            configuration: ThreadConfiguration(
                runtime: ThreadRuntimeConfiguration(
                    inference: ModelEndpoint(backendID: backend.backendID, modelID: "demo")
                ),
                instructions: "Reply plainly."
            )
        )

        let reply = try await session.reply(to: "Hello")

        XCTAssertEqual(reply.metadata.budget.accuracy, .approximate)
    }

    func testBudgetFallsBackToDefaultContextWindowWhenBackendHasNoEstimate() async throws {
        let registry = RuntimeRegistry()
        let backend = FakeContextBackend(
            state: FakeContextBackendState(
                scripts: [.init(textResponses: [.success("Fallback")])]
            ),
            contextWindowTokens: nil
        )
        await registry.register(backend)

        let manager = ContextManager(
            configuration: ContextManagerConfiguration(
                runtimeRegistry: registry,
                budget: BudgetPolicy(defaultContextWindowTokens: 2048),
                persistence: PersistencePolicy(
                    threads: InMemoryThreadStore(),
                    memories: InMemoryMemoryStore(),
                    blobs: InMemoryBlobStore()
                ),
                diagnostics: DiagnosticsPolicy(isEnabled: false, logToOSLog: false)
            )
        )

        let session = try await manager.session(
            id: "thread-context-window-fallback",
            configuration: ThreadConfiguration(
                runtime: ThreadRuntimeConfiguration(
                    inference: ModelEndpoint(backendID: backend.backendID, modelID: "demo")
                ),
                instructions: "Reply plainly."
            )
        )

        let reply = try await session.reply(to: "Hello")

        XCTAssertEqual(reply.metadata.budget.contextWindowTokens, 2048)
    }

    func testManualCompactionPersistsDowngrade() async throws {
        let registry = RuntimeRegistry()
        let backend = FakeContextBackend(
            state: FakeContextBackendState(
                scripts: [.init(textResponses: [.success("unused")])]
            )
        )
        await registry.register(backend)

        let manager = ContextManager(
            configuration: ContextManagerConfiguration(
                runtimeRegistry: registry,
                compaction: CompactionPolicy(mode: .structuredSummary, maxRecentTurns: 2, chunkTargetTokens: 40, chunkSummaryTargetTokens: 20, maxMergeDepth: 1),
                memory: MemoryPolicy(automaticallyExtractMemories: false, retrievalLimit: 5, inlineBlobByteLimit: 2048),
                persistence: PersistencePolicy(
                    threads: InMemoryThreadStore(),
                    memories: InMemoryMemoryStore(),
                    blobs: InMemoryBlobStore()
                ),
                diagnostics: DiagnosticsPolicy(isEnabled: false, logToOSLog: false)
            )
        )

        let session = try await manager.session(
            id: "thread-downgrade",
            configuration: ThreadConfiguration(
                runtime: ThreadRuntimeConfiguration(
                    inference: ModelEndpoint(backendID: backend.backendID, modelID: "demo", contextWindowOverride: 64)
                ),
                instructions: "Keep a compact history."
            )
        )
        try await session.maintenance.importHistory(
            [
                NormalizedTurn(role: .user, text: String(repeating: "alpha ", count: 20), priority: 900, windowIndex: 0),
                NormalizedTurn(role: .assistant, text: String(repeating: "beta ", count: 20), priority: 800, windowIndex: 0),
                NormalizedTurn(role: .user, text: String(repeating: "gamma ", count: 20), priority: 900, windowIndex: 0),
                NormalizedTurn(role: .assistant, text: String(repeating: "delta ", count: 20), priority: 800, windowIndex: 0)
            ],
            replaceExisting: true
        )

        let report = try await session.maintenance.compact()
        let diagnostics = await session.inspection.diagnostics()

        XCTAssertEqual(report.requestedMode, .structuredSummary)
        XCTAssertEqual(report.effectiveMode, .slidingWindow)
        XCTAssertNotNil(report.downgradeReason)
        XCTAssertEqual(diagnostics?.lastCompaction?.downgradeReason, report.downgradeReason)
    }

    func testBlobSpillingCreatesBlobRefs() async throws {
        let registry = RuntimeRegistry()
        let backend = FakeContextBackend(
            state: FakeContextBackendState(
                scripts: [.init(textResponses: [.success("unused")])]
            )
        )
        await registry.register(backend)

        let manager = ContextManager(
            configuration: ContextManagerConfiguration(
                runtimeRegistry: registry,
                memory: MemoryPolicy(automaticallyExtractMemories: false, retrievalLimit: 5, inlineBlobByteLimit: 16),
                persistence: PersistencePolicy(
                    threads: InMemoryThreadStore(),
                    memories: InMemoryMemoryStore(),
                    blobs: InMemoryBlobStore()
                ),
                diagnostics: DiagnosticsPolicy(isEnabled: false, logToOSLog: false)
            )
        )

        let session = try await manager.session(
            id: "thread-blobs",
            configuration: ThreadConfiguration(
                runtime: ThreadRuntimeConfiguration(
                    inference: ModelEndpoint(backendID: backend.backendID, modelID: "demo")
                )
            )
        )
        try await session.maintenance.importHistory(
            [
                NormalizedTurn(role: .user, text: String(repeating: "oversized-user-", count: 20), priority: 900, windowIndex: 0),
                NormalizedTurn(role: .assistant, text: String(repeating: "oversized-assistant-", count: 20), priority: 800, windowIndex: 0)
            ],
            replaceExisting: true
        )

        let report = try await session.maintenance.compact()
        let history = try await session.inspection.history()
        let memory = try await session.inspection.durableMemory()

        XCTAssertGreaterThan(report.spilledBlobCount, 0)
        XCTAssertTrue(history.contains { $0.blobIDs.isEmpty == false })
        XCTAssertTrue(memory.contains { $0.kind == .blobRef })
    }

    func testRetrieverInjectionAppearsInBudgetPrompt() async throws {
        let registry = RuntimeRegistry()
        let estimator = RecordingTokenEstimator()
        let backend = FakeContextBackend(
            state: FakeContextBackendState(
                scripts: [.init(textResponses: [.success("retrieved")])]
            ),
            estimator: estimator
        )
        await registry.register(backend)

        let retriever = StaticRetriever(
            records: [DurableMemoryRecord(kind: .fact, text: "alpha project owner", priority: 900)]
        )
        let manager = ContextManager(
            configuration: ContextManagerConfiguration(
                runtimeRegistry: registry,
                budget: BudgetPolicy(exactCountingPreferred: true),
                persistence: PersistencePolicy(
                    threads: InMemoryThreadStore(),
                    memories: InMemoryMemoryStore(),
                    blobs: InMemoryBlobStore(),
                    retriever: retriever
                ),
                diagnostics: DiagnosticsPolicy(isEnabled: false, logToOSLog: false)
            )
        )

        let session = try await manager.session(
            id: "thread-retrieval",
            configuration: ThreadConfiguration(
                runtime: ThreadRuntimeConfiguration(
                    inference: ModelEndpoint(backendID: backend.backendID, modelID: "demo")
                )
            )
        )

        _ = try await session.reply(to: "Who owns alpha?")
        let prompt = await estimator.lastPrompt()

        XCTAssertTrue(prompt?.components.contains(where: {
            $0.section == .retrievedMemory && $0.text.contains("alpha project owner")
        }) == true)
    }

    func testFileStoresRehydrateAcrossManagers() async throws {
        let registry = RuntimeRegistry()
        let backend = FakeContextBackend(
            state: FakeContextBackendState(
                scripts: [.init(textResponses: [.success("persisted")]), .init(textResponses: [.success("reused")])]
            )
        )
        await registry.register(backend)

        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let threadsURL = root.appendingPathComponent("threads", isDirectory: true)
        let memoryURL = root.appendingPathComponent("memory", isDirectory: true)
        let blobsURL = root.appendingPathComponent("blobs", isDirectory: true)

        let configuration = ContextManagerConfiguration(
            runtimeRegistry: registry,
            persistence: PersistencePolicy(
                threads: FileThreadStore(directoryURL: threadsURL),
                memories: FileMemoryStore(directoryURL: memoryURL),
                blobs: FileBlobStore(directoryURL: blobsURL)
            ),
            diagnostics: DiagnosticsPolicy(isEnabled: false, logToOSLog: false)
        )

        let manager1 = ContextManager(configuration: configuration)
        let session1 = try await manager1.session(
            id: "thread-file-stores",
            configuration: ThreadConfiguration(
                runtime: ThreadRuntimeConfiguration(
                    inference: ModelEndpoint(backendID: backend.backendID, modelID: "demo")
                )
            )
        )
        _ = try await session1.reply(to: "Persist this")

        let manager2 = ContextManager(configuration: configuration)
        let session2 = try await manager2.session(
            id: "thread-file-stores",
            configuration: ThreadConfiguration(
                runtime: ThreadRuntimeConfiguration(
                    inference: ModelEndpoint(backendID: backend.backendID, modelID: "demo")
                )
            )
        )
        let history = try await session2.inspection.history()

        XCTAssertEqual(history.map(\.text), ["Persist this", "persisted"])
    }

    func testStreamingPersistsCompletion() async throws {
        let registry = RuntimeRegistry()
        let backend = FakeContextBackend(
            state: FakeContextBackendState(
                scripts: [.init(streamFragments: ["hel", "lo"], streamFinalText: "hello")]
            )
        )
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
            id: "thread-streaming",
            configuration: ThreadConfiguration(
                runtime: ThreadRuntimeConfiguration(
                    inference: ModelEndpoint(backendID: backend.backendID, modelID: "demo")
                )
            )
        )

        let stream = session.stream("Hello")
        var completed: TextGenerationResult?
        for try await event in stream {
            if case .completed(let result) = event {
                completed = result
            }
        }
        let history = try await session.inspection.history()

        XCTAssertEqual(completed, TextGenerationResult(text: "hello"))
        XCTAssertEqual(history.map(\.text), ["Hello", "hello"])
    }
}

private struct StructuredPayload: Codable, Sendable, Equatable {
    let title: String
}

private struct FakeContextBackend: InferenceBackend {
    let backendID: String
    let state: FakeContextBackendState
    let estimator: (any TokenEstimating)?
    let contextWindowTokensValue: Int?

    init(
        backendID: String = "fake-context",
        state: FakeContextBackendState,
        estimator: (any TokenEstimating)? = nil,
        contextWindowTokens: Int? = 4096
    ) {
        self.backendID = backendID
        self.state = state
        self.estimator = estimator
        self.contextWindowTokensValue = contextWindowTokens
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
        endpoint.contextWindowOverride ?? contextWindowTokensValue
    }

    func exactTokenEstimator(for endpoint: ModelEndpoint) async -> (any TokenEstimating)? {
        _ = endpoint
        return estimator
    }
}

private actor FakeContextBackendState {
    private var scripts: [FakeSessionScript]
    private var seeds: [String?] = []
    private var liveStructuredCalls = 0

    init(scripts: [FakeSessionScript]) {
        self.scripts = scripts
    }

    func makeSession(instructions: String?) -> any InferenceSession {
        seeds.append(instructions)
        let script = scripts.isEmpty ? FakeSessionScript() : scripts.removeFirst()
        if script.structuredPayload != nil {
            return FakeLiveStructuredContextSession(script: script, state: self)
        }
        return FakeContextSession(script: script)
    }

    func recordedSeeds() -> [String?] {
        seeds
    }

    func recordLiveStructuredCall() {
        liveStructuredCalls += 1
    }

    func liveStructuredCallCount() -> Int {
        liveStructuredCalls
    }
}

private struct FakeSessionScript: Sendable {
    var textResponses: [Result<String, RuntimeError>] = []
    var streamFragments: [String] = []
    var streamFinalText: String?
    var structuredPayload: String?
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
        let fragments = script.streamFragments
        let finalText = script.streamFinalText ?? fragments.joined()
        return AsyncThrowingStream { continuation in
            for fragment in fragments {
                continuation.yield(.partial(fragment))
            }
            continuation.yield(.completed(.init(text: finalText.isEmpty ? "unused" : finalText)))
            continuation.finish()
        }
    }
}

private actor FakeLiveStructuredContextSession: LiveStructuredGenerationSession {
    private var script: FakeSessionScript
    private let state: FakeContextBackendState

    init(script: FakeSessionScript, state: FakeContextBackendState) {
        self.script = script
        self.state = state
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
        let fragments = script.streamFragments
        let finalText = script.streamFinalText ?? fragments.joined()
        return AsyncThrowingStream { continuation in
            for fragment in fragments {
                continuation.yield(.partial(fragment))
            }
            continuation.yield(.completed(.init(text: finalText.isEmpty ? "unused" : finalText)))
            continuation.finish()
        }
    }

    func generateStructured<Value: Sendable>(
        prompt: String,
        spec: StructuredOutputSpec<Value>,
        options: StructuredGenerationOptions
    ) async throws -> StructuredGenerationResult<Value> {
        _ = prompt
        _ = options
        await state.recordLiveStructuredCall()
        let payload = script.structuredPayload ?? #"{"title":"live"}"#
        let value = try spec.decode(Data(payload.utf8))
        return StructuredGenerationResult(
            value: value,
            transcriptText: spec.transcriptRenderer(value)
        )
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

private actor RecordingTokenEstimator: TokenEstimating {
    private var prompt: RenderedPrompt?

    func estimate(
        prompt: RenderedPrompt,
        reservedOutputTokens: Int
    ) async -> TokenEstimate? {
        _ = reservedOutputTokens
        self.prompt = prompt
        return TokenEstimate(
            inputTokens: 24,
            breakdown: [.currentPrompt: 24]
        )
    }

    func lastPrompt() -> RenderedPrompt? {
        prompt
    }
}

private struct StaticRetriever: Retriever {
    let records: [DurableMemoryRecord]

    func retrieve(
        query: String,
        threadID: String,
        limit: Int
    ) async throws -> [DurableMemoryRecord] {
        _ = query
        _ = threadID
        return Array(records.prefix(limit))
    }
}
