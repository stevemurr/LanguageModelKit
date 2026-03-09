import Foundation

public struct ContextSession: Sendable {
    fileprivate let runtime: ContextManager
    public let id: String

    init(runtime: ContextManager, id: String) {
        self.runtime = runtime
        self.id = id
    }

    public func respond(_ prompt: String) async throws -> String {
        try await runtime.respondText(to: prompt, threadID: id).text
    }

    public func reply(to prompt: String) async throws -> TextReply {
        try await runtime.respondText(to: prompt, threadID: id)
    }

    public func stream(_ prompt: String) -> AsyncThrowingStream<TextStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    try await runtime.streamText(
                        to: prompt,
                        threadID: id,
                        continuation: continuation
                    )
                    continuation.finish()
                } catch is CancellationError {
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    public func generate<Value: Sendable>(
        _ prompt: String,
        spec: StructuredOutputSpec<Value>
    ) async throws -> Value {
        try await runtime.respondStructured(
            to: prompt,
            spec: spec,
            threadID: id
        ).value
    }

    public func reply<Value: Sendable>(
        to prompt: String,
        spec: StructuredOutputSpec<Value>
    ) async throws -> StructuredReply<Value> {
        try await runtime.respondStructured(
            to: prompt,
            spec: spec,
            threadID: id
        )
    }

    public var inspection: SessionInspection {
        SessionInspection(runtime: runtime, id: id)
    }

    public var maintenance: SessionMaintenance {
        SessionMaintenance(runtime: runtime, id: id)
    }
}

public struct SessionInspection: Sendable {
    fileprivate let runtime: ContextManager
    public let id: String

    public func diagnostics() async -> SessionDiagnostics? {
        await runtime.diagnostics(threadID: id)
    }

    public func history() async throws -> [NormalizedTurn] {
        try await runtime.threadState(threadID: id).turns
    }

    public func durableMemory() async throws -> [DurableMemoryRecord] {
        try await runtime.durableMemories(threadID: id)
    }
}

public struct SessionMaintenance: Sendable {
    fileprivate let runtime: ContextManager
    public let id: String

    public func compact() async throws -> CompactionReport {
        try await runtime.compact(threadID: id)
    }

    public func reset() async throws {
        try await runtime.resetThread(threadID: id)
    }

    public func importHistory(
        _ turns: [NormalizedTurn],
        durableMemory: [DurableMemoryRecord] = [],
        replaceExisting: Bool = false
    ) async throws {
        try await runtime.importHistory(
            turns,
            durableMemory: durableMemory,
            replaceExisting: replaceExisting,
            threadID: id
        )
    }

    public func appendTurns(_ turns: [NormalizedTurn]) async throws {
        try await runtime.appendTurns(turns, threadID: id)
    }

    public func appendMemory(
        _ records: [DurableMemoryRecord],
        deduplicate: Bool = true
    ) async throws {
        try await runtime.appendMemories(
            records,
            threadID: id,
            deduplicate: deduplicate
        )
    }
}
