@_exported import LanguageModelRuntime
@_exported import LanguageModelStructuredCore
import Foundation
import OSLog

public struct ThreadRuntimeConfiguration: Codable, Sendable, Equatable {
    public var inference: ModelEndpoint
    public var structuredOutput: ModelEndpoint?

    public init(
        inference: ModelEndpoint,
        structuredOutput: ModelEndpoint? = nil
    ) {
        self.inference = inference
        self.structuredOutput = structuredOutput
    }
}

public struct ContextManagerConfiguration: Sendable {
    public var runtimeRegistry: RuntimeRegistry
    public var structuredBackends: [String: any StructuredOutputBackend]
    public var budget: BudgetPolicy
    public var compaction: CompactionPolicy
    public var memory: MemoryPolicy
    public var persistence: PersistencePolicy
    public var diagnostics: DiagnosticsPolicy

    public init(
        runtimeRegistry: RuntimeRegistry = RuntimeRegistry(),
        structuredBackends: [String: any StructuredOutputBackend] = [:],
        budget: BudgetPolicy = .default,
        compaction: CompactionPolicy = .default,
        memory: MemoryPolicy = .default,
        persistence: PersistencePolicy = .default,
        diagnostics: DiagnosticsPolicy = .default
    ) {
        self.runtimeRegistry = runtimeRegistry
        self.structuredBackends = structuredBackends
        self.budget = budget
        self.compaction = compaction
        self.memory = memory
        self.persistence = persistence
        self.diagnostics = diagnostics
    }

    public static var `default`: ContextManagerConfiguration {
        ContextManagerConfiguration()
    }
}

public struct ThreadConfiguration: Sendable {
    public var runtime: ThreadRuntimeConfiguration
    public var instructions: String?
    public var locale: Locale?

    public init(
        runtime: ThreadRuntimeConfiguration,
        instructions: String? = nil,
        locale: Locale? = nil
    ) {
        self.runtime = runtime
        self.instructions = instructions
        self.locale = locale
    }
}

public enum ContextManagerError: Error, Sendable, Equatable {
    case threadNotFound(String)
    case persistenceFailed(String)
    case budgetExhausted(SessionDiagnostics)
}

public struct BudgetPolicy: Sendable, Equatable {
    public var reservedOutputTokens: Int
    public var preemptiveCompactionFraction: Double
    public var emergencyFraction: Double
    public var maxBridgeRetries: Int
    public var exactCountingPreferred: Bool
    public var heuristicSafetyMultiplier: Double
    public var defaultContextWindowTokens: Int

    public init(
        reservedOutputTokens: Int = 768,
        preemptiveCompactionFraction: Double = 0.78,
        emergencyFraction: Double = 0.90,
        maxBridgeRetries: Int = 2,
        exactCountingPreferred: Bool = true,
        heuristicSafetyMultiplier: Double = 1.10,
        defaultContextWindowTokens: Int = 4096
    ) {
        self.reservedOutputTokens = reservedOutputTokens
        self.preemptiveCompactionFraction = preemptiveCompactionFraction
        self.emergencyFraction = emergencyFraction
        self.maxBridgeRetries = maxBridgeRetries
        self.exactCountingPreferred = exactCountingPreferred
        self.heuristicSafetyMultiplier = heuristicSafetyMultiplier
        self.defaultContextWindowTokens = defaultContextWindowTokens
    }

    public static let `default` = BudgetPolicy()
}

public enum CompactionMode: String, Codable, Sendable, Equatable {
    case slidingWindow
    case structuredSummary
    case hybrid
}

public struct CompactionPolicy: Sendable, Equatable {
    public var mode: CompactionMode
    public var maxRecentTurns: Int
    public var chunkTargetTokens: Int
    public var chunkSummaryTargetTokens: Int
    public var maxMergeDepth: Int

    public init(
        mode: CompactionMode = .hybrid,
        maxRecentTurns: Int = 8,
        chunkTargetTokens: Int = 1200,
        chunkSummaryTargetTokens: Int = 160,
        maxMergeDepth: Int = 3
    ) {
        self.mode = mode
        self.maxRecentTurns = maxRecentTurns
        self.chunkTargetTokens = chunkTargetTokens
        self.chunkSummaryTargetTokens = chunkSummaryTargetTokens
        self.maxMergeDepth = maxMergeDepth
    }

    public static let `default` = CompactionPolicy()
}

public struct MemoryPolicy: Sendable, Equatable {
    public var automaticallyExtractMemories: Bool
    public var retrievalLimit: Int
    public var inlineBlobByteLimit: Int

    public init(
        automaticallyExtractMemories: Bool = true,
        retrievalLimit: Int = 5,
        inlineBlobByteLimit: Int = 2048
    ) {
        self.automaticallyExtractMemories = automaticallyExtractMemories
        self.retrievalLimit = retrievalLimit
        self.inlineBlobByteLimit = inlineBlobByteLimit
    }

    public static let `default` = MemoryPolicy()
}

public struct PersistencePolicy: Sendable {
    public var threads: any ThreadStore
    public var memories: any MemoryStore
    public var blobs: any BlobStore
    public var retriever: (any Retriever)?

    public init(
        threads: any ThreadStore,
        memories: any MemoryStore,
        blobs: any BlobStore,
        retriever: (any Retriever)? = nil
    ) {
        self.threads = threads
        self.memories = memories
        self.blobs = blobs
        self.retriever = retriever
    }

    public static var `default`: PersistencePolicy {
        PersistencePolicy(
            threads: InMemoryThreadStore(),
            memories: InMemoryMemoryStore(),
            blobs: InMemoryBlobStore(),
            retriever: nil
        )
    }
}

public struct DiagnosticsPolicy: Sendable, Equatable {
    public var isEnabled: Bool
    public var logToOSLog: Bool

    public init(
        isEnabled: Bool = true,
        logToOSLog: Bool = true
    ) {
        self.isEnabled = isEnabled
        self.logToOSLog = logToOSLog
    }

    public static let `default` = DiagnosticsPolicy()
}

public protocol ThreadStore: Sendable {
    func load(threadID: String) async throws -> PersistedThreadState?
    func save(_ state: PersistedThreadState, threadID: String) async throws
    func delete(threadID: String) async throws
}

public protocol MemoryStore: Sendable {
    func load(threadID: String) async throws -> [DurableMemoryRecord]
    func save(_ records: [DurableMemoryRecord], threadID: String) async throws
    func append(_ record: DurableMemoryRecord, threadID: String) async throws
    func deleteAll(threadID: String) async throws
}

public protocol BlobStore: Sendable {
    func put(_ data: Data) async throws -> UUID
    func get(_ id: UUID) async throws -> Data?
    func delete(_ id: UUID) async throws
}

public protocol Retriever: Sendable {
    func retrieve(
        query: String,
        threadID: String,
        limit: Int
    ) async throws -> [DurableMemoryRecord]
}

public struct TextReply: Sendable, Equatable {
    public let text: String
    public let metadata: TurnMetadata
}

public struct StructuredReply<Value: Sendable>: Sendable {
    public let value: Value
    public let transcriptText: String
    public let metadata: TurnMetadata
}

public struct TurnMetadata: Sendable, Equatable {
    public let budget: BudgetReport
    public let compaction: CompactionReport?
    public let bridge: BridgeReport?
}

public struct SessionDiagnostics: Codable, Sendable, Equatable {
    public let sessionID: String
    public let windowIndex: Int
    public let lastBudget: BudgetReport?
    public let lastCompaction: CompactionReport?
    public let lastBridge: BridgeReport?
    public let turnCount: Int
    public let durableMemoryCount: Int
    public let blobCount: Int
}

public enum BudgetComponent: String, Codable, Sendable, Hashable, CaseIterable {
    case instructions
    case durableMemory
    case retrievedMemory
    case recentTail
    case currentPrompt
    case schema
    case outputReserve
    case other
}

public struct BudgetReport: Codable, Sendable, Equatable {
    public enum Accuracy: String, Codable, Sendable, Equatable {
        case exact
        case approximate
    }

    public let accuracy: Accuracy
    public let contextWindowTokens: Int
    public let estimatedInputTokens: Int
    public let reservedOutputTokens: Int
    public let projectedTotalTokens: Int
    public let softLimitTokens: Int
    public let emergencyLimitTokens: Int
    public let breakdown: [BudgetComponent: Int]

    public init(
        accuracy: Accuracy,
        contextWindowTokens: Int,
        estimatedInputTokens: Int,
        reservedOutputTokens: Int,
        projectedTotalTokens: Int,
        softLimitTokens: Int,
        emergencyLimitTokens: Int,
        breakdown: [BudgetComponent: Int]
    ) {
        self.accuracy = accuracy
        self.contextWindowTokens = contextWindowTokens
        self.estimatedInputTokens = estimatedInputTokens
        self.reservedOutputTokens = reservedOutputTokens
        self.projectedTotalTokens = projectedTotalTokens
        self.softLimitTokens = softLimitTokens
        self.emergencyLimitTokens = emergencyLimitTokens
        self.breakdown = breakdown
    }
}

public enum ReducerKind: String, Codable, Sendable, Equatable {
    case toolPayloadDigester
    case dropLowPriorityRetrievedMemory
    case slidingTail
    case structuredSummary
    case emergencyReset
}

public struct CompactionReport: Codable, Sendable, Equatable {
    public let requestedMode: CompactionMode
    public let effectiveMode: CompactionMode
    public let downgradeReason: String?
    public let tokensBefore: Int
    public let tokensAfter: Int
    public let reducersApplied: [ReducerKind]
    public let summaryCreated: Bool
    public let spilledBlobCount: Int

    public var mode: CompactionMode { requestedMode }

    public init(
        requestedMode: CompactionMode,
        effectiveMode: CompactionMode? = nil,
        downgradeReason: String? = nil,
        tokensBefore: Int,
        tokensAfter: Int,
        reducersApplied: [ReducerKind],
        summaryCreated: Bool,
        spilledBlobCount: Int
    ) {
        self.requestedMode = requestedMode
        self.effectiveMode = effectiveMode ?? requestedMode
        self.downgradeReason = downgradeReason
        self.tokensBefore = tokensBefore
        self.tokensAfter = tokensAfter
        self.reducersApplied = reducersApplied
        self.summaryCreated = summaryCreated
        self.spilledBlobCount = spilledBlobCount
    }
}

public struct BridgeReport: Codable, Sendable, Equatable {
    public let fromWindowIndex: Int
    public let toWindowIndex: Int
    public let reason: String
    public let carriedTurnCount: Int
    public let summaryUsed: Bool

    public init(
        fromWindowIndex: Int,
        toWindowIndex: Int,
        reason: String,
        carriedTurnCount: Int,
        summaryUsed: Bool
    ) {
        self.fromWindowIndex = fromWindowIndex
        self.toWindowIndex = toWindowIndex
        self.reason = reason
        self.carriedTurnCount = carriedTurnCount
        self.summaryUsed = summaryUsed
    }
}

public struct NormalizedTurn: Codable, Sendable, Identifiable, Equatable {
    public enum Role: String, Codable, Sendable, Equatable {
        case user
        case assistant
        case system
        case summary
    }

    public let id: UUID
    public let role: Role
    public let text: String
    public let createdAt: Date
    public let priority: Int
    public let tags: [String]
    public let blobIDs: [UUID]
    public let windowIndex: Int
    public let compacted: Bool

    public init(
        id: UUID = UUID(),
        role: Role,
        text: String,
        createdAt: Date = Date(),
        priority: Int,
        tags: [String] = [],
        blobIDs: [UUID] = [],
        windowIndex: Int,
        compacted: Bool = false
    ) {
        self.id = id
        self.role = role
        self.text = text
        self.createdAt = createdAt
        self.priority = priority
        self.tags = tags
        self.blobIDs = blobIDs
        self.windowIndex = windowIndex
        self.compacted = compacted
    }
}

public struct DurableMemoryRecord: Codable, Sendable, Identifiable, Equatable {
    public enum Kind: String, Codable, Sendable, Equatable {
        case fact
        case constraint
        case decision
        case openTask
        case summary
        case blobRef
    }

    public let id: UUID
    public let kind: Kind
    public let text: String
    public let createdAt: Date
    public let priority: Int
    public let tags: [String]
    public let blobIDs: [UUID]
    public let pinned: Bool

    public init(
        id: UUID = UUID(),
        kind: Kind,
        text: String,
        createdAt: Date = Date(),
        priority: Int,
        tags: [String] = [],
        blobIDs: [UUID] = [],
        pinned: Bool = false
    ) {
        self.id = id
        self.kind = kind
        self.text = text
        self.createdAt = createdAt
        self.priority = priority
        self.tags = tags
        self.blobIDs = blobIDs
        self.pinned = pinned
    }
}

public struct PersistedThreadState: Codable, Sendable, Equatable {
    public let threadID: String
    public var instructions: String?
    public var localeIdentifier: String?
    public var runtime: ThreadRuntimeConfiguration
    public var activeWindowIndex: Int
    public var turns: [NormalizedTurn]
    public var lastBudget: BudgetReport?
    public var lastCompaction: CompactionReport?
    public var lastBridge: BridgeReport?
    public var createdAt: Date
    public var updatedAt: Date

    public init(
        threadID: String,
        instructions: String?,
        localeIdentifier: String?,
        runtime: ThreadRuntimeConfiguration,
        activeWindowIndex: Int = 0,
        turns: [NormalizedTurn] = [],
        lastBudget: BudgetReport? = nil,
        lastCompaction: CompactionReport? = nil,
        lastBridge: BridgeReport? = nil,
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.threadID = threadID
        self.instructions = instructions
        self.localeIdentifier = localeIdentifier
        self.runtime = runtime
        self.activeWindowIndex = activeWindowIndex
        self.turns = turns
        self.lastBudget = lastBudget
        self.lastCompaction = lastCompaction
        self.lastBridge = lastBridge
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }
}

public actor InMemoryThreadStore: ThreadStore {
    private var storage: [String: PersistedThreadState] = [:]

    public init() {}

    public func load(threadID: String) async throws -> PersistedThreadState? {
        storage[threadID]
    }

    public func save(_ state: PersistedThreadState, threadID: String) async throws {
        storage[threadID] = state
    }

    public func delete(threadID: String) async throws {
        storage.removeValue(forKey: threadID)
    }
}

public actor FileThreadStore: ThreadStore {
    private let directoryURL: URL
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    public init(directoryURL: URL) {
        self.directoryURL = directoryURL
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        decoder.dateDecodingStrategy = .iso8601
    }

    public func load(threadID: String) async throws -> PersistedThreadState? {
        let url = fileURL(for: threadID)
        guard FileManager.default.fileExists(atPath: url.path()) else {
            return nil
        }
        let data = try Data(contentsOf: url)
        return try decoder.decode(PersistedThreadState.self, from: data)
    }

    public func save(_ state: PersistedThreadState, threadID: String) async throws {
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true, attributes: nil)
        let data = try encoder.encode(state)
        try data.write(to: fileURL(for: threadID), options: .atomic)
    }

    public func delete(threadID: String) async throws {
        let url = fileURL(for: threadID)
        guard FileManager.default.fileExists(atPath: url.path()) else {
            return
        }
        try FileManager.default.removeItem(at: url)
    }

    private func fileURL(for threadID: String) -> URL {
        directoryURL
            .appendingPathComponent(StorageKey.filename(threadID))
            .appendingPathExtension("json")
    }
}

public actor InMemoryMemoryStore: MemoryStore {
    private var storage: [String: [DurableMemoryRecord]] = [:]

    public init() {}

    public func load(threadID: String) async throws -> [DurableMemoryRecord] {
        storage[threadID] ?? []
    }

    public func save(_ records: [DurableMemoryRecord], threadID: String) async throws {
        storage[threadID] = records
    }

    public func append(_ record: DurableMemoryRecord, threadID: String) async throws {
        storage[threadID, default: []].append(record)
    }

    public func deleteAll(threadID: String) async throws {
        storage.removeValue(forKey: threadID)
    }
}

public actor FileMemoryStore: MemoryStore {
    private let directoryURL: URL
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    public init(directoryURL: URL) {
        self.directoryURL = directoryURL
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        decoder.dateDecodingStrategy = .iso8601
    }

    public func load(threadID: String) async throws -> [DurableMemoryRecord] {
        let url = fileURL(for: threadID)
        guard FileManager.default.fileExists(atPath: url.path()) else {
            return []
        }
        let data = try Data(contentsOf: url)
        return try decoder.decode([DurableMemoryRecord].self, from: data)
    }

    public func save(_ records: [DurableMemoryRecord], threadID: String) async throws {
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true, attributes: nil)
        let data = try encoder.encode(records)
        try data.write(to: fileURL(for: threadID), options: .atomic)
    }

    public func append(_ record: DurableMemoryRecord, threadID: String) async throws {
        var records = try await load(threadID: threadID)
        records.append(record)
        try await save(records, threadID: threadID)
    }

    public func deleteAll(threadID: String) async throws {
        let url = fileURL(for: threadID)
        guard FileManager.default.fileExists(atPath: url.path()) else {
            return
        }
        try FileManager.default.removeItem(at: url)
    }

    private func fileURL(for threadID: String) -> URL {
        directoryURL
            .appendingPathComponent(StorageKey.filename(threadID))
            .appendingPathExtension("json")
    }
}

public actor InMemoryBlobStore: BlobStore {
    private var storage: [UUID: Data] = [:]

    public init() {}

    public func put(_ data: Data) async throws -> UUID {
        let id = UUID()
        storage[id] = data
        return id
    }

    public func get(_ id: UUID) async throws -> Data? {
        storage[id]
    }

    public func delete(_ id: UUID) async throws {
        storage.removeValue(forKey: id)
    }
}

public actor FileBlobStore: BlobStore {
    private let directoryURL: URL

    public init(directoryURL: URL) {
        self.directoryURL = directoryURL
    }

    public func put(_ data: Data) async throws -> UUID {
        try FileManager.default.createDirectory(at: directoryURL, withIntermediateDirectories: true, attributes: nil)
        let id = UUID()
        try data.write(to: fileURL(for: id), options: .atomic)
        return id
    }

    public func get(_ id: UUID) async throws -> Data? {
        let url = fileURL(for: id)
        guard FileManager.default.fileExists(atPath: url.path()) else {
            return nil
        }
        return try Data(contentsOf: url)
    }

    public func delete(_ id: UUID) async throws {
        let url = fileURL(for: id)
        guard FileManager.default.fileExists(atPath: url.path()) else {
            return
        }
        try FileManager.default.removeItem(at: url)
    }

    private func fileURL(for id: UUID) -> URL {
        directoryURL
            .appendingPathComponent(id.uuidString)
            .appendingPathExtension("blob")
    }
}

public actor KeywordRetriever: Retriever {
    private let memoryStore: any MemoryStore

    public init(memoryStore: any MemoryStore) {
        self.memoryStore = memoryStore
    }

    public func retrieve(
        query: String,
        threadID: String,
        limit: Int
    ) async throws -> [DurableMemoryRecord] {
        let queryTerms = Set(Self.tokenize(query))
        guard queryTerms.isEmpty == false else {
            return []
        }

        let records = try await memoryStore.load(threadID: threadID)
        return records
            .map { record in
                let score = queryTerms.intersection(Set(Self.tokenize(record.text))).count
                return (record, score)
            }
            .filter { $0.1 > 0 }
            .sorted { lhs, rhs in
                if lhs.1 == rhs.1 {
                    return lhs.0.priority > rhs.0.priority
                }
                return lhs.1 > rhs.1
            }
            .prefix(limit)
            .map(\.0)
    }

    private static func tokenize(_ text: String) -> [String] {
        text.lowercased()
            .split { !$0.isLetter && !$0.isNumber }
            .map(String.init)
    }
}

package enum StorageKey {
    package static func filename(_ input: String) -> String {
        Data(input.utf8)
            .base64EncodedString()
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "=", with: "")
    }
}

package struct DiagnosticsLogger: Sendable {
    private let policy: DiagnosticsPolicy
    private let subsystem = "LanguageModelContextManagement"

    package init(policy: DiagnosticsPolicy) {
        self.policy = policy
    }

    package func budget(_ message: String) {
        log(category: "budget", message: message)
    }

    package func compaction(_ message: String) {
        log(category: "compaction", message: message)
    }

    package func bridge(_ message: String) {
        log(category: "bridge", message: message)
    }

    package func error(_ message: String) {
        log(category: "errors", message: message)
    }

    private func log(category: String, message: String) {
        guard policy.isEnabled, policy.logToOSLog else {
            return
        }
        let logger = Logger(subsystem: subsystem, category: category)
        logger.log("\(message, privacy: .public)")
    }
}
