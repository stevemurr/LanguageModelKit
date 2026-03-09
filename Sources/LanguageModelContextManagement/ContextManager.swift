import Foundation

public actor ContextManager {
    private let configuration: ContextManagerConfiguration
    private let logger: DiagnosticsLogger
    private let bridgeSeedBuilder = BridgeSeedBuilder()

    private var threads: [String: LogicalThread] = [:]
    private var liveSessions: [String: WindowSession] = [:]

    public init(configuration: ContextManagerConfiguration) {
        self.configuration = configuration
        self.logger = DiagnosticsLogger(policy: configuration.diagnostics)
    }

    public func session(
        id: String = UUID().uuidString,
        configuration: ThreadConfiguration
    ) async throws -> ContextSession {
        try await openThread(id: id, configuration: configuration)
        return ContextSession(runtime: self, id: id)
    }

    public func availability(for endpoint: ModelEndpoint) async -> RuntimeAvailability {
        do {
            let backend = try await configuration.runtimeRegistry.backend(for: endpoint.backendID)
            return await backend.availability(for: endpoint)
        } catch let error as RuntimeError {
            return RuntimeAvailability(
                status: .unavailable(reason: errorDescription(error)),
                capabilities: RuntimeCapabilities()
            )
        } catch {
            return RuntimeAvailability(
                status: .unavailable(reason: error.localizedDescription),
                capabilities: RuntimeCapabilities()
            )
        }
    }

    func openThread(
        id: String,
        configuration threadConfiguration: ThreadConfiguration
    ) async throws {
        let persisted = try await configuration.persistence.threads.load(threadID: id)
        let state = persisted ?? PersistedThreadState(
            threadID: id,
            instructions: threadConfiguration.instructions,
            localeIdentifier: threadConfiguration.locale?.identifier,
            runtime: threadConfiguration.runtime
        )

        var updated = state
        updated.instructions = threadConfiguration.instructions
        updated.localeIdentifier = threadConfiguration.locale?.identifier
        updated.runtime = threadConfiguration.runtime
        updated.updatedAt = Date()

        threads[id] = LogicalThread(
            state: updated,
            configuration: threadConfiguration
        )
        liveSessions.removeValue(forKey: id)
        try await saveThreadState(updated, threadID: id)
    }

    func respondText(
        to prompt: String,
        threadID: String
    ) async throws -> TextReply {
        let logicalThread = try await requireThread(threadID)
        try await validateAvailability(for: logicalThread)
        var prepared = try await preparePlan(
            for: logicalThread,
            prompt: prompt,
            schemaDescription: nil,
            compactionOptions: .standard(memoryPolicy: configuration.memory)
        )
        var bridge = prepared.bridge
        var attempts = 0

        while true {
            do {
                let session = try await session(
                    for: prepared.thread,
                    durableMemory: prepared.plan.durableMemory,
                    recentTail: prepared.plan.recentTail,
                    forceBridge: prepared.plan.requiresBridge || bridge != nil
                )
                let result = try await session.generateText(
                    prompt: prompt,
                    options: TextGenerationOptions(
                        maximumResponseTokens: configuration.budget.reservedOutputTokens
                    )
                )
                return try await finalizeTextResponse(
                    prompt: prompt,
                    text: result.text,
                    prepared: &prepared,
                    bridge: bridge
                )
            } catch let error as RuntimeError {
                if case .contextOverflow = error {
                    guard attempts < configuration.budget.maxBridgeRetries else {
                        throw ContextManagerError.budgetExhausted(
                            makeDiagnostics(
                                threadID: threadID,
                                state: prepared.thread.state,
                                durableMemory: prepared.plan.durableMemory,
                                budget: prepared.plan.budget,
                                compaction: makeCompactionReport(from: prepared.plan),
                                bridge: bridge
                            )
                        )
                    }
                    attempts += 1
                    prepared.plan.requiresBridge = true
                    prepared.thread.state.activeWindowIndex += 1
                    bridge = BridgeReport(
                        fromWindowIndex: max(0, prepared.thread.state.activeWindowIndex - 1),
                        toWindowIndex: prepared.thread.state.activeWindowIndex,
                        reason: "exceededContextWindowSize",
                        carriedTurnCount: prepared.plan.recentTail.count,
                        summaryUsed: prepared.plan.summaryCreated
                    )
                    liveSessions.removeValue(forKey: threadID)
                    continue
                }
                throw error
            }
        }
    }

    func respondStructured<Value: Sendable>(
        to prompt: String,
        spec: StructuredOutputSpec<Value>,
        threadID: String
    ) async throws -> StructuredReply<Value> {
        let logicalThread = try await requireThread(threadID)
        try await validateAvailability(for: logicalThread)
        let schemaDescription = (try? String(data: JSONEncoder().encode(spec.schema), encoding: .utf8)) ?? String(describing: spec.schema)
        var prepared = try await preparePlan(
            for: logicalThread,
            prompt: prompt,
            schemaDescription: schemaDescription,
            compactionOptions: .standard(memoryPolicy: configuration.memory)
        )
        var bridge = prepared.bridge
        var attempts = 0

        while true {
            do {
                let result = try await generateStructured(
                    prepared: prepared,
                    prompt: prompt,
                    spec: spec
                )
                return try await finalizeStructuredResponse(
                    prompt: prompt,
                    result: result,
                    prepared: &prepared,
                    bridge: bridge,
                    transcriptRenderer: spec.transcriptRenderer
                )
            } catch let error as RuntimeError {
                if case .contextOverflow = error {
                    guard attempts < configuration.budget.maxBridgeRetries else {
                        throw ContextManagerError.budgetExhausted(
                            makeDiagnostics(
                                threadID: threadID,
                                state: prepared.thread.state,
                                durableMemory: prepared.plan.durableMemory,
                                budget: prepared.plan.budget,
                                compaction: makeCompactionReport(from: prepared.plan),
                                bridge: bridge
                            )
                        )
                    }
                    attempts += 1
                    prepared.plan.requiresBridge = true
                    prepared.thread.state.activeWindowIndex += 1
                    bridge = BridgeReport(
                        fromWindowIndex: max(0, prepared.thread.state.activeWindowIndex - 1),
                        toWindowIndex: prepared.thread.state.activeWindowIndex,
                        reason: "exceededContextWindowSize",
                        carriedTurnCount: prepared.plan.recentTail.count,
                        summaryUsed: prepared.plan.summaryCreated
                    )
                    continue
                }
                throw error
            }
        }
    }

    func streamText(
        to prompt: String,
        threadID: String,
        continuation: AsyncThrowingStream<TextStreamEvent, Error>.Continuation
    ) async throws {
        let logicalThread = try await requireThread(threadID)
        try await validateAvailability(for: logicalThread)
        var prepared = try await preparePlan(
            for: logicalThread,
            prompt: prompt,
            schemaDescription: nil,
            compactionOptions: .standard(memoryPolicy: configuration.memory)
        )
        var bridge = prepared.bridge
        var attempts = 0

        while true {
            do {
                let session = try await session(
                    for: prepared.thread,
                    durableMemory: prepared.plan.durableMemory,
                    recentTail: prepared.plan.recentTail,
                    forceBridge: prepared.plan.requiresBridge || bridge != nil
                )
                let stream = await session.streamText(
                    prompt: prompt,
                    options: TextGenerationOptions(
                        maximumResponseTokens: configuration.budget.reservedOutputTokens
                    )
                )
                var finalText: String?
                for try await event in stream {
                    try Task.checkCancellation()
                    switch event {
                    case .partial(let text):
                        continuation.yield(.partial(text))
                    case .completed(let result):
                        finalText = result.text
                    }
                }
                guard let finalText else {
                    throw RuntimeError.generationFailed("Streaming finished without a completed result")
                }
                let reply = try await finalizeTextResponse(
                    prompt: prompt,
                    text: finalText,
                    prepared: &prepared,
                    bridge: bridge
                )
                continuation.yield(.completed(.init(text: reply.text)))
                return
            } catch is CancellationError {
                throw CancellationError()
            } catch let error as RuntimeError {
                if case .contextOverflow = error {
                    guard attempts < configuration.budget.maxBridgeRetries else {
                        throw ContextManagerError.budgetExhausted(
                            makeDiagnostics(
                                threadID: threadID,
                                state: prepared.thread.state,
                                durableMemory: prepared.plan.durableMemory,
                                budget: prepared.plan.budget,
                                compaction: makeCompactionReport(from: prepared.plan),
                                bridge: bridge
                            )
                        )
                    }
                    attempts += 1
                    prepared.plan.requiresBridge = true
                    prepared.thread.state.activeWindowIndex += 1
                    bridge = BridgeReport(
                        fromWindowIndex: max(0, prepared.thread.state.activeWindowIndex - 1),
                        toWindowIndex: prepared.thread.state.activeWindowIndex,
                        reason: "exceededContextWindowSize",
                        carriedTurnCount: prepared.plan.recentTail.count,
                        summaryUsed: prepared.plan.summaryCreated
                    )
                    liveSessions.removeValue(forKey: threadID)
                    continue
                }
                throw error
            }
        }
    }

    func compact(threadID: String) async throws -> CompactionReport {
        let logicalThread = try await requireThread(threadID)
        var prepared = try await preparePlan(
            for: logicalThread,
            prompt: "Manual compaction request",
            schemaDescription: nil,
            compactionOptions: .manual(memoryPolicy: configuration.memory)
        )
        prepared.plan.requiresBridge = true
        let report = makeCompactionReport(from: prepared.plan) ?? CompactionReport(
            mode: configuration.compaction.mode,
            tokensBefore: prepared.plan.budget.projectedTotalTokens,
            tokensAfter: prepared.plan.budget.projectedTotalTokens,
            reducersApplied: [],
            summaryCreated: false,
            spilledBlobCount: 0
        )
        prepared.thread.state.lastCompaction = report
        try await persist(thread: prepared.thread, durableMemory: prepared.plan.durableMemory)
        threads[threadID] = prepared.thread
        liveSessions.removeValue(forKey: threadID)
        return report
    }

    func resetThread(threadID: String) async throws {
        let state = try await requireThreadState(threadID: threadID)
        let memories = try await configuration.persistence.memories.load(threadID: threadID)
        try await deleteBlobs(ids: Set(state.turns.flatMap(\.blobIDs) + memories.flatMap(\.blobIDs)))
        threads.removeValue(forKey: threadID)
        liveSessions.removeValue(forKey: threadID)
        try await configuration.persistence.threads.delete(threadID: threadID)
        try await configuration.persistence.memories.deleteAll(threadID: threadID)
    }

    func importHistory(
        _ turns: [NormalizedTurn],
        durableMemory: [DurableMemoryRecord],
        replaceExisting: Bool,
        threadID: String
    ) async throws {
        let thread = try await requireThread(threadID)
        let existingState = replaceExisting ? nil : try await loadedThreadState(threadID: threadID)
        let existingMemory = replaceExisting ? [] : try await configuration.persistence.memories.load(threadID: threadID)
        let combinedTurns = replaceExisting ? turns : (existingState?.turns ?? []) + turns
        let combinedMemory = replaceExisting ? durableMemory : existingMemory + durableMemory
        let state = PersistedThreadState(
            threadID: threadID,
            instructions: thread.configuration.instructions,
            localeIdentifier: thread.configuration.locale?.identifier,
            runtime: thread.configuration.runtime,
            activeWindowIndex: max(
                existingState?.activeWindowIndex ?? 0,
                combinedTurns.map(\.windowIndex).max() ?? 0
            ),
            turns: deduplicatedImportedTurns(combinedTurns),
            createdAt: existingState?.createdAt ?? Date(),
            updatedAt: Date()
        )
        let logicalThread = LogicalThread(state: state, configuration: thread.configuration)
        threads[threadID] = logicalThread
        liveSessions.removeValue(forKey: threadID)
        try await persist(
            thread: logicalThread,
            durableMemory: deduplicatedMemories(combinedMemory)
        )
    }

    func appendTurns(
        _ turns: [NormalizedTurn],
        threadID: String
    ) async throws {
        var state = try await requireThreadState(threadID: threadID)
        state.turns.append(contentsOf: turns)
        state.turns.sort(by: Self.sortTurnsByCreatedAt)
        state.activeWindowIndex = max(state.activeWindowIndex, turns.map(\.windowIndex).max() ?? state.activeWindowIndex)
        state.updatedAt = Date()
        try await saveThreadState(state, threadID: threadID)
        if var thread = threads[threadID] {
            thread.state = state
            threads[threadID] = thread
        }
        liveSessions.removeValue(forKey: threadID)
    }

    func appendMemories(
        _ records: [DurableMemoryRecord],
        threadID: String,
        deduplicate: Bool
    ) async throws {
        var state = try await requireThreadState(threadID: threadID)
        let existing = try await configuration.persistence.memories.load(threadID: threadID)
        let merged = deduplicate ? deduplicatedMemories(existing + records) : existing + records
        state.updatedAt = Date()
        try await saveThreadState(state, threadID: threadID)
        try await saveMemories(merged, threadID: threadID)
        if var thread = threads[threadID] {
            thread.state = state
            threads[threadID] = thread
        }
        liveSessions.removeValue(forKey: threadID)
    }

    func threadState(threadID: String) async throws -> PersistedThreadState {
        try await requireThreadState(threadID: threadID)
    }

    func durableMemories(threadID: String) async throws -> [DurableMemoryRecord] {
        _ = try await requireThreadState(threadID: threadID)
        return try await configuration.persistence.memories.load(threadID: threadID)
    }

    func diagnostics(threadID: String) async -> SessionDiagnostics? {
        let state = try? await requireThreadState(threadID: threadID)
        guard let state else {
            return nil
        }
        let memories = (try? await configuration.persistence.memories.load(threadID: threadID)) ?? []
        return makeDiagnostics(
            threadID: threadID,
            state: state,
            durableMemory: memories,
            budget: state.lastBudget,
            compaction: state.lastCompaction,
            bridge: state.lastBridge
        )
    }

    private func generateStructured<Value: Sendable>(
        prepared: PreparedRequest,
        prompt: String,
        spec: StructuredOutputSpec<Value>
    ) async throws -> StructuredGenerationResult<Value> {
        let endpoint = prepared.thread.configuration.runtime.structuredOutput ?? prepared.thread.configuration.runtime.inference
        guard let backend = configuration.structuredBackends[endpoint.backendID] else {
            throw RuntimeError.unsupportedCapability("No structured backend configured for \(endpoint.backendID)")
        }
        let seed = bridgeSeedBuilder.makeSeed(
            for: prepared.thread.state,
            durableMemory: prepared.plan.durableMemory,
            recentTail: prepared.plan.recentTail
        )
        return try await backend.generateStructured(
            endpoint: endpoint,
            instructions: seed.instructions,
            locale: prepared.thread.configuration.locale,
            prompt: prompt,
            spec: spec,
            options: StructuredGenerationOptions(
                maximumResponseTokens: configuration.budget.reservedOutputTokens,
                deterministic: true
            )
        )
    }

    private func preparePlan(
        for logicalThread: LogicalThread,
        prompt: String,
        schemaDescription: String?,
        compactionOptions: CompactionOptions
    ) async throws -> PreparedRequest {
        let durableMemory = try await configuration.persistence.memories.load(threadID: logicalThread.state.threadID)
        let retrievedMemory = try await configuration.persistence.retriever?.retrieve(
            query: prompt,
            threadID: logicalThread.state.threadID,
            limit: configuration.memory.retrievalLimit
        ) ?? []
        let recentTail = recentTail(from: logicalThread.state.turns)
        let budget = try await calculateBudget(
            thread: logicalThread,
            durableMemory: durableMemory,
            retrievedMemory: retrievedMemory,
            recentTail: recentTail,
            prompt: prompt,
            schemaDescription: schemaDescription
        )
        var plan = ContextPlan(
            state: logicalThread.state,
            durableMemory: durableMemory,
            retrievedMemory: retrievedMemory,
            recentTail: recentTail,
            currentPrompt: prompt,
            schemaDescription: schemaDescription,
            budget: budget,
            originalProjectedTotalTokens: nil
        )
        plan = try await compact(
            plan: plan,
            threadConfiguration: logicalThread.configuration,
            options: compactionOptions
        )

        var thread = logicalThread
        thread.state = plan.state
        var bridge: BridgeReport?
        if liveSessions[thread.state.threadID] == nil && (thread.state.turns.isEmpty == false || plan.durableMemory.isEmpty == false) {
            bridge = BridgeReport(
                fromWindowIndex: max(0, thread.state.activeWindowIndex - 1),
                toWindowIndex: thread.state.activeWindowIndex,
                reason: "rehydrate",
                carriedTurnCount: plan.recentTail.count,
                summaryUsed: plan.summaryCreated
            )
            plan.requiresBridge = true
        }

        return PreparedRequest(thread: thread, plan: plan, bridge: bridge)
    }

    private func calculateBudget(
        thread: LogicalThread,
        durableMemory: [DurableMemoryRecord],
        retrievedMemory: [DurableMemoryRecord],
        recentTail: [NormalizedTurn],
        prompt: String,
        schemaDescription: String?
    ) async throws -> BudgetReport {
        let contextWindowTokens = try await resolvedContextWindowTokens(for: thread.configuration.runtime.inference)
        let renderedPrompt = renderedPrompt(
            instructions: thread.state.instructions,
            durableMemory: durableMemory,
            retrievedMemory: retrievedMemory,
            recentTail: recentTail,
            currentPrompt: prompt,
            schemaDescription: schemaDescription
        )

        if configuration.budget.exactCountingPreferred,
           let estimator = try await exactEstimator(for: thread.configuration.runtime.inference),
           let estimate = await estimator.estimate(
            prompt: renderedPrompt,
            reservedOutputTokens: configuration.budget.reservedOutputTokens
           ) {
            let breakdown = mapExactBreakdown(estimate.breakdown)
            let projected = estimate.inputTokens + configuration.budget.reservedOutputTokens
            return BudgetReport(
                accuracy: .exact,
                contextWindowTokens: contextWindowTokens,
                estimatedInputTokens: estimate.inputTokens,
                reservedOutputTokens: configuration.budget.reservedOutputTokens,
                projectedTotalTokens: projected,
                softLimitTokens: Int(Double(contextWindowTokens) * configuration.budget.preemptiveCompactionFraction),
                emergencyLimitTokens: Int(Double(contextWindowTokens) * configuration.budget.emergencyFraction),
                breakdown: breakdown.merging([.outputReserve: configuration.budget.reservedOutputTokens], uniquingKeysWith: { $1 })
            )
        }

        let heuristic = HeuristicBudgetEstimator()
        let report = heuristic.estimate(
            instructions: thread.state.instructions,
            durableMemory: durableMemory,
            retrievedMemory: retrievedMemory,
            recentTail: recentTail,
            currentPrompt: prompt,
            schemaDescription: schemaDescription,
            budgetPolicy: configuration.budget,
            contextWindowTokens: contextWindowTokens
        )
        logger.budget("projected=\(report.projectedTotalTokens) soft=\(report.softLimitTokens)")
        return report
    }

    private func compact(
        plan initialPlan: ContextPlan,
        threadConfiguration: ThreadConfiguration,
        options: CompactionOptions
    ) async throws -> ContextPlan {
        var plan = initialPlan
        plan.originalProjectedTotalTokens = initialPlan.budget.projectedTotalTokens

        guard options.force || plan.budget.projectedTotalTokens > plan.budget.softLimitTokens else {
            return plan
        }

        for reducer in reducers(for: configuration.compaction.mode) {
            let changed = try await apply(
                reducer: reducer,
                to: &plan,
                threadConfiguration: threadConfiguration,
                options: options
            )
            guard changed else {
                continue
            }

            plan.reducersApplied.append(reducer)
            plan.requiresBridge = true
            plan.budget = try await calculateBudget(
                thread: LogicalThread(state: plan.state, configuration: threadConfiguration),
                durableMemory: plan.durableMemory,
                retrievedMemory: plan.retrievedMemory,
                recentTail: plan.recentTail,
                prompt: plan.currentPrompt,
                schemaDescription: plan.schemaDescription
            )

            if plan.budget.projectedTotalTokens <= plan.budget.softLimitTokens {
                break
            }
        }

        logger.compaction(
            "before=\(initialPlan.budget.projectedTotalTokens) after=\(plan.budget.projectedTotalTokens) reducers=\(plan.reducersApplied.map(\.rawValue).joined(separator: ","))"
        )
        return plan
    }

    private func reducers(for mode: CompactionMode) -> [ReducerKind] {
        switch mode {
        case .slidingWindow:
            return [.toolPayloadDigester, .dropLowPriorityRetrievedMemory, .slidingTail, .emergencyReset]
        case .structuredSummary:
            return [.toolPayloadDigester, .dropLowPriorityRetrievedMemory, .structuredSummary, .slidingTail, .emergencyReset]
        case .hybrid:
            return [.toolPayloadDigester, .dropLowPriorityRetrievedMemory, .slidingTail, .structuredSummary, .emergencyReset]
        }
    }

    private func apply(
        reducer: ReducerKind,
        to plan: inout ContextPlan,
        threadConfiguration: ThreadConfiguration,
        options: CompactionOptions
    ) async throws -> Bool {
        switch reducer {
        case .toolPayloadDigester:
            return try await applyToolPayloadDigester(to: &plan)
        case .dropLowPriorityRetrievedMemory:
            guard plan.retrievedMemory.isEmpty == false else {
                return false
            }
            plan.retrievedMemory = []
            return true
        case .slidingTail:
            guard plan.recentTail.count > 4 else {
                return false
            }
            let reducedCount = max(4, plan.recentTail.count / 2)
            plan.recentTail = Array(plan.recentTail.suffix(reducedCount))
            return true
        case .structuredSummary:
            return try await applyStructuredSummary(
                to: &plan,
                threadConfiguration: threadConfiguration,
                options: options
            )
        case .emergencyReset:
            let currentPrompt = plan.currentPrompt
            let reducedMemory = plan.durableMemory.filter { $0.pinned || $0.priority >= 900 }
            let reducedTail = Array(plan.state.turns.suffix(2))
            let changed = reducedMemory.count != plan.durableMemory.count || reducedTail.count != plan.recentTail.count
            plan.durableMemory = reducedMemory
            plan.retrievedMemory = []
            plan.recentTail = reducedTail
            plan.currentPrompt = currentPrompt
            return changed
        }
    }

    private func applyToolPayloadDigester(
        to plan: inout ContextPlan
    ) async throws -> Bool {
        var changed = false
        var updatedTurns: [NormalizedTurn] = []
        var blobRecords: [DurableMemoryRecord] = []

        for turn in plan.state.turns {
            guard turn.blobIDs.isEmpty else {
                updatedTurns.append(turn)
                continue
            }

            let byteCount = turn.text.lengthOfBytes(using: .utf8)
            guard byteCount > configuration.memory.inlineBlobByteLimit else {
                updatedTurns.append(turn)
                continue
            }

            let blobID = try await configuration.persistence.blobs.put(Data(turn.text.utf8))
            let digest = String(turn.text.prefix(240))
            updatedTurns.append(
                NormalizedTurn(
                    id: turn.id,
                    role: turn.role,
                    text: "[spilled blob \(blobID.uuidString)] \(digest)",
                    createdAt: turn.createdAt,
                    priority: turn.priority,
                    tags: turn.tags,
                    blobIDs: [blobID],
                    windowIndex: turn.windowIndex,
                    compacted: turn.compacted
                )
            )
            blobRecords.append(
                DurableMemoryRecord(
                    kind: .blobRef,
                    text: "Blob \(blobID.uuidString) from \(turn.role.rawValue) turn",
                    priority: 250,
                    tags: ["blob"],
                    blobIDs: [blobID]
                )
            )
            changed = true
        }

        guard changed else {
            return false
        }
        plan.state.turns = updatedTurns
        plan.durableMemory.append(contentsOf: blobRecords)
        plan.spilledBlobCount += blobRecords.count
        return true
    }

    private func applyStructuredSummary(
        to plan: inout ContextPlan,
        threadConfiguration: ThreadConfiguration,
        options: CompactionOptions
    ) async throws -> Bool {
        let endpoint = threadConfiguration.runtime.structuredOutput ?? threadConfiguration.runtime.inference
        guard configuration.structuredBackends[endpoint.backendID] != nil else {
            logger.compaction("structured summary unavailable for \(endpoint.backendID), downgrading to sliding window")
            return false
        }

        let tailIDs = Set(plan.recentTail.map(\.id))
        let candidates = plan.state.turns.filter { tailIDs.contains($0.id) == false && $0.compacted == false }
        guard candidates.isEmpty == false else {
            return false
        }

        let summaryPrompt = candidates
            .map { "\($0.role.rawValue.capitalized): \($0.text)" }
            .joined(separator: "\n")

        do {
            let response = try await generateStructured(
                prepared: PreparedRequest(
                    thread: LogicalThread(state: plan.state, configuration: threadConfiguration),
                    plan: plan,
                    bridge: nil
                ),
                prompt: """
                Compact this conversation history conservatively. Preserve only durable facts, constraints, decisions, open tasks, and a short summaryText. Do not invent information.

                Transcript:
                \(summaryPrompt)
                """,
                spec: compactionSpec
            )
            plan.state.turns = plan.state.turns.map { turn in
                guard candidates.contains(where: { $0.id == turn.id }) else {
                    return turn
                }
                return NormalizedTurn(
                    id: turn.id,
                    role: turn.role,
                    text: turn.text,
                    createdAt: turn.createdAt,
                    priority: turn.priority,
                    tags: turn.tags,
                    blobIDs: turn.blobIDs,
                    windowIndex: turn.windowIndex,
                    compacted: true
                )
            }
            let summaryText = renderSummary(response.value)
            plan.state.turns.append(
                NormalizedTurn(
                    role: .summary,
                    text: summaryText,
                    priority: 500,
                    tags: ["summary"],
                    windowIndex: plan.state.activeWindowIndex
                )
            )
            if options.allowMemoryExtraction {
                plan.durableMemory = merge(summary: response.value, into: plan.durableMemory)
            }
            plan.summaryCreated = true
            return true
        } catch let error as RuntimeError {
            if case .unsupportedCapability = error {
                logger.compaction("structured summary translator unavailable for \(endpoint.backendID), downgrading to sliding window")
                return false
            }
            throw error
        }
    }

    private func finalizeTextResponse(
        prompt: String,
        text: String,
        prepared: inout PreparedRequest,
        bridge: BridgeReport?
    ) async throws -> TextReply {
        try Task.checkCancellation()
        let compaction = makeCompactionReport(from: prepared.plan)
        let metadata = TurnMetadata(
            budget: prepared.plan.budget,
            compaction: compaction,
            bridge: bridge
        )
        capture(
            prompt: prompt,
            responseText: text,
            thread: &prepared.thread,
            budget: prepared.plan.budget,
            compaction: compaction,
            bridge: bridge
        )
        try await persist(thread: prepared.thread, durableMemory: prepared.plan.durableMemory)
        return TextReply(text: text, metadata: metadata)
    }

    private func finalizeStructuredResponse<Value: Sendable>(
        prompt: String,
        result: StructuredGenerationResult<Value>,
        prepared: inout PreparedRequest,
        bridge: BridgeReport?,
        transcriptRenderer: @Sendable (Value) -> String
    ) async throws -> StructuredReply<Value> {
        try Task.checkCancellation()
        let compaction = makeCompactionReport(from: prepared.plan)
        let metadata = TurnMetadata(
            budget: prepared.plan.budget,
            compaction: compaction,
            bridge: bridge
        )
        capture(
            prompt: prompt,
            responseText: transcriptRenderer(result.value),
            thread: &prepared.thread,
            budget: prepared.plan.budget,
            compaction: compaction,
            bridge: bridge
        )
        try await persist(thread: prepared.thread, durableMemory: prepared.plan.durableMemory)
        return StructuredReply(
            value: result.value,
            transcriptText: result.transcriptText,
            metadata: metadata
        )
    }

    private func capture(
        prompt: String,
        responseText: String,
        thread: inout LogicalThread,
        budget: BudgetReport,
        compaction: CompactionReport?,
        bridge: BridgeReport?
    ) {
        let windowIndex = thread.state.activeWindowIndex
        thread.state.turns.append(
            NormalizedTurn(
                role: .user,
                text: prompt,
                priority: 950,
                tags: ["prompt"],
                windowIndex: windowIndex
            )
        )
        thread.state.turns.append(
            NormalizedTurn(
                role: .assistant,
                text: responseText,
                priority: 800,
                tags: ["response"],
                windowIndex: windowIndex
            )
        )
        thread.state.lastBudget = budget
        thread.state.lastCompaction = compaction
        thread.state.lastBridge = bridge
        thread.state.updatedAt = Date()
        threads[thread.state.threadID] = thread
    }

    private func session(
        for thread: LogicalThread,
        durableMemory: [DurableMemoryRecord],
        recentTail: [NormalizedTurn],
        forceBridge: Bool
    ) async throws -> any InferenceSession {
        if forceBridge == false, let existing = liveSessions[thread.state.threadID] {
            return existing.handle
        }

        let seed = bridgeSeedBuilder.makeSeed(
            for: thread.state,
            durableMemory: durableMemory,
            recentTail: recentTail
        )
        let backend = try await configuration.runtimeRegistry.backend(for: thread.configuration.runtime.inference.backendID)
        let handle = try await backend.makeSession(
            endpoint: thread.configuration.runtime.inference,
            instructions: seed.instructions,
            locale: thread.configuration.locale
        )
        liveSessions[thread.state.threadID] = WindowSession(
            windowIndex: thread.state.activeWindowIndex,
            handle: handle
        )
        return handle
    }

    private func validateAvailability(for logicalThread: LogicalThread) async throws {
        let backend = try await configuration.runtimeRegistry.backend(for: logicalThread.configuration.runtime.inference.backendID)
        let availability = await backend.availability(for: logicalThread.configuration.runtime.inference)
        switch availability.status {
        case .available:
            return
        case .unavailable(let reason):
            throw RuntimeError.unavailable(reason)
        }
    }

    private func resolvedContextWindowTokens(for endpoint: ModelEndpoint) async throws -> Int {
        if let override = endpoint.contextWindowOverride {
            return override
        }
        let backend = try await configuration.runtimeRegistry.backend(for: endpoint.backendID)
        return await backend.contextWindowTokens(for: endpoint) ?? configuration.budget.defaultContextWindowTokens
    }

    private func exactEstimator(for endpoint: ModelEndpoint) async throws -> (any TokenEstimating)? {
        let backend = try await configuration.runtimeRegistry.backend(for: endpoint.backendID)
        return await backend.exactTokenEstimator(for: endpoint)
    }

    private func renderedPrompt(
        instructions: String?,
        durableMemory: [DurableMemoryRecord],
        retrievedMemory: [DurableMemoryRecord],
        recentTail: [NormalizedTurn],
        currentPrompt: String,
        schemaDescription: String?
    ) -> RenderedPrompt {
        RenderedPrompt(
            instructions: instructions,
            durableMemory: durableMemory.map { "[\($0.kind.rawValue)] \($0.text)" },
            retrievedMemory: retrievedMemory.map { "[\($0.kind.rawValue)] \($0.text)" },
            recentTail: recentTail.map { "\($0.role.rawValue.capitalized): \($0.text)" },
            currentPrompt: currentPrompt,
            schemaText: schemaDescription
        )
    }

    private func loadedThreadState(threadID: String) async throws -> PersistedThreadState? {
        if let thread = threads[threadID] {
            return thread.state
        }
        return try await configuration.persistence.threads.load(threadID: threadID)
    }

    private func requireThread(_ threadID: String) async throws -> LogicalThread {
        if let thread = threads[threadID] {
            return thread
        }
        if let persisted = try await configuration.persistence.threads.load(threadID: threadID) {
            let configuration = ThreadConfiguration(
                runtime: persisted.runtime,
                instructions: persisted.instructions,
                locale: persisted.locale
            )
            let thread = LogicalThread(state: persisted, configuration: configuration)
            threads[threadID] = thread
            return thread
        }
        throw ContextManagerError.threadNotFound(threadID)
    }

    private func requireThreadState(threadID: String) async throws -> PersistedThreadState {
        guard let state = try await loadedThreadState(threadID: threadID) else {
            throw ContextManagerError.threadNotFound(threadID)
        }
        return state
    }

    private func saveThreadState(
        _ state: PersistedThreadState,
        threadID: String
    ) async throws {
        do {
            try await configuration.persistence.threads.save(state, threadID: threadID)
        } catch {
            logger.error("persist failed for thread \(threadID): \(error.localizedDescription)")
            throw ContextManagerError.persistenceFailed(error.localizedDescription)
        }
    }

    private func saveMemories(
        _ durableMemory: [DurableMemoryRecord],
        threadID: String
    ) async throws {
        do {
            try await configuration.persistence.memories.save(durableMemory, threadID: threadID)
        } catch {
            logger.error("persist memories failed for thread \(threadID): \(error.localizedDescription)")
            throw ContextManagerError.persistenceFailed(error.localizedDescription)
        }
    }

    private func persist(
        thread: LogicalThread,
        durableMemory: [DurableMemoryRecord]
    ) async throws {
        try await saveThreadState(thread.state, threadID: thread.state.threadID)
        try await saveMemories(durableMemory, threadID: thread.state.threadID)
        threads[thread.state.threadID] = thread
    }

    private func deleteBlobs(ids: Set<UUID>) async throws {
        for id in ids {
            try await configuration.persistence.blobs.delete(id)
        }
    }

    private func makeCompactionReport(from plan: ContextPlan) -> CompactionReport? {
        guard plan.reducersApplied.isEmpty == false else {
            return nil
        }
        return CompactionReport(
            mode: configuration.compaction.mode,
            tokensBefore: plan.originalProjectedTotalTokens ?? plan.budget.projectedTotalTokens,
            tokensAfter: plan.budget.projectedTotalTokens,
            reducersApplied: plan.reducersApplied,
            summaryCreated: plan.summaryCreated,
            spilledBlobCount: plan.spilledBlobCount
        )
    }

    private func makeDiagnostics(
        threadID: String,
        state: PersistedThreadState,
        durableMemory: [DurableMemoryRecord],
        budget: BudgetReport?,
        compaction: CompactionReport?,
        bridge: BridgeReport?
    ) -> SessionDiagnostics {
        SessionDiagnostics(
            sessionID: threadID,
            windowIndex: state.activeWindowIndex,
            lastBudget: budget,
            lastCompaction: compaction,
            lastBridge: bridge,
            turnCount: state.turns.count,
            durableMemoryCount: durableMemory.count,
            blobCount: Set(state.turns.flatMap(\.blobIDs) + durableMemory.flatMap(\.blobIDs)).count
        )
    }

    private func recentTail(from turns: [NormalizedTurn]) -> [NormalizedTurn] {
        let visible = turns.filter { $0.compacted == false || $0.role == .summary }
        guard configuration.compaction.maxRecentTurns > 0 else {
            return []
        }
        return Array(visible.suffix(configuration.compaction.maxRecentTurns))
    }

    private func deduplicatedMemories(_ records: [DurableMemoryRecord]) -> [DurableMemoryRecord] {
        var seen: Set<String> = []
        var unique: [DurableMemoryRecord] = []
        for record in records {
            let key = "\(record.kind.rawValue)\u{1F}\(record.text)"
            guard seen.insert(key).inserted else {
                continue
            }
            unique.append(record)
        }
        return unique
    }

    private func deduplicatedImportedTurns(_ turns: [NormalizedTurn]) -> [NormalizedTurn] {
        var seenIDs: Set<UUID> = []
        var seenKeys: Set<ImportedTurnKey> = []
        var unique: [NormalizedTurn] = []
        for turn in turns.sorted(by: Self.sortTurnsByCreatedAt) {
            let key = ImportedTurnKey(turn: turn)
            guard seenIDs.insert(turn.id).inserted, seenKeys.insert(key).inserted else {
                continue
            }
            unique.append(turn)
        }
        return unique
    }

    private static func sortTurnsByCreatedAt(
        lhs: NormalizedTurn,
        rhs: NormalizedTurn
    ) -> Bool {
        if lhs.createdAt != rhs.createdAt {
            return lhs.createdAt < rhs.createdAt
        }
        return lhs.id.uuidString < rhs.id.uuidString
    }

    private func mapExactBreakdown(
        _ breakdown: [RenderedPromptSection: Int]
    ) -> [BudgetComponent: Int] {
        var mapped: [BudgetComponent: Int] = [:]
        for (section, tokens) in breakdown {
            let key: BudgetComponent
            switch section {
            case .instructions:
                key = .instructions
            case .durableMemory:
                key = .durableMemory
            case .retrievedMemory:
                key = .retrievedMemory
            case .recentTail:
                key = .recentTail
            case .currentPrompt:
                key = .currentPrompt
            case .schema:
                key = .schema
            }
            mapped[key] = tokens
        }
        return mapped
    }

    private func merge(
        summary: CompactionEnvelope,
        into records: [DurableMemoryRecord]
    ) -> [DurableMemoryRecord] {
        var merged = records
        merged.append(contentsOf: summary.stableFacts.map {
            DurableMemoryRecord(kind: .fact, text: "\($0.key): \($0.value)", priority: 900)
        })
        merged.append(contentsOf: summary.userConstraints.map {
            DurableMemoryRecord(kind: .constraint, text: $0, priority: 850)
        })
        merged.append(contentsOf: summary.decisions.map {
            DurableMemoryRecord(kind: .decision, text: $0, priority: 850)
        })
        merged.append(contentsOf: summary.openTasks.map {
            DurableMemoryRecord(kind: .openTask, text: "\($0.description) [\($0.status)]", priority: 800)
        })
        if summary.summaryText.isEmpty == false {
            merged.append(
                DurableMemoryRecord(
                    kind: .summary,
                    text: summary.summaryText,
                    priority: 700
                )
            )
        }
        return deduplicatedMemories(merged)
    }

    private func renderSummary(_ summary: CompactionEnvelope) -> String {
        if summary.summaryText.isEmpty == false {
            return summary.summaryText
        }
        var lines: [String] = []
        if summary.stableFacts.isEmpty == false {
            lines.append("Facts: " + summary.stableFacts.map { "\($0.key)=\($0.value)" }.joined(separator: "; "))
        }
        if summary.userConstraints.isEmpty == false {
            lines.append("Constraints: " + summary.userConstraints.joined(separator: "; "))
        }
        if summary.decisions.isEmpty == false {
            lines.append("Decisions: " + summary.decisions.joined(separator: "; "))
        }
        if summary.openTasks.isEmpty == false {
            lines.append("Open tasks: " + summary.openTasks.map { "\($0.description) [\($0.status)]" }.joined(separator: "; "))
        }
        if summary.entities.isEmpty == false {
            lines.append("Entities: " + summary.entities.map { "\($0.name) (\($0.type))" }.joined(separator: "; "))
        }
        return lines.joined(separator: "\n")
    }

    private func errorDescription(_ error: RuntimeError) -> String {
        switch error {
        case .unavailable(let value),
             .unsupportedCapability(let value),
             .unsupportedLocale(let value),
             .contextOverflow(let value),
             .refusal(let value),
             .generationFailed(let value),
             .transportFailed(let value):
            return value
        }
    }
}

private struct LogicalThread: Sendable {
    var state: PersistedThreadState
    var configuration: ThreadConfiguration
}

private struct WindowSession: Sendable {
    var windowIndex: Int
    var handle: any InferenceSession
}

private struct ContextPlan: Sendable {
    var state: PersistedThreadState
    var durableMemory: [DurableMemoryRecord]
    var retrievedMemory: [DurableMemoryRecord]
    var recentTail: [NormalizedTurn]
    var currentPrompt: String
    var schemaDescription: String?
    var budget: BudgetReport
    var originalProjectedTotalTokens: Int?
    var summaryCreated = false
    var spilledBlobCount = 0
    var reducersApplied: [ReducerKind] = []
    var requiresBridge = false
}

private struct PreparedRequest: Sendable {
    var thread: LogicalThread
    var plan: ContextPlan
    var bridge: BridgeReport?
}

private struct CompactionOptions: Sendable {
    var force: Bool
    var allowMemoryExtraction: Bool

    static func standard(memoryPolicy: MemoryPolicy) -> CompactionOptions {
        CompactionOptions(
            force: false,
            allowMemoryExtraction: memoryPolicy.automaticallyExtractMemories
        )
    }

    static func manual(memoryPolicy: MemoryPolicy) -> CompactionOptions {
        CompactionOptions(
            force: true,
            allowMemoryExtraction: memoryPolicy.automaticallyExtractMemories
        )
    }
}

private struct HeuristicBudgetEstimator: Sendable {
    let perMessageOverhead: Int = 6

    func estimate(
        instructions: String?,
        durableMemory: [DurableMemoryRecord],
        retrievedMemory: [DurableMemoryRecord],
        recentTail: [NormalizedTurn],
        currentPrompt: String,
        schemaDescription: String?,
        budgetPolicy: BudgetPolicy,
        contextWindowTokens: Int
    ) -> BudgetReport {
        var breakdown: [BudgetComponent: Int] = [:]
        breakdown[.instructions] = tokenEstimate(instructions ?? "", safetyMultiplier: budgetPolicy.heuristicSafetyMultiplier)
        breakdown[.durableMemory] = durableMemory.reduce(0) { $0 + tokenEstimate($1.text, safetyMultiplier: budgetPolicy.heuristicSafetyMultiplier) }
        breakdown[.retrievedMemory] = retrievedMemory.reduce(0) { $0 + tokenEstimate($1.text, safetyMultiplier: budgetPolicy.heuristicSafetyMultiplier) }
        breakdown[.recentTail] = recentTail.reduce(0) { $0 + tokenEstimate($1.text, safetyMultiplier: budgetPolicy.heuristicSafetyMultiplier) }
        breakdown[.currentPrompt] = tokenEstimate(currentPrompt, safetyMultiplier: budgetPolicy.heuristicSafetyMultiplier)
        breakdown[.schema] = tokenEstimate(schemaDescription ?? "", safetyMultiplier: budgetPolicy.heuristicSafetyMultiplier)
        breakdown[.outputReserve] = budgetPolicy.reservedOutputTokens

        let inputTokens = breakdown
            .filter { $0.key != .outputReserve }
            .map(\.value)
            .reduce(0, +)
        let projected = inputTokens + budgetPolicy.reservedOutputTokens
        return BudgetReport(
            accuracy: .approximate,
            contextWindowTokens: contextWindowTokens,
            estimatedInputTokens: inputTokens,
            reservedOutputTokens: budgetPolicy.reservedOutputTokens,
            projectedTotalTokens: projected,
            softLimitTokens: Int(Double(contextWindowTokens) * budgetPolicy.preemptiveCompactionFraction),
            emergencyLimitTokens: Int(Double(contextWindowTokens) * budgetPolicy.emergencyFraction),
            breakdown: breakdown
        )
    }

    private func tokenEstimate(
        _ text: String,
        safetyMultiplier: Double
    ) -> Int {
        guard text.isEmpty == false else {
            return 0
        }
        let bytes = text.lengthOfBytes(using: .utf8)
        let words = text.split { $0.isWhitespace || $0.isNewline }.count
        let base = max(
            Int(ceil(Double(bytes) / 4.0)),
            Int(ceil(Double(words) * 1.35))
        )
        return Int(ceil(Double(base + perMessageOverhead) * safetyMultiplier))
    }
}

private struct BridgeSeedBuilder: Sendable {
    func makeSeed(
        for thread: PersistedThreadState,
        durableMemory: [DurableMemoryRecord],
        recentTail: [NormalizedTurn]
    ) -> SessionSeed {
        guard durableMemory.isEmpty == false || recentTail.isEmpty == false else {
            return SessionSeed(instructions: thread.instructions)
        }

        let memoryText = durableMemory
            .sorted { $0.priority > $1.priority }
            .prefix(12)
            .map { "[\($0.kind.rawValue)] \($0.text)" }
            .joined(separator: "\n")
        let tailText = recentTail
            .map { "\($0.role.rawValue.capitalized): \($0.text)" }
            .joined(separator: "\n")

        let instructions = """
        \(thread.instructions ?? "")

        Prior logical thread context:
        Pinned and durable memory:
        \(memoryText.isEmpty ? "None" : memoryText)

        Recent turns:
        \(tailText.isEmpty ? "None" : tailText)
        """

        return SessionSeed(
            instructions: instructions.trimmingCharacters(in: .whitespacesAndNewlines)
        )
    }
}

private struct SessionSeed: Sendable {
    var instructions: String?
}

private struct ImportedTurnKey: Hashable {
    let role: NormalizedTurn.Role
    let text: String
    let createdAt: Date
    let priority: Int
    let tags: [String]
    let blobIDs: [UUID]
    let windowIndex: Int
    let compacted: Bool

    init(turn: NormalizedTurn) {
        role = turn.role
        text = turn.text
        createdAt = turn.createdAt
        priority = turn.priority
        tags = turn.tags
        blobIDs = turn.blobIDs
        windowIndex = turn.windowIndex
        compacted = turn.compacted
    }
}

private struct CompactionEnvelope: Codable, Sendable, Equatable {
    var summaryText: String
    var stableFacts: [CompactionFact]
    var userConstraints: [String]
    var openTasks: [CompactionTask]
    var decisions: [String]
    var entities: [CompactionEntity]
    var retrievalHints: [String]
}

private struct CompactionFact: Codable, Sendable, Equatable {
    var key: String
    var value: String
}

private struct CompactionTask: Codable, Sendable, Equatable {
    var description: String
    var status: String
}

private struct CompactionEntity: Codable, Sendable, Equatable {
    var name: String
    var type: String
}

private let compactionSpec = StructuredOutputSpec<CompactionEnvelope>(
    schema: .object(
        ObjectSchema(
            name: "CompactionEnvelope",
            description: "Conservative structured summary of durable thread state.",
            properties: [
                .init(name: "summaryText", schema: .string()),
                .init(
                    name: "stableFacts",
                    schema: .array(
                        .init(
                            item: .object(
                                ObjectSchema(
                                    name: "CompactionFact",
                                    properties: [
                                        .init(name: "key", schema: .string()),
                                        .init(name: "value", schema: .string())
                                    ]
                                )
                            )
                        )
                    )
                ),
                .init(name: "userConstraints", schema: .array(.init(item: .string()))),
                .init(
                    name: "openTasks",
                    schema: .array(
                        .init(
                            item: .object(
                                ObjectSchema(
                                    name: "CompactionTask",
                                    properties: [
                                        .init(name: "description", schema: .string()),
                                        .init(name: "status", schema: .string())
                                    ]
                                )
                            )
                        )
                    )
                ),
                .init(name: "decisions", schema: .array(.init(item: .string()))),
                .init(
                    name: "entities",
                    schema: .array(
                        .init(
                            item: .object(
                                ObjectSchema(
                                    name: "CompactionEntity",
                                    properties: [
                                        .init(name: "name", schema: .string()),
                                        .init(name: "type", schema: .string())
                                    ]
                                )
                            )
                        )
                    )
                ),
                .init(name: "retrievalHints", schema: .array(.init(item: .string())))
            ]
        )
    ),
    decode: { data in
        try JSONDecoder().decode(CompactionEnvelope.self, from: data)
    },
    transcriptRenderer: { envelope in
        envelope.summaryText
    }
)

private extension PersistedThreadState {
    var locale: Locale? {
        localeIdentifier.map(Locale.init(identifier:))
    }
}
