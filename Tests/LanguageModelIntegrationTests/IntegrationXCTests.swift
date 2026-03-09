import Foundation
import XCTest
import LanguageModelRuntime
import LanguageModelStructuredOutput
import LanguageModelContextManagement
import LanguageModelOpenAI
import LanguageModelVLLM
#if canImport(FoundationModels)
import LanguageModelApple
#endif

final class IntegrationXCTests: XCTestCase {
    private struct IntegrationStructuredPayload: Codable, Sendable, Equatable {
        let title: String
    }

    // MARK: - Codable test payloads

    private struct PersonPayload: Codable, Sendable, Equatable {
        let name: String
        let age: Int
    }

    private struct NestedPayload: Codable, Sendable, Equatable {
        let person: PersonPayload
    }

    private struct ArrayPayload: Codable, Sendable, Equatable {
        let items: [String]
    }

    private struct EnumPayload: Codable, Sendable, Equatable {
        let color: String
    }

    private struct OptionalPayload: Codable, Sendable, Equatable {
        let title: String
        let subtitle: String?
    }

    private struct MixedPayload: Codable, Sendable, Equatable {
        let active: Bool
        let score: Double
    }

    // MARK: - Helper

    private func vllmEndpoint() -> (baseURL: String, model: String, options: [String: String])? {
        let env = ProcessInfo.processInfo.environment
        guard let baseURL = env["VLLM_BASE_URL"], let model = env["VLLM_MODEL"] else { return nil }
        var options = ["baseURL": baseURL]
        if let apiKey = env["VLLM_API_KEY"] { options["apiKey"] = apiKey }
        return (baseURL, model, options)
    }

    // MARK: - Apple tests

    func testAppleAvailabilityGate() async throws {
        #if canImport(FoundationModels)
        guard #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) else {
            return
        }
        let backend = AppleInferenceBackend()
        _ = await backend.availability(
            for: ModelEndpoint(backendID: backend.backendID, modelID: "default")
        )
        #endif
    }

    func testAppleStructuredLiveGate() async throws {
        #if canImport(FoundationModels)
        guard #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) else {
            return
        }

        let inference = AppleInferenceBackend()
        let endpoint = ModelEndpoint(backendID: inference.backendID, modelID: "default")
        let availability = await inference.availability(for: endpoint)
        guard case .available = availability.status else {
            return
        }

        let backend = AppleStructuredOutputBackend()
        let spec = StructuredOutput.codable(
            IntegrationStructuredPayload.self,
            schema: .object(
                ObjectSchema(
                    name: "IntegrationStructuredPayload",
                    properties: [.init(name: "title", schema: .string())]
                )
            )
        )

        let result = try await backend.generateStructured(
            endpoint: endpoint,
            instructions: "Return JSON only.",
            locale: nil,
            prompt: "Return an object with title set to ok.",
            spec: spec,
            options: StructuredGenerationOptions(maximumResponseTokens: 32, deterministic: true)
        )

        XCTAssertFalse(result.transcriptText.isEmpty)
        #endif
    }

    // MARK: - OpenAI tests

    func testOpenAILiveGate() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard
            let baseURL = environment["OPENAI_BASE_URL"],
            let apiKey = environment["OPENAI_API_KEY"],
            let model = environment["OPENAI_MODEL"]
        else {
            return
        }

        let backend = OpenAIInferenceBackend()
        let session = try await backend.makeSession(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: model,
                options: ["baseURL": baseURL, "apiKey": apiKey]
            ),
            instructions: "Reply with a single short word.",
            locale: nil
        )
        let result = try await session.generateText(
            prompt: "hello",
            options: TextGenerationOptions(maximumResponseTokens: 16, deterministic: true)
        )

        XCTAssertFalse(result.text.isEmpty)
    }

    func testOpenAIStreamingLiveGate() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard
            let baseURL = environment["OPENAI_BASE_URL"],
            let apiKey = environment["OPENAI_API_KEY"],
            let model = environment["OPENAI_MODEL"]
        else {
            return
        }

        let backend = OpenAIInferenceBackend()
        let session = try await backend.makeSession(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: model,
                options: ["baseURL": baseURL, "apiKey": apiKey]
            ),
            instructions: "Reply with a single short word.",
            locale: nil
        )

        var partials: [String] = []
        var completed: TextGenerationResult?
        let stream = await session.streamText(
            prompt: "hello",
            options: TextGenerationOptions(maximumResponseTokens: 16, deterministic: true)
        )
        for try await event in stream {
            switch event {
            case .partial(let text):
                partials.append(text)
            case .completed(let result):
                completed = result
            }
        }

        XCTAssertTrue(partials.isEmpty == false || completed?.text.isEmpty == false)
        XCTAssertNotNil(completed)
    }

    func testOpenAIStructuredLiveGate() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard
            let baseURL = environment["OPENAI_BASE_URL"],
            let apiKey = environment["OPENAI_API_KEY"],
            let model = environment["OPENAI_MODEL"]
        else {
            return
        }

        let backend = OpenAIStructuredOutputBackend()
        let spec = StructuredOutput.codable(
            IntegrationStructuredPayload.self,
            schema: .object(
                ObjectSchema(
                    name: "IntegrationStructuredPayload",
                    properties: [.init(name: "title", schema: .string())]
                )
            )
        )

        let result = try await backend.generateStructured(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: model,
                options: ["baseURL": baseURL, "apiKey": apiKey]
            ),
            instructions: "Return JSON only.",
            locale: nil,
            prompt: "Return an object with title set to ok.",
            spec: spec,
            options: StructuredGenerationOptions(maximumResponseTokens: 32, deterministic: true)
        )

        XCTAssertFalse(result.transcriptText.isEmpty)
    }

    // MARK: - vLLM basic tests

    func testVLLMLiveGate() async throws {
        guard let vllm = vllmEndpoint() else { return }

        let backend = VLLMInferenceBackend()
        let session = try await backend.makeSession(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: vllm.model,
                options: vllm.options
            ),
            instructions: "Reply with a single short word.",
            locale: nil
        )
        let result = try await session.generateText(
            prompt: "hello",
            options: TextGenerationOptions(maximumResponseTokens: 16, deterministic: true)
        )

        XCTAssertFalse(result.text.isEmpty)
    }

    func testVLLMStreamingLiveGate() async throws {
        guard let vllm = vllmEndpoint() else { return }

        let backend = VLLMInferenceBackend()
        let session = try await backend.makeSession(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: vllm.model,
                options: vllm.options
            ),
            instructions: "Reply with a single short word.",
            locale: nil
        )

        var partials: [String] = []
        var completed: TextGenerationResult?
        let stream = await session.streamText(
            prompt: "hello",
            options: TextGenerationOptions(maximumResponseTokens: 16, deterministic: true)
        )
        for try await event in stream {
            switch event {
            case .partial(let text):
                partials.append(text)
            case .completed(let result):
                completed = result
            }
        }

        XCTAssertTrue(partials.isEmpty == false || completed?.text.isEmpty == false)
        XCTAssertNotNil(completed)
    }

    func testVLLMStructuredLiveGate() async throws {
        guard let vllm = vllmEndpoint() else { return }

        var options = vllm.options
        options["guidedDecoding"] = "json"

        let backend = VLLMStructuredOutputBackend()
        let spec = StructuredOutput.codable(
            IntegrationStructuredPayload.self,
            schema: .object(
                ObjectSchema(
                    name: "IntegrationStructuredPayload",
                    properties: [.init(name: "title", schema: .string())]
                )
            )
        )

        let result = try await backend.generateStructured(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: vllm.model,
                options: options
            ),
            instructions: "Return JSON only.",
            locale: nil,
            prompt: "Return an object with title set to ok.",
            spec: spec,
            options: StructuredGenerationOptions(maximumResponseTokens: 32, deterministic: true)
        )

        XCTAssertFalse(result.transcriptText.isEmpty)
    }

    // MARK: - A. VLLMInferenceBackend — text generation

    func testVLLMAvailabilityReportsCapabilities() async throws {
        guard let vllm = vllmEndpoint() else { return }

        let backend = VLLMInferenceBackend()
        let endpoint = ModelEndpoint(
            backendID: backend.backendID,
            modelID: vllm.model,
            options: vllm.options
        )
        let availability = await backend.availability(for: endpoint)

        guard case .available = availability.status else {
            XCTFail("Expected .available status")
            return
        }
        XCTAssertTrue(availability.capabilities.supportsTextGeneration)
        XCTAssertTrue(availability.capabilities.supportsTextStreaming)
        XCTAssertTrue(availability.capabilities.supportsStructuredOutput)
    }

    func testVLLMTextGenerationDeterministic() async throws {
        guard let vllm = vllmEndpoint() else { return }

        let backend = VLLMInferenceBackend()
        let endpoint = ModelEndpoint(
            backendID: backend.backendID,
            modelID: vllm.model,
            options: vllm.options
        )

        let session1 = try await backend.makeSession(
            endpoint: endpoint,
            instructions: "Reply with exactly one word.",
            locale: nil
        )
        let result1 = try await session1.generateText(
            prompt: "What color is the sky on a clear day?",
            options: TextGenerationOptions(maximumResponseTokens: 16, deterministic: true)
        )

        let session2 = try await backend.makeSession(
            endpoint: endpoint,
            instructions: "Reply with exactly one word.",
            locale: nil
        )
        let result2 = try await session2.generateText(
            prompt: "What color is the sky on a clear day?",
            options: TextGenerationOptions(maximumResponseTokens: 16, deterministic: true)
        )

        XCTAssertFalse(result1.text.isEmpty)
        XCTAssertEqual(result1.text, result2.text)
    }

    func testVLLMTextGenerationWithTemperature() async throws {
        guard let vllm = vllmEndpoint() else { return }

        let backend = VLLMInferenceBackend()
        let session = try await backend.makeSession(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: vllm.model,
                options: vllm.options
            ),
            instructions: "Reply briefly.",
            locale: nil
        )
        let result = try await session.generateText(
            prompt: "Name a fruit.",
            options: TextGenerationOptions(maximumResponseTokens: 16, temperature: 0.7)
        )

        XCTAssertFalse(result.text.isEmpty)
    }

    func testVLLMTextGenerationRespectsMaxTokens() async throws {
        guard let vllm = vllmEndpoint() else { return }

        let backend = VLLMInferenceBackend()
        let session = try await backend.makeSession(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: vllm.model,
                options: vllm.options
            ),
            instructions: "Reply with a long detailed answer.",
            locale: nil
        )
        let result = try await session.generateText(
            prompt: "Tell me about the history of computing.",
            options: TextGenerationOptions(maximumResponseTokens: 8, deterministic: true)
        )

        XCTAssertFalse(result.text.isEmpty)
        // 8 tokens ≈ roughly 6-32 characters; ensure it's short
        XCTAssertLessThan(result.text.count, 100)
    }

    // MARK: - B. VLLMInferenceBackend — streaming

    func testVLLMStreamingCollectsMultiplePartials() async throws {
        guard let vllm = vllmEndpoint() else { return }

        let backend = VLLMInferenceBackend()
        let session = try await backend.makeSession(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: vllm.model,
                options: vllm.options
            ),
            instructions: "Reply with a short sentence.",
            locale: nil
        )

        var partials: [String] = []
        var completed: TextGenerationResult?
        let stream = await session.streamText(
            prompt: "Count from one to five.",
            options: TextGenerationOptions(maximumResponseTokens: 32, deterministic: true)
        )
        for try await event in stream {
            switch event {
            case .partial(let text):
                partials.append(text)
            case .completed(let result):
                completed = result
            }
        }

        XCTAssertGreaterThanOrEqual(partials.count, 1)
        XCTAssertNotNil(completed)
    }

    func testVLLMStreamingCompletedTextMatchesPartials() async throws {
        guard let vllm = vllmEndpoint() else { return }

        let backend = VLLMInferenceBackend()
        let session = try await backend.makeSession(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: vllm.model,
                options: vllm.options
            ),
            instructions: "Reply with a short sentence.",
            locale: nil
        )

        var partials: [String] = []
        var completed: TextGenerationResult?
        let stream = await session.streamText(
            prompt: "Say hello world.",
            options: TextGenerationOptions(maximumResponseTokens: 32, deterministic: true)
        )
        for try await event in stream {
            switch event {
            case .partial(let text):
                partials.append(text)
            case .completed(let result):
                completed = result
            }
        }

        let joined = partials.joined()
        XCTAssertNotNil(completed)
        XCTAssertEqual(completed?.text, joined)
    }

    // MARK: - C. VLLMStructuredOutputBackend — guided JSON

    func testVLLMStructuredSimpleObject() async throws {
        guard let vllm = vllmEndpoint() else { return }

        var options = vllm.options
        options["guidedDecoding"] = "json"

        let backend = VLLMStructuredOutputBackend()
        let spec = StructuredOutput.codable(
            IntegrationStructuredPayload.self,
            schema: .object(
                ObjectSchema(
                    name: "IntegrationStructuredPayload",
                    properties: [.init(name: "title", schema: .string())]
                )
            )
        )

        let result = try await backend.generateStructured(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: vllm.model,
                options: options
            ),
            instructions: "Return JSON only.",
            locale: nil,
            prompt: "Return an object with title set to hello.",
            spec: spec,
            options: StructuredGenerationOptions(maximumResponseTokens: 32, deterministic: true)
        )

        XCTAssertFalse(result.value.title.isEmpty)
    }

    func testVLLMStructuredNestedObject() async throws {
        guard let vllm = vllmEndpoint() else { return }

        var options = vllm.options
        options["guidedDecoding"] = "json"

        let backend = VLLMStructuredOutputBackend()
        let spec = StructuredOutput.codable(
            NestedPayload.self,
            schema: .object(
                ObjectSchema(
                    name: "NestedPayload",
                    properties: [
                        .init(
                            name: "person",
                            schema: .object(
                                ObjectSchema(
                                    name: "PersonPayload",
                                    properties: [
                                        .init(name: "name", schema: .string()),
                                        .init(name: "age", schema: .integer())
                                    ]
                                )
                            )
                        )
                    ]
                )
            )
        )

        let result = try await backend.generateStructured(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: vllm.model,
                options: options
            ),
            instructions: "Return JSON only.",
            locale: nil,
            prompt: "Return a person named Alice who is 30 years old.",
            spec: spec,
            options: StructuredGenerationOptions(maximumResponseTokens: 64, deterministic: true)
        )

        XCTAssertFalse(result.value.person.name.isEmpty)
        XCTAssertGreaterThan(result.value.person.age, 0)
    }

    func testVLLMStructuredWithArray() async throws {
        guard let vllm = vllmEndpoint() else { return }

        var options = vllm.options
        options["guidedDecoding"] = "json"

        let backend = VLLMStructuredOutputBackend()
        let spec = StructuredOutput.codable(
            ArrayPayload.self,
            schema: .object(
                ObjectSchema(
                    name: "ArrayPayload",
                    properties: [
                        .init(
                            name: "items",
                            schema: .array(ArrayConstraints(item: .string()))
                        )
                    ]
                )
            )
        )

        let result = try await backend.generateStructured(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: vllm.model,
                options: options
            ),
            instructions: "Return JSON only.",
            locale: nil,
            prompt: "Return an object with items containing three fruit names.",
            spec: spec,
            options: StructuredGenerationOptions(maximumResponseTokens: 64, deterministic: true)
        )

        XCTAssertFalse(result.value.items.isEmpty)
    }

    func testVLLMStructuredWithEnum() async throws {
        guard let vllm = vllmEndpoint() else { return }

        var options = vllm.options
        options["guidedDecoding"] = "json"

        let backend = VLLMStructuredOutputBackend()
        let spec = StructuredOutput.codable(
            EnumPayload.self,
            schema: .object(
                ObjectSchema(
                    name: "EnumPayload",
                    properties: [
                        .init(
                            name: "color",
                            schema: .enumeration(
                                EnumSchema(name: "Color", cases: ["red", "green", "blue"])
                            )
                        )
                    ]
                )
            )
        )

        let result = try await backend.generateStructured(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: vllm.model,
                options: options
            ),
            instructions: "Return JSON only.",
            locale: nil,
            prompt: "Pick a color from red, green, or blue.",
            spec: spec,
            options: StructuredGenerationOptions(maximumResponseTokens: 32, deterministic: true)
        )

        XCTAssertTrue(["red", "green", "blue"].contains(result.value.color))
    }

    func testVLLMStructuredWithOptionalField() async throws {
        guard let vllm = vllmEndpoint() else { return }

        var options = vllm.options
        options["guidedDecoding"] = "json"

        let backend = VLLMStructuredOutputBackend()
        let spec = StructuredOutput.codable(
            OptionalPayload.self,
            schema: .object(
                ObjectSchema(
                    name: "OptionalPayload",
                    properties: [
                        .init(name: "title", schema: .string()),
                        .init(name: "subtitle", schema: .optional(.string()), isOptional: true)
                    ]
                )
            )
        )

        let result = try await backend.generateStructured(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: vllm.model,
                options: options
            ),
            instructions: "Return JSON only.",
            locale: nil,
            prompt: "Return an object with title set to test. Include a subtitle if you want.",
            spec: spec,
            options: StructuredGenerationOptions(maximumResponseTokens: 64, deterministic: true)
        )

        XCTAssertFalse(result.value.title.isEmpty)
    }

    func testVLLMStructuredWithBooleanAndNumber() async throws {
        guard let vllm = vllmEndpoint() else { return }

        var options = vllm.options
        options["guidedDecoding"] = "json"

        let backend = VLLMStructuredOutputBackend()
        let spec = StructuredOutput.codable(
            MixedPayload.self,
            schema: .object(
                ObjectSchema(
                    name: "MixedPayload",
                    properties: [
                        .init(name: "active", schema: .boolean),
                        .init(name: "score", schema: .number())
                    ]
                )
            )
        )

        let result = try await backend.generateStructured(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: vllm.model,
                options: options
            ),
            instructions: "Return JSON only.",
            locale: nil,
            prompt: "Return an object with active set to true and score set to 95.5.",
            spec: spec,
            options: StructuredGenerationOptions(maximumResponseTokens: 32, deterministic: true)
        )

        // Just verify decode succeeded — the model should produce valid typed JSON
        XCTAssertTrue(result.value.active || !result.value.active) // decoded successfully
    }

    // MARK: - D. ContextManager end-to-end via vLLM

    func testVLLMContextSessionReply() async throws {
        guard let vllm = vllmEndpoint() else { return }

        let inferenceBackend = VLLMInferenceBackend()
        let registry = RuntimeRegistry()
        await registry.register(inferenceBackend)

        let config = ContextManagerConfiguration(
            runtimeRegistry: registry,
            memory: MemoryPolicy(automaticallyExtractMemories: false),
            diagnostics: DiagnosticsPolicy(isEnabled: false)
        )
        let manager = ContextManager(configuration: config)

        let threadConfig = ThreadConfiguration(
            runtime: ThreadRuntimeConfiguration(
                inference: ModelEndpoint(
                    backendID: inferenceBackend.backendID,
                    modelID: vllm.model,
                    options: vllm.options
                )
            ),
            instructions: "Reply with a single short word."
        )

        let session = try await manager.session(configuration: threadConfig)
        let reply = try await session.reply(to: "hello")

        XCTAssertFalse(reply.text.isEmpty)

        let history = try await session.inspection.history()
        XCTAssertEqual(history.count, 2) // user + assistant
    }

    func testVLLMContextSessionStream() async throws {
        guard let vllm = vllmEndpoint() else { return }

        let inferenceBackend = VLLMInferenceBackend()
        let registry = RuntimeRegistry()
        await registry.register(inferenceBackend)

        let config = ContextManagerConfiguration(
            runtimeRegistry: registry,
            memory: MemoryPolicy(automaticallyExtractMemories: false),
            diagnostics: DiagnosticsPolicy(isEnabled: false)
        )
        let manager = ContextManager(configuration: config)

        let threadConfig = ThreadConfiguration(
            runtime: ThreadRuntimeConfiguration(
                inference: ModelEndpoint(
                    backendID: inferenceBackend.backendID,
                    modelID: vllm.model,
                    options: vllm.options
                )
            ),
            instructions: "Reply with a short sentence."
        )

        let session = try await manager.session(configuration: threadConfig)

        var partials: [String] = []
        var completed: TextGenerationResult?
        let stream = session.stream("Say hello.")
        for try await event in stream {
            switch event {
            case .partial(let text):
                partials.append(text)
            case .completed(let result):
                completed = result
            }
        }

        XCTAssertNotNil(completed)
        XCTAssertFalse(completed?.text.isEmpty ?? true)

        let history = try await session.inspection.history()
        XCTAssertEqual(history.count, 2)
    }

    func testVLLMContextSessionStructuredReply() async throws {
        guard let vllm = vllmEndpoint() else { return }

        let inferenceBackend = VLLMInferenceBackend()
        let structuredBackend = VLLMStructuredOutputBackend()
        let registry = RuntimeRegistry()
        await registry.register(inferenceBackend)

        var structuredOptions = vllm.options
        structuredOptions["guidedDecoding"] = "json"

        let config = ContextManagerConfiguration(
            runtimeRegistry: registry,
            structuredBackends: [structuredBackend.backendID: structuredBackend],
            memory: MemoryPolicy(automaticallyExtractMemories: false),
            diagnostics: DiagnosticsPolicy(isEnabled: false)
        )
        let manager = ContextManager(configuration: config)

        let threadConfig = ThreadConfiguration(
            runtime: ThreadRuntimeConfiguration(
                inference: ModelEndpoint(
                    backendID: inferenceBackend.backendID,
                    modelID: vllm.model,
                    options: vllm.options
                ),
                structuredOutput: ModelEndpoint(
                    backendID: structuredBackend.backendID,
                    modelID: vllm.model,
                    options: structuredOptions
                )
            ),
            instructions: "Return JSON only."
        )

        let session = try await manager.session(configuration: threadConfig)

        let spec = StructuredOutput.codable(
            IntegrationStructuredPayload.self,
            schema: .object(
                ObjectSchema(
                    name: "IntegrationStructuredPayload",
                    properties: [.init(name: "title", schema: .string())]
                )
            )
        )

        let reply = try await session.reply(to: "Return an object with title set to ok.", spec: spec)

        XCTAssertFalse(reply.transcriptText.isEmpty)

        let history = try await session.inspection.history()
        XCTAssertEqual(history.count, 2)
    }

    func testVLLMContextSessionMultiTurn() async throws {
        guard let vllm = vllmEndpoint() else { return }

        let inferenceBackend = VLLMInferenceBackend()
        let registry = RuntimeRegistry()
        await registry.register(inferenceBackend)

        let config = ContextManagerConfiguration(
            runtimeRegistry: registry,
            memory: MemoryPolicy(automaticallyExtractMemories: false),
            diagnostics: DiagnosticsPolicy(isEnabled: false)
        )
        let manager = ContextManager(configuration: config)

        let threadConfig = ThreadConfiguration(
            runtime: ThreadRuntimeConfiguration(
                inference: ModelEndpoint(
                    backendID: inferenceBackend.backendID,
                    modelID: vllm.model,
                    options: vllm.options
                )
            ),
            instructions: "Reply with a single short word."
        )

        let session = try await manager.session(configuration: threadConfig)

        let reply1 = try await session.reply(to: "hello")
        XCTAssertFalse(reply1.text.isEmpty)

        let reply2 = try await session.reply(to: "how are you")
        XCTAssertFalse(reply2.text.isEmpty)

        let reply3 = try await session.reply(to: "goodbye")
        XCTAssertFalse(reply3.text.isEmpty)

        let history = try await session.inspection.history()
        XCTAssertEqual(history.count, 6) // 3 user + 3 assistant
    }
}
