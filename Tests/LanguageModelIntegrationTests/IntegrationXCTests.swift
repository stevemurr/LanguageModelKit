import Foundation
import XCTest
import LanguageModelRuntime
import LanguageModelStructuredOutput
import LanguageModelOpenAI
import LanguageModelVLLM
#if canImport(FoundationModels)
import LanguageModelApple
#endif

final class IntegrationXCTests: XCTestCase {
    private struct IntegrationStructuredPayload: Codable, Sendable, Equatable {
        let title: String
    }

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

    func testVLLMLiveGate() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard
            let baseURL = environment["VLLM_BASE_URL"],
            let model = environment["VLLM_MODEL"]
        else {
            return
        }

        var options = ["baseURL": baseURL]
        if let apiKey = environment["VLLM_API_KEY"] {
            options["apiKey"] = apiKey
        }

        let backend = VLLMInferenceBackend()
        let session = try await backend.makeSession(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: model,
                options: options
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
        let environment = ProcessInfo.processInfo.environment
        guard
            let baseURL = environment["VLLM_BASE_URL"],
            let model = environment["VLLM_MODEL"]
        else {
            return
        }

        var options = ["baseURL": baseURL]
        if let apiKey = environment["VLLM_API_KEY"] {
            options["apiKey"] = apiKey
        }

        let backend = VLLMInferenceBackend()
        let session = try await backend.makeSession(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: model,
                options: options
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
        let environment = ProcessInfo.processInfo.environment
        guard
            let baseURL = environment["VLLM_BASE_URL"],
            let model = environment["VLLM_MODEL"]
        else {
            return
        }

        var options = [
            "baseURL": baseURL,
            "guidedDecoding": "json"
        ]
        if let apiKey = environment["VLLM_API_KEY"] {
            options["apiKey"] = apiKey
        }

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
                modelID: model,
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
}
