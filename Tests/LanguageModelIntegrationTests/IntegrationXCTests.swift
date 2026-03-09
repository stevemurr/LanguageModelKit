import Foundation
import XCTest
import LanguageModelRuntime
import LanguageModelOpenAI
import LanguageModelVLLM
#if canImport(FoundationModels)
import LanguageModelApple
#endif

final class IntegrationXCTests: XCTestCase {
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
}
