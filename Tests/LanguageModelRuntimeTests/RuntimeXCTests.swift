import Foundation
import XCTest
@testable import LanguageModelRuntime

final class RuntimeXCTests: XCTestCase {
    func testRegistryResolvesBackend() async throws {
        let registry = RuntimeRegistry()
        let backend = FakeInferenceBackend()

        await registry.register(backend)
        let resolved = try await registry.backend(for: backend.backendID)
        let availability = await resolved.availability(
            for: ModelEndpoint(backendID: backend.backendID, modelID: "demo")
        )

        XCTAssertEqual(availability.status, .available)
        XCTAssertTrue(availability.capabilities.supportsTextStreaming)
    }

    func testRuntimeSessionInterfaces() async throws {
        let backend = FakeInferenceBackend()
        let session = try await backend.makeSession(
            endpoint: ModelEndpoint(backendID: backend.backendID, modelID: "demo"),
            instructions: "Be concise.",
            locale: nil
        )

        let generated = try await session.generateText(
            prompt: "Hello",
            options: TextGenerationOptions(maximumResponseTokens: 32)
        )

        var fragments: [String] = []
        var completed: TextGenerationResult?
        let stream = await session.streamText(
            prompt: "Hello",
            options: TextGenerationOptions(maximumResponseTokens: 32)
        )
        for try await event in stream {
            switch event {
            case .partial(let text):
                fragments.append(text)
            case .completed(let result):
                completed = result
            }
        }

        XCTAssertEqual(generated.text, "fake-response")
        XCTAssertEqual(fragments, ["fake-", "response"])
        XCTAssertEqual(completed, TextGenerationResult(text: "fake-response"))
    }
}

private struct FakeInferenceBackend: InferenceBackend {
    let backendID = "fake-runtime"

    func availability(for endpoint: ModelEndpoint) async -> RuntimeAvailability {
        _ = endpoint
        return RuntimeAvailability(
            status: .available,
            capabilities: RuntimeCapabilities(
                supportsTextGeneration: true,
                supportsTextStreaming: true,
                supportsStructuredOutput: false,
                supportsExactTokenEstimation: false,
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
        _ = instructions
        _ = locale
        return FakeInferenceSession()
    }

    func contextWindowTokens(for endpoint: ModelEndpoint) async -> Int? {
        endpoint.contextWindowOverride ?? 4096
    }

    func exactTokenEstimator(for endpoint: ModelEndpoint) async -> (any TokenEstimating)? {
        _ = endpoint
        return nil
    }
}

private actor FakeInferenceSession: InferenceSession {
    func generateText(
        prompt: String,
        options: TextGenerationOptions
    ) async throws -> TextGenerationResult {
        _ = prompt
        _ = options
        return TextGenerationResult(text: "fake-response")
    }

    func streamText(
        prompt: String,
        options: TextGenerationOptions
    ) async -> AsyncThrowingStream<TextStreamEvent, Error> {
        _ = prompt
        _ = options
        return AsyncThrowingStream { continuation in
            continuation.yield(.partial("fake-"))
            continuation.yield(.partial("response"))
            continuation.yield(.completed(.init(text: "fake-response")))
            continuation.finish()
        }
    }
}
