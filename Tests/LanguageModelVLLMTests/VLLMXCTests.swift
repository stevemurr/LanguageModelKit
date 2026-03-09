import Foundation
import XCTest
@testable import LanguageModelStructuredOutput
@testable import LanguageModelVLLM

final class VLLMXCTests: XCTestCase {
    func testTextRequestFormation() async throws {
        let recorder = RequestRecorder()
        MockVLLMURLProtocol.requestHandler = { request in
            await recorder.record(request)
            return (
                HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!,
                Data(#"{"choices":[{"message":{"content":"hello"}}]}"#.utf8)
            )
        }
        let session = makeSession()
        let backend = VLLMInferenceBackend(session: session)
        let runtimeSession = try await backend.makeSession(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: "demo",
                options: ["baseURL": "https://example.com", "apiKey": "optional-key"]
            ),
            instructions: "Keep it short.",
            locale: nil
        )
        let result = try await runtimeSession.generateText(
            prompt: "Hello",
            options: TextGenerationOptions(maximumResponseTokens: 16)
        )
        let request = await recorder.lastRequest()

        XCTAssertEqual(result.text, "hello")
        XCTAssertEqual(request?.url?.absoluteString, "https://example.com/v1/chat/completions")
        XCTAssertEqual(request?.value(forHTTPHeaderField: "Authorization"), "Bearer optional-key")
    }

    func testStructuredGuidedJSON() async throws {
        let recorder = RequestRecorder()
        MockVLLMURLProtocol.requestHandler = { request in
            await recorder.record(request)
            return (
                HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!,
                Data(#"{"choices":[{"message":{"content":"{\"title\":\"done\"}"}}]}"#.utf8)
            )
        }
        let session = makeSession()
        let backend = VLLMStructuredOutputBackend(session: session)
        let spec = StructuredOutput.codable(
            VLLMStructuredPayload.self,
            schema: .object(
                ObjectSchema(
                    name: "VLLMStructuredPayload",
                    properties: [.init(name: "title", schema: .string())]
                )
            )
        )

        let result = try await backend.generateStructured(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: "demo",
                options: ["baseURL": "https://example.com", "guidedDecoding": "auto"]
            ),
            instructions: "Return JSON.",
            locale: nil,
            prompt: "Generate",
            spec: spec,
            options: StructuredGenerationOptions(maximumResponseTokens: 32, deterministic: true)
        )
        let request = await recorder.lastRequest()
        let body = try unwrapJSON(from: request)

        XCTAssertEqual(result.value, VLLMStructuredPayload(title: "done"))
        XCTAssertTrue((body["guided_json"] as? String)?.contains(#""title""#) == true)
    }

    func testStructuredGuidedGrammar() async throws {
        let recorder = RequestRecorder()
        MockVLLMURLProtocol.requestHandler = { request in
            await recorder.record(request)
            return (
                HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!,
                Data(#"{"choices":[{"message":{"content":"{\"title\":\"done\"}"}}]}"#.utf8)
            )
        }
        let backend = VLLMStructuredOutputBackend(session: makeSession())
        let spec = StructuredOutput.codable(
            VLLMStructuredPayload.self,
            schema: .object(
                ObjectSchema(
                    name: "VLLMStructuredPayload",
                    properties: [.init(name: "title", schema: .string())]
                )
            )
        )

        _ = try await backend.generateStructured(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: "demo",
                options: ["baseURL": "https://example.com", "guidedDecoding": "grammar"]
            ),
            instructions: "Return JSON.",
            locale: nil,
            prompt: "Generate",
            spec: spec,
            options: StructuredGenerationOptions(maximumResponseTokens: 32, deterministic: true)
        )
        let request = await recorder.lastRequest()
        let body = try unwrapJSON(from: request)

        XCTAssertTrue((body["guided_grammar"] as? String)?.contains("root ::=") == true)
    }

    func testUnsupportedGrammarShapeFailsDeterministically() async throws {
        let backend = VLLMStructuredOutputBackend(session: makeSession())
        let spec = StructuredOutput.codable(
            VLLMStructuredPayload.self,
            schema: .object(
                ObjectSchema(
                    name: "VLLMStructuredPayload",
                    properties: [.init(name: "title", schema: .string(.init(regex: "^[a-z]+$")))]
                )
            )
        )

        do {
            _ = try await backend.generateStructured(
                endpoint: ModelEndpoint(
                    backendID: backend.backendID,
                    modelID: "demo",
                    options: ["baseURL": "https://example.com", "guidedDecoding": "grammar"]
                ),
                instructions: nil,
                locale: nil,
                prompt: "Generate",
                spec: spec,
                options: StructuredGenerationOptions()
            )
            XCTFail("Expected unsupported capability")
        } catch let error as RuntimeError {
            XCTAssertEqual(
                error,
                .unsupportedCapability("vLLM grammar mode does not support constrained strings")
            )
        }
    }

    func testGuidedDecodingRejectionMapsUnsupportedCapability() async throws {
        MockVLLMURLProtocol.requestHandler = { request in
            (
                HTTPURLResponse(url: request.url!, statusCode: 400, httpVersion: nil, headerFields: nil)!,
                Data(#"{"error":{"message":"guided_grammar payload rejected","type":"invalid_request_error","code":"guided_grammar_error"}}"#.utf8)
            )
        }
        let backend = VLLMStructuredOutputBackend(session: makeSession())
        let spec = StructuredOutput.codable(
            VLLMStructuredPayload.self,
            schema: .object(
                ObjectSchema(
                    name: "VLLMStructuredPayload",
                    properties: [.init(name: "title", schema: .string())]
                )
            )
        )

        do {
            _ = try await backend.generateStructured(
                endpoint: ModelEndpoint(
                    backendID: backend.backendID,
                    modelID: "demo",
                    options: ["baseURL": "https://example.com", "guidedDecoding": "grammar"]
                ),
                instructions: nil,
                locale: nil,
                prompt: "Generate",
                spec: spec,
                options: StructuredGenerationOptions()
            )
            XCTFail("Expected unsupported capability")
        } catch let error as RuntimeError {
            XCTAssertEqual(error, .unsupportedCapability("guided_grammar payload rejected"))
        }
    }

    func testStreamingYieldsPartials() async throws {
        MockVLLMURLProtocol.requestHandler = { request in
            let payload = "data: {\"choices\":[{\"delta\":{\"content\":\"hel\"}}]}\n\n"
                + "data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n"
                + "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"stop\"}]}\n\n"
                + "data: [DONE]\n\n"
            return (
                HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!,
                Data(payload.utf8)
            )
        }
        let backend = VLLMInferenceBackend(session: makeSession())
        let session = try await backend.makeSession(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: "demo",
                options: ["baseURL": "https://example.com"]
            ),
            instructions: nil,
            locale: nil
        )

        var partials: [String] = []
        var completed: TextGenerationResult?
        let stream = await session.streamText(prompt: "Hello", options: TextGenerationOptions())
        for try await event in stream {
            switch event {
            case .partial(let text):
                partials.append(text)
            case .completed(let result):
                completed = result
            }
        }

        XCTAssertEqual(partials, ["hel", "lo"])
        XCTAssertEqual(completed, TextGenerationResult(text: "hello"))
    }

    func testStreamingMapsRefusal() async throws {
        MockVLLMURLProtocol.requestHandler = { request in
            let payload = "data: {\"choices\":[{\"delta\":{\"refusal\":\"Blocked\"},\"finish_reason\":\"content_filter\"}]}\n\n"
                + "data: [DONE]\n\n"
            return (
                HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!,
                Data(payload.utf8)
            )
        }
        let backend = VLLMInferenceBackend(session: makeSession())
        let session = try await backend.makeSession(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: "demo",
                options: ["baseURL": "https://example.com"]
            ),
            instructions: nil,
            locale: nil
        )

        let stream = await session.streamText(prompt: "Hello", options: TextGenerationOptions())

        do {
            for try await _ in stream {}
            XCTFail("Expected refusal")
        } catch let error as RuntimeError {
            XCTAssertEqual(error, .refusal("Blocked"))
        }
    }
}

private struct VLLMStructuredPayload: Codable, Sendable, Equatable {
    let title: String
}

private actor RequestRecorder {
    private var request: URLRequest?

    func record(_ request: URLRequest) {
        self.request = request
    }

    func lastRequest() -> URLRequest? {
        request
    }
}

private func makeSession() -> URLSession {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [MockVLLMURLProtocol.self]
    return URLSession(configuration: configuration)
}

private func unwrapJSON(from request: URLRequest?) throws -> [String: Any] {
    let data = try bodyData(from: request)
    guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        throw RuntimeError.transportFailed("Request body was not a JSON object")
    }
    return object
}

private func bodyData(from request: URLRequest?) throws -> Data {
    guard let request else {
        throw RuntimeError.transportFailed("Missing request body")
    }
    if let data = request.httpBody {
        return data
    }
    guard let stream = request.httpBodyStream else {
        throw RuntimeError.transportFailed("Missing request body")
    }

    stream.open()
    defer { stream.close() }

    var result = Data()
    let bufferSize = 1024
    let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
    defer { buffer.deallocate() }

    while stream.hasBytesAvailable {
        let count = stream.read(buffer, maxLength: bufferSize)
        if count < 0, let error = stream.streamError {
            throw error
        }
        if count == 0 {
            break
        }
        result.append(buffer, count: count)
    }
    return result
}

private final class MockVLLMURLProtocol: URLProtocol, @unchecked Sendable {
    nonisolated(unsafe) static var requestHandler: ((URLRequest) async throws -> (HTTPURLResponse, Data))?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        Task {
            do {
                guard let handler = Self.requestHandler else {
                    throw RuntimeError.transportFailed("Missing URLProtocol handler")
                }
                let (response, data) = try await handler(request)
                client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
                client?.urlProtocol(self, didLoad: data)
                client?.urlProtocolDidFinishLoading(self)
            } catch {
                client?.urlProtocol(self, didFailWithError: error)
            }
        }
    }

    override func stopLoading() {}
}
