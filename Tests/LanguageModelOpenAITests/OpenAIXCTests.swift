import Foundation
import XCTest
@testable import LanguageModelOpenAI
@testable import LanguageModelStructuredOutput

final class OpenAIXCTests: XCTestCase {
    func testTextRequestFormation() async throws {
        let recorder = RequestRecorder()
        MockOpenAIURLProtocol.requestHandler = { request in
            await recorder.record(request)
            return (
                HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!,
                Data(#"{"choices":[{"message":{"content":"hello"}}]}"#.utf8)
            )
        }
        let session = makeSession()
        let backend = OpenAIInferenceBackend(session: session)
        let model = ModelEndpoint(
            backendID: backend.backendID,
            modelID: "gpt-demo",
            options: [
                "baseURL": "https://example.com",
                "apiKey": "secret",
                "organization": "org-1"
            ]
        )

        let runtimeSession = try await backend.makeSession(
            endpoint: model,
            instructions: "Be concise.",
            locale: nil
        )
        let response = try await runtimeSession.generateText(
            prompt: "Hi",
            options: TextGenerationOptions(maximumResponseTokens: 32, deterministic: true)
        )
        let request = await recorder.lastRequest()
        let body = try unwrapJSON(from: request)

        XCTAssertEqual(response.text, "hello")
        XCTAssertEqual(request?.value(forHTTPHeaderField: "Authorization"), "Bearer secret")
        XCTAssertEqual(body["model"] as? String, "gpt-demo")
        XCTAssertEqual(body["stream"] as? Bool, false)
        XCTAssertEqual((body["messages"] as? [[String: String]])?.count, 2)
    }

    func testStructuredRequestFormation() async throws {
        let recorder = RequestRecorder()
        MockOpenAIURLProtocol.requestHandler = { request in
            await recorder.record(request)
            return (
                HTTPURLResponse(url: request.url!, statusCode: 200, httpVersion: nil, headerFields: nil)!,
                Data(#"{"choices":[{"message":{"content":"{\"title\":\"done\"}"}}]}"#.utf8)
            )
        }
        let session = makeSession()
        let backend = OpenAIStructuredOutputBackend(session: session)
        let spec = StructuredOutput.codable(
            OpenAIStructuredPayload.self,
            schema: .object(
                ObjectSchema(
                    name: "OpenAIStructuredPayload",
                    properties: [.init(name: "title", schema: .string())]
                )
            )
        )

        let result = try await backend.generateStructured(
            endpoint: ModelEndpoint(
                backendID: backend.backendID,
                modelID: "gpt-demo",
                options: ["baseURL": "https://example.com", "apiKey": "secret"]
            ),
            instructions: "Return JSON.",
            locale: nil,
            prompt: "Generate",
            spec: spec,
            options: StructuredGenerationOptions(maximumResponseTokens: 64, deterministic: true)
        )
        let request = await recorder.lastRequest()
        let body = try unwrapJSON(from: request)
        let responseFormat = body["response_format"] as? [String: Any]
        let jsonSchema = responseFormat?["json_schema"] as? [String: Any]

        XCTAssertEqual(result.value, OpenAIStructuredPayload(title: "done"))
        XCTAssertEqual(responseFormat?["type"] as? String, "json_schema")
        XCTAssertEqual(jsonSchema?["strict"] as? Bool, true)
    }
}

private struct OpenAIStructuredPayload: Codable, Sendable, Equatable {
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
    configuration.protocolClasses = [MockOpenAIURLProtocol.self]
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

private final class MockOpenAIURLProtocol: URLProtocol, @unchecked Sendable {
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
