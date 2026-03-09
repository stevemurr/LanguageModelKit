import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import LanguageModelRuntime
import LanguageModelStructuredCore
import LanguageModelStructuredOutput

public final class VLLMInferenceBackend: InferenceBackend {
    public let backendID: String
    private let client: HTTPClient

    public init(
        backendID: String = "vllm",
        session: URLSession = .shared
    ) {
        self.backendID = backendID
        self.client = HTTPClient(session: session)
    }

    public func availability(for endpoint: ModelEndpoint) async -> RuntimeAvailability {
        guard endpoint.options["baseURL"].flatMap(URL.init(string:)) != nil else {
            return RuntimeAvailability(
                status: .unavailable(reason: "Missing required option baseURL"),
                capabilities: capabilities
            )
        }
        return RuntimeAvailability(status: .available, capabilities: capabilities)
    }

    public func makeSession(
        endpoint: ModelEndpoint,
        instructions: String?,
        locale: Locale?
    ) async throws -> any InferenceSession {
        _ = locale
        guard endpoint.options["baseURL"].flatMap(URL.init(string:)) != nil else {
            throw RuntimeError.unavailable("Missing required option baseURL")
        }
        return VLLMInferenceSession(
            endpoint: endpoint,
            instructions: instructions,
            client: client
        )
    }

    public func contextWindowTokens(for endpoint: ModelEndpoint) async -> Int? {
        endpoint.contextWindowOverride
    }

    public func exactTokenEstimator(for endpoint: ModelEndpoint) async -> (any TokenEstimating)? {
        _ = endpoint
        return nil
    }

    private var capabilities: RuntimeCapabilities {
        RuntimeCapabilities(
            supportsTextGeneration: true,
            supportsTextStreaming: true,
            supportsStructuredOutput: true,
            supportsExactTokenEstimation: false,
            supportsLocaleHints: false
        )
    }
}

public final class VLLMStructuredOutputBackend: StructuredOutputBackend {
    public let backendID: String
    private let client: HTTPClient

    public init(
        backendID: String = "vllm",
        session: URLSession = .shared
    ) {
        self.backendID = backendID
        self.client = HTTPClient(session: session)
    }

    public func generateStructured<Value>(
        endpoint: ModelEndpoint,
        instructions: String?,
        locale: Locale?,
        prompt: String,
        spec: StructuredOutputSpec<Value>,
        options: StructuredGenerationOptions
    ) async throws -> StructuredGenerationResult<Value> {
        _ = locale
        let transport = try VLLMTransport(endpoint: endpoint, client: client)
        let mode = endpoint.options["guidedDecoding"] ?? "auto"
        let extraBody = try guidedBody(
            mode: mode,
            schema: spec.schema
        )
        let request = try transport.makeRequest(
            modelID: endpoint.modelID,
            instructions: instructions,
            prompt: prompt,
            maximumResponseTokens: options.maximumResponseTokens,
            temperature: options.deterministic ? 0 : nil,
            stream: false,
            extraBody: extraBody
        )
        let response = try await transport.performCompletion(request)
        let payload = try response.requireMessageContent()
        let data = Data(payload.utf8)
        let value = try spec.decode(data)
        return StructuredGenerationResult(
            value: value,
            transcriptText: spec.transcriptRenderer(value)
        )
    }

    private func guidedBody(
        mode: String,
        schema: OutputSchema
    ) throws -> [String: Any] {
        switch mode {
        case "auto", "json":
            let schemaString = try JSONSchemaRenderer.renderString(schema: schema, name: "response")
            return ["guided_json": schemaString]
        case "grammar":
            throw RuntimeError.unsupportedCapability("Portable schemas cannot be lowered into vLLM grammar mode in v1")
        default:
            throw RuntimeError.unsupportedCapability("Unsupported guided decoding mode \(mode)")
        }
    }
}

actor VLLMInferenceSession: InferenceSession {
    private let endpoint: ModelEndpoint
    private let instructions: String?
    private let client: HTTPClient

    init(
        endpoint: ModelEndpoint,
        instructions: String?,
        client: HTTPClient
    ) {
        self.endpoint = endpoint
        self.instructions = instructions
        self.client = client
    }

    func generateText(
        prompt: String,
        options: TextGenerationOptions
    ) async throws -> TextGenerationResult {
        let transport = try VLLMTransport(endpoint: endpoint, client: client)
        let request = try transport.makeRequest(
            modelID: endpoint.modelID,
            instructions: instructions,
            prompt: prompt,
            maximumResponseTokens: options.maximumResponseTokens,
            temperature: options.deterministic ? 0 : options.temperature,
            stream: false
        )
        let response = try await transport.performCompletion(request)
        return TextGenerationResult(text: try response.requireMessageContent())
    }

    func streamText(
        prompt: String,
        options: TextGenerationOptions
    ) async -> AsyncThrowingStream<TextStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let transport = try VLLMTransport(endpoint: endpoint, client: client)
                    let request = try transport.makeRequest(
                        modelID: endpoint.modelID,
                        instructions: instructions,
                        prompt: prompt,
                        maximumResponseTokens: options.maximumResponseTokens,
                        temperature: options.deterministic ? 0 : options.temperature,
                        stream: true
                    )
                    var aggregate = ""
                    for try await chunk in transport.streamCompletion(request) {
                        let fragment = chunk.deltaText
                        if fragment.isEmpty {
                            continue
                        }
                        aggregate += fragment
                        continuation.yield(.partial(fragment))
                    }
                    continuation.yield(.completed(.init(text: aggregate)))
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
}

private struct VLLMTransport {
    private let endpoint: ModelEndpoint
    private let client: HTTPClient

    init(endpoint: ModelEndpoint, client: HTTPClient) throws {
        guard endpoint.options["baseURL"].flatMap(URL.init(string:)) != nil else {
            throw RuntimeError.unavailable("Missing required option baseURL")
        }
        self.endpoint = endpoint
        self.client = client
    }

    func makeRequest(
        modelID: String,
        instructions: String?,
        prompt: String,
        maximumResponseTokens: Int?,
        temperature: Double?,
        stream: Bool,
        extraBody: [String: Any] = [:]
    ) throws -> URLRequest {
        let url = try completionsURL(baseURLString: endpoint.options["baseURL"] ?? "")
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let apiKey = endpoint.options["apiKey"], apiKey.isEmpty == false {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }

        var messages: [[String: Any]] = []
        if let instructions, instructions.isEmpty == false {
            messages.append(["role": "system", "content": instructions])
        }
        messages.append(["role": "user", "content": prompt])

        var body: [String: Any] = [
            "model": modelID,
            "messages": messages,
            "stream": stream
        ]
        if let maximumResponseTokens {
            body["max_tokens"] = maximumResponseTokens
        }
        if let temperature {
            body["temperature"] = temperature
        }
        for (key, value) in extraBody {
            body[key] = value
        }
        request.httpBody = try JSONSerialization.data(withJSONObject: body, options: [.sortedKeys])
        return request
    }

    func performCompletion(_ request: URLRequest) async throws -> VLLMCompletionResponse {
        let response = try await client.data(for: request)
        if response.response.statusCode >= 400 {
            let text = String(decoding: response.data, as: UTF8.self).lowercased()
            if text.contains("guided") || text.contains("json") || text.contains("grammar") {
                throw RuntimeError.unsupportedCapability("The configured vLLM server rejected the guided decoding payload")
            }
            throw RuntimeError.transportFailed("HTTP \(response.response.statusCode)")
        }
        do {
            return try JSONDecoder().decode(VLLMCompletionResponse.self, from: response.data)
        } catch {
            throw RuntimeError.transportFailed(error.localizedDescription)
        }
    }

    func streamCompletion(_ request: URLRequest) -> AsyncThrowingStream<VLLMStreamResponse, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let (bytes, response) = try await client.bytes(for: request)
                    if response.statusCode >= 400 {
                        throw RuntimeError.transportFailed("HTTP \(response.statusCode)")
                    }
                    for try await payload in bytes.sseEvents {
                        let data = Data(payload.utf8)
                        let chunk = try JSONDecoder().decode(VLLMStreamResponse.self, from: data)
                        continuation.yield(chunk)
                    }
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

    private func completionsURL(baseURLString: String) throws -> URL {
        guard let baseURL = URL(string: baseURLString) else {
            throw RuntimeError.unavailable("Invalid baseURL")
        }
        if baseURL.path.hasSuffix("/v1") {
            return baseURL.appendingPathComponent("chat/completions")
        }
        return baseURL.appendingPathComponent("v1/chat/completions")
    }
}

private struct VLLMCompletionResponse: Decodable {
    struct Choice: Decodable {
        struct Message: Decodable {
            let content: VLLMContentPayload
        }

        let message: Message
    }

    let choices: [Choice]

    func requireMessageContent() throws -> String {
        guard let content = choices.first?.message.content.text else {
            throw RuntimeError.generationFailed("Missing completion content")
        }
        return content
    }
}

private struct VLLMStreamResponse: Decodable {
    struct Choice: Decodable {
        struct Delta: Decodable {
            let content: String?
        }

        let delta: Delta
    }

    let choices: [Choice]

    var deltaText: String {
        choices.first?.delta.content ?? ""
    }
}

private struct VLLMContentPayload: Decodable {
    struct Part: Decodable {
        let type: String?
        let text: String?
    }

    let text: String?

    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let string = try? container.decode(String.self) {
            text = string
            return
        }
        if let parts = try? container.decode([Part].self) {
            text = parts.compactMap(\.text).joined()
            return
        }
        text = nil
    }
}
