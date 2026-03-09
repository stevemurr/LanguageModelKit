import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif
import LanguageModelRuntime
import LanguageModelStructuredCore
import LanguageModelStructuredOutput

public final class OpenAIInferenceBackend: InferenceBackend {
    public let backendID: String
    private let client: HTTPClient

    public init(
        backendID: String = "openai",
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
        guard endpoint.options["apiKey"].map({ $0.isEmpty == false }) == true else {
            return RuntimeAvailability(
                status: .unavailable(reason: "Missing required option apiKey"),
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
        try validate(endpoint: endpoint)
        return OpenAIInferenceSession(
            endpoint: endpoint,
            instructions: instructions,
            locale: locale,
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

    private func validate(endpoint: ModelEndpoint) throws {
        guard endpoint.options["baseURL"].flatMap(URL.init(string:)) != nil else {
            throw RuntimeError.unavailable("Missing required option baseURL")
        }
        guard endpoint.options["apiKey"].map({ $0.isEmpty == false }) == true else {
            throw RuntimeError.unavailable("Missing required option apiKey")
        }
    }
}

public final class OpenAIStructuredOutputBackend: StructuredOutputBackend {
    public let backendID: String
    private let client: HTTPClient

    public init(
        backendID: String = "openai",
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
        let transport = try OpenAITransport(endpoint: endpoint, client: client)
        let request = try transport.makeStructuredRequest(
            modelID: endpoint.modelID,
            instructions: instructions,
            prompt: prompt,
            schema: spec.schema,
            options: options
        )

        let response = try await transport.performCompletion(request)
        let payload = try response.requireMessageContent()
        let data = Data(payload.utf8)

        do {
            let value = try spec.decode(data)
            return StructuredGenerationResult(
                value: value,
                transcriptText: spec.transcriptRenderer(value)
            )
        } catch {
            throw RuntimeError.generationFailed(error.localizedDescription)
        }
    }
}

actor OpenAIInferenceSession: LiveStructuredGenerationSession {
    private let endpoint: ModelEndpoint
    private let instructions: String?
    private let locale: Locale?
    private let client: HTTPClient

    init(
        endpoint: ModelEndpoint,
        instructions: String?,
        locale: Locale?,
        client: HTTPClient
    ) {
        self.endpoint = endpoint
        self.instructions = instructions
        self.locale = locale
        self.client = client
    }

    func generateText(
        prompt: String,
        options: TextGenerationOptions
    ) async throws -> TextGenerationResult {
        _ = locale
        let transport = try OpenAITransport(endpoint: endpoint, client: client)
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

    func generateStructured<Value: Sendable>(
        prompt: String,
        spec: StructuredOutputSpec<Value>,
        options: StructuredGenerationOptions
    ) async throws -> StructuredGenerationResult<Value> {
        _ = locale
        let transport = try OpenAITransport(endpoint: endpoint, client: client)
        let request = try transport.makeStructuredRequest(
            modelID: endpoint.modelID,
            instructions: instructions,
            prompt: prompt,
            schema: spec.schema,
            options: options
        )
        let response = try await transport.performCompletion(request)
        let payload = try response.requireMessageContent()
        let data = Data(payload.utf8)

        do {
            let value = try spec.decode(data)
            return StructuredGenerationResult(
                value: value,
                transcriptText: spec.transcriptRenderer(value)
            )
        } catch {
            throw RuntimeError.generationFailed(error.localizedDescription)
        }
    }

    func streamText(
        prompt: String,
        options: TextGenerationOptions
    ) async -> AsyncThrowingStream<TextStreamEvent, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let transport = try OpenAITransport(endpoint: endpoint, client: client)
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

private struct OpenAITransport {
    private let endpoint: ModelEndpoint
    private let client: HTTPClient

    init(endpoint: ModelEndpoint, client: HTTPClient) throws {
        guard endpoint.options["baseURL"].flatMap(URL.init(string:)) != nil else {
            throw RuntimeError.unavailable("Missing required option baseURL")
        }
        guard endpoint.options["apiKey"].map({ $0.isEmpty == false }) == true else {
            throw RuntimeError.unavailable("Missing required option apiKey")
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
        if let apiKey = endpoint.options["apiKey"] {
            request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        }
        if let organization = endpoint.options["organization"], organization.isEmpty == false {
            request.setValue(organization, forHTTPHeaderField: "OpenAI-Organization")
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

    func makeStructuredRequest(
        modelID: String,
        instructions: String?,
        prompt: String,
        schema: OutputSchema,
        options: StructuredGenerationOptions
    ) throws -> URLRequest {
        let schemaData = try PortableSchemaRenderer.renderData(schema: schema, name: "response")
        let schemaObject = try JSONSerialization.jsonObject(with: schemaData)
        return try makeRequest(
            modelID: modelID,
            instructions: instructions,
            prompt: prompt,
            maximumResponseTokens: options.maximumResponseTokens,
            temperature: options.deterministic ? 0 : nil,
            stream: false,
            extraBody: [
                "response_format": [
                    "type": "json_schema",
                    "json_schema": [
                        "name": "response",
                        "strict": RuntimeOptionParser.bool(
                            from: endpoint.options,
                            key: "strictStructuredOutputs",
                            default: true
                        ),
                        "schema": schemaObject
                    ]
                ]
            ]
        )
    }

    func performCompletion(_ request: URLRequest) async throws -> OpenAICompletionResponse {
        let response = try await client.data(for: request)
        if response.response.statusCode >= 400 {
            throw mapFailure(data: response.data, statusCode: response.response.statusCode)
        }
        do {
            return try JSONDecoder().decode(OpenAICompletionResponse.self, from: response.data)
        } catch {
            throw RuntimeError.transportFailed(error.localizedDescription)
        }
    }

    func streamCompletion(_ request: URLRequest) -> AsyncThrowingStream<OpenAIStreamResponse, Error> {
        AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    var (bytes, response) = try await client.bytes(for: request)
                    if response.statusCode >= 400 {
                        let data = try await collectData(from: &bytes)
                        throw mapFailure(data: data, statusCode: response.statusCode)
                    }
                    for try await payload in bytes.sseEvents {
                        let data = Data(payload.utf8)
                        if let error = try? JSONDecoder().decode(OpenAIErrorEnvelope.self, from: data) {
                            throw mapFailure(errorEnvelope: error, statusCode: nil)
                        }
                        let chunk = try JSONDecoder().decode(OpenAIStreamResponse.self, from: data)
                        if let runtimeError = chunk.runtimeError {
                            throw runtimeError
                        }
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

    private func mapFailure(data: Data, statusCode: Int) -> RuntimeError {
        if let envelope = try? JSONDecoder().decode(OpenAIErrorEnvelope.self, from: data) {
            return mapFailure(errorEnvelope: envelope, statusCode: statusCode)
        }
        let payload = String(decoding: data, as: UTF8.self)
        return mapFailure(message: payload, code: nil, type: nil, statusCode: statusCode)
    }

    private func mapFailure(
        errorEnvelope: OpenAIErrorEnvelope,
        statusCode: Int?
    ) -> RuntimeError {
        mapFailure(
            message: errorEnvelope.error.message ?? "OpenAI-compatible request failed",
            code: errorEnvelope.error.code,
            type: errorEnvelope.error.type,
            statusCode: statusCode
        )
    }

    private func mapFailure(
        message: String,
        code: String?,
        type: String?,
        statusCode: Int?
    ) -> RuntimeError {
        let lowered = [message, code, type]
            .compactMap { $0 }
            .joined(separator: " ")
            .lowercased()

        if lowered.contains("response_format")
            || lowered.contains("json_schema")
            || lowered.contains("strict")
        {
            return .unsupportedCapability(
                message.isEmpty ? "The configured server does not support strict structured outputs" : message
            )
        }

        if lowered.contains("context_length")
            || lowered.contains("maximum context length")
            || lowered.contains("context window")
            || lowered.contains("too many tokens")
        {
            return .contextOverflow(message)
        }

        if lowered.contains("content_filter")
            || lowered.contains("refusal")
            || lowered.contains("safety")
            || lowered.contains("policy")
        {
            return .refusal(message)
        }

        if let statusCode {
            return .transportFailed(message.isEmpty ? "HTTP \(statusCode)" : message)
        }
        return .transportFailed(message)
    }

    private func collectData(
        from bytes: inout URLSession.AsyncBytes
    ) async throws -> Data {
        var data = Data()
        for try await byte in bytes {
            data.append(byte)
        }
        return data
    }
}

private struct OpenAIErrorEnvelope: Decodable {
    struct ErrorPayload: Decodable {
        let message: String?
        let type: String?
        let code: String?
    }

    let error: ErrorPayload
}

private struct OpenAICompletionResponse: Decodable {
    struct Choice: Decodable {
        struct Message: Decodable {
            let content: ContentPayload
            let refusal: String?
        }

        let message: Message
        let finishReason: String?

        private enum CodingKeys: String, CodingKey {
            case message
            case finishReason = "finish_reason"
        }
    }

    let choices: [Choice]

    func requireMessageContent() throws -> String {
        guard let choice = choices.first else {
            throw RuntimeError.generationFailed("Missing completion choice")
        }
        if let refusal = choice.message.refusal, refusal.isEmpty == false {
            throw RuntimeError.refusal(refusal)
        }
        if let refusal = choice.message.content.refusal, refusal.isEmpty == false {
            throw RuntimeError.refusal(refusal)
        }
        if choice.finishReason == "content_filter" {
            throw RuntimeError.refusal("The response was blocked by content filtering")
        }
        guard let content = choice.message.content.text else {
            throw RuntimeError.generationFailed("Missing completion content")
        }
        return content
    }
}

private struct OpenAIStreamResponse: Decodable {
    struct Choice: Decodable {
        struct Delta: Decodable {
            let content: String?
            let refusal: String?
        }

        let delta: Delta
        let finishReason: String?

        private enum CodingKeys: String, CodingKey {
            case delta
            case finishReason = "finish_reason"
        }
    }

    let choices: [Choice]

    var deltaText: String {
        choices.first?.delta.content ?? ""
    }

    var runtimeError: RuntimeError? {
        guard let choice = choices.first else {
            return nil
        }
        if let refusal = choice.delta.refusal, refusal.isEmpty == false {
            return .refusal(refusal)
        }
        if choice.finishReason == "content_filter" {
            return .refusal("The response was blocked by content filtering")
        }
        return nil
    }
}

private struct ContentPayload: Decodable {
    struct Part: Decodable {
        let type: String?
        let text: String?
    }

    let text: String?
    let refusal: String?

    init(from decoder: any Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let string = try? container.decode(String.self) {
            text = string
            refusal = nil
            return
        }
        if let parts = try? container.decode([Part].self) {
            let textual = parts
                .filter { $0.type != "refusal" }
                .compactMap(\.text)
                .joined()
            let refusalText = parts
                .filter { $0.type == "refusal" }
                .compactMap(\.text)
                .joined()
            text = textual.isEmpty ? nil : textual
            refusal = refusalText.isEmpty ? nil : refusalText
            return
        }
        text = nil
        refusal = nil
    }
}
