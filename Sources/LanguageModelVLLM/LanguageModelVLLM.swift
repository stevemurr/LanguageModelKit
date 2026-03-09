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
        let value = try spec.decode(data)
        return StructuredGenerationResult(
            value: value,
            transcriptText: spec.transcriptRenderer(value)
        )
    }
}

actor VLLMInferenceSession: LiveStructuredGenerationSession {
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

    func generateStructured<Value: Sendable>(
        prompt: String,
        spec: StructuredOutputSpec<Value>,
        options: StructuredGenerationOptions
    ) async throws -> StructuredGenerationResult<Value> {
        let transport = try VLLMTransport(endpoint: endpoint, client: client)
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
        let value = try spec.decode(data)
        return StructuredGenerationResult(
            value: value,
            transcriptText: spec.transcriptRenderer(value)
        )
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

    func makeStructuredRequest(
        modelID: String,
        instructions: String?,
        prompt: String,
        schema: OutputSchema,
        options: StructuredGenerationOptions
    ) throws -> URLRequest {
        let extraBody = try guidedBody(
            mode: endpoint.options["guidedDecoding"] ?? "auto",
            schema: schema
        )
        return try makeRequest(
            modelID: modelID,
            instructions: instructions,
            prompt: prompt,
            maximumResponseTokens: options.maximumResponseTokens,
            temperature: options.deterministic ? 0 : nil,
            stream: false,
            extraBody: extraBody
        )
    }

    func performCompletion(_ request: URLRequest) async throws -> VLLMCompletionResponse {
        let response = try await client.data(for: request)
        if response.response.statusCode >= 400 {
            throw mapFailure(data: response.data, statusCode: response.response.statusCode)
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
                    var (bytes, response) = try await client.bytes(for: request)
                    if response.statusCode >= 400 {
                        let data = try await collectData(from: &bytes)
                        throw mapFailure(data: data, statusCode: response.statusCode)
                    }
                    for try await payload in bytes.sseEvents {
                        let data = Data(payload.utf8)
                        if let error = try? JSONDecoder().decode(VLLMErrorEnvelope.self, from: data) {
                            throw mapFailure(errorEnvelope: error, statusCode: nil)
                        }
                        let chunk = try JSONDecoder().decode(VLLMStreamResponse.self, from: data)
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

    private func guidedBody(
        mode: String,
        schema: OutputSchema
    ) throws -> [String: Any] {
        switch mode {
        case "auto", "json":
            let schemaString = try PortableSchemaRenderer.renderString(schema: schema, name: "response")
            return ["guided_json": schemaString]
        case "grammar":
            let grammar = try VLLMGrammarRenderer.render(schema: schema)
            return ["guided_grammar": grammar]
        default:
            throw RuntimeError.unsupportedCapability("Unsupported guided decoding mode \(mode)")
        }
    }

    private func mapFailure(data: Data, statusCode: Int) -> RuntimeError {
        if let envelope = try? JSONDecoder().decode(VLLMErrorEnvelope.self, from: data) {
            return mapFailure(errorEnvelope: envelope, statusCode: statusCode)
        }
        let message = String(decoding: data, as: UTF8.self)
        return mapFailure(message: message, code: nil, type: nil, statusCode: statusCode)
    }

    private func mapFailure(
        errorEnvelope: VLLMErrorEnvelope,
        statusCode: Int?
    ) -> RuntimeError {
        mapFailure(
            message: errorEnvelope.error.message ?? "vLLM request failed",
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

        if lowered.contains("guided")
            || lowered.contains("guided_json")
            || lowered.contains("guided_grammar")
            || lowered.contains("grammar")
        {
            return .unsupportedCapability(
                message.isEmpty ? "The configured vLLM server rejected the guided decoding payload" : message
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

private struct VLLMErrorEnvelope: Decodable {
    struct ErrorPayload: Decodable {
        let message: String?
        let type: String?
        let code: String?
    }

    let error: ErrorPayload
}

private struct VLLMCompletionResponse: Decodable {
    struct Choice: Decodable {
        struct Message: Decodable {
            let content: VLLMContentPayload
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

private struct VLLMStreamResponse: Decodable {
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

private struct VLLMContentPayload: Decodable {
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

private enum VLLMGrammarRenderer {
    static func render(schema: OutputSchema) throws -> String {
        var builder = VLLMGrammarBuilder()
        return try builder.render(schema: schema)
    }
}

private struct VLLMGrammarBuilder {
    private var rules: [(String, String)] = []
    private var seenNames: Set<String> = []

    mutating func render(schema: OutputSchema) throws -> String {
        let rootRule = try ruleReference(for: schema, preferredName: "value")
        addRule(name: "root", expression: "ws \(rootRule) ws")
        addBaseRules()
        return rules
            .map { "\($0.0) ::= \($0.1)" }
            .joined(separator: "\n")
    }

    private mutating func ruleReference(
        for schema: OutputSchema,
        preferredName: String
    ) throws -> String {
        switch schema {
        case .string(let constraints):
            if constraints.regex != nil || constraints.minLength != nil || constraints.maxLength != nil {
                throw RuntimeError.unsupportedCapability("vLLM grammar mode does not support constrained strings")
            }
            return "json_string"
        case .integer(let constraints):
            if constraints.minimum != nil || constraints.maximum != nil {
                throw RuntimeError.unsupportedCapability("vLLM grammar mode does not support constrained integers")
            }
            return "json_integer"
        case .number(let constraints):
            if constraints.minimum != nil || constraints.maximum != nil {
                throw RuntimeError.unsupportedCapability("vLLM grammar mode does not support constrained numbers")
            }
            return "json_number"
        case .boolean:
            return "json_boolean"
        case .enumeration(let schema):
            let name = uniqueName(preferredName.isEmpty ? "enum" : preferredName)
            let literals = schema.cases.map(Self.quotedJSONStringLiteral).joined(separator: " | ")
            addRule(name: name, expression: literals)
            return name
        case .optional(let wrapped):
            let wrappedRef = try ruleReference(for: wrapped, preferredName: "\(preferredName)_value")
            let name = uniqueName(preferredName.isEmpty ? "optional" : preferredName)
            addRule(name: name, expression: "\(wrappedRef) | \"null\"")
            return name
        case .array(let constraints):
            let name = uniqueName(preferredName.isEmpty ? "array" : preferredName)
            let itemRef = try ruleReference(for: constraints.item, preferredName: "\(name)_item")
            let listPattern = try listExpression(
                itemReference: itemRef,
                minimumCount: constraints.minimumCount ?? 0,
                maximumCount: constraints.maximumCount
            )
            addRule(
                name: name,
                expression: "\"[\" ws \(listPattern) ws \"]\""
            )
            return name
        case .object(let object):
            let name = uniqueName(object.name.isEmpty ? preferredName : object.name)
            let pairNames = try object.properties.enumerated().map { index, property in
                let pairName = uniqueName("\(name)_pair_\(index)")
                let valueRef = try ruleReference(
                    for: property.schema,
                    preferredName: "\(pairName)_value"
                )
                addRule(
                    name: pairName,
                    expression: "\(Self.quotedJSONStringLiteral(property.name)) ws \":\" ws \(valueRef)"
                )
                return pairName
            }
            let membersName = uniqueName("\(name)_members")
            addRule(
                name: membersName,
                expression: try membersExpression(
                    pairNames: pairNames,
                    properties: object.properties
                )
            )
            addRule(
                name: name,
                expression: "\"{\" ws \(membersName) ws \"}\""
            )
            return name
        }
    }

    private mutating func listExpression(
        itemReference: String,
        minimumCount: Int,
        maximumCount: Int?
    ) throws -> String {
        if let maximumCount, maximumCount < minimumCount {
            throw RuntimeError.unsupportedCapability("vLLM grammar mode requires array maximumCount >= minimumCount")
        }

        let separator = "ws \",\" ws \(itemReference)"
        if let maximumCount {
            let counts = Array(minimumCount...maximumCount)
            let variants = counts.map { count -> String in
                if count == 0 {
                    return "\"\""
                }
                return Array(repeating: itemReference, count: count)
                    .enumerated()
                    .map { index, _ in
                        index == 0 ? itemReference : separator
                    }
                    .joined(separator: " ")
            }
            return variants.joined(separator: " | ")
        }

        if minimumCount == 0 {
            return "\"\" | \(itemReference) (\(separator))*"
        }

        let required = (0..<minimumCount).map { index in
            index == 0 ? itemReference : separator
        }.joined(separator: " ")
        return "\(required) (\(separator))*"
    }

    private mutating func membersExpression(
        pairNames: [String],
        properties: [ObjectSchema.Property]
    ) throws -> String {
        let variants = try memberVariants(
            pairNames: pairNames,
            properties: properties,
            index: 0,
            hasPrevious: false
        )
        return variants.map { $0.isEmpty ? "\"\"" : $0 }.joined(separator: " | ")
    }

    private mutating func memberVariants(
        pairNames: [String],
        properties: [ObjectSchema.Property],
        index: Int,
        hasPrevious: Bool
    ) throws -> [String] {
        guard index < properties.count else {
            return [""]
        }

        let property = properties[index]
        let pairName = pairNames[index]
        let restIncluded = try memberVariants(
            pairNames: pairNames,
            properties: properties,
            index: index + 1,
            hasPrevious: true
        )
        let prefix = hasPrevious ? "ws \",\" ws " : ""
        let included = restIncluded.map { suffix in
            let segment = prefix + pairName
            return suffix.isEmpty ? segment : "\(segment) \(suffix)"
        }

        if property.isOptional {
            let restSkipped = try memberVariants(
                pairNames: pairNames,
                properties: properties,
                index: index + 1,
                hasPrevious: hasPrevious
            )
            return restSkipped + included
        }

        return included
    }

    private mutating func addBaseRules() {
        addRule(name: "ws", expression: "([ \\n\\r\\t])*")
        addRule(name: "json_boolean", expression: "\"true\" | \"false\"")
        addRule(name: "json_integer", expression: "\"-\"? ([0-9])+")
        addRule(name: "json_number", expression: "\"-\"? ([0-9])+ (\".\" ([0-9])+)? ([eE] [+-]? ([0-9])+)?")
        addRule(name: "hex", expression: "[0-9a-fA-F]")
        addRule(name: "escape", expression: "\"\\\\\" ([\"\\\\/bfnrt] | (\"u\" hex hex hex hex))")
        addRule(name: "string_char", expression: "[^\"\\\\\\x00-\\x1F] | escape")
        addRule(name: "json_string", expression: "\"\\\"\" (string_char)* \"\\\"\"")
    }

    private mutating func addRule(name: String, expression: String) {
        guard seenNames.insert(name).inserted else {
            return
        }
        rules.append((name, expression))
    }

    private mutating func uniqueName(_ preferredName: String) -> String {
        let cleaned = Self.clean(preferredName)
        if seenNames.contains(cleaned) == false {
            return cleaned
        }
        var index = 1
        while seenNames.contains("\(cleaned)_\(index)") {
            index += 1
        }
        return "\(cleaned)_\(index)"
    }

    private static func clean(_ value: String) -> String {
        let mapped = value.map { character -> Character in
            if character.isLetter || character.isNumber || character == "_" {
                return character
            }
            return "_"
        }
        let cleaned = String(mapped).trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        return cleaned.isEmpty ? "rule" : cleaned
    }

    private static func quotedJSONStringLiteral(_ value: String) -> String {
        let data = try? JSONSerialization.data(withJSONObject: [value], options: [])
        let string = data.flatMap { String(data: $0, encoding: .utf8) } ?? "[\"\"]"
        let literal = String(string.dropFirst().dropLast())
        return literal
    }
}
