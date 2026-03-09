import Foundation
#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

public struct ModelEndpoint: Codable, Sendable, Equatable {
    public var backendID: String
    public var modelID: String
    public var options: [String: String]
    public var contextWindowOverride: Int?

    public init(
        backendID: String,
        modelID: String,
        options: [String: String] = [:],
        contextWindowOverride: Int? = nil
    ) {
        self.backendID = backendID
        self.modelID = modelID
        self.options = options
        self.contextWindowOverride = contextWindowOverride
    }
}

public struct RuntimeCapabilities: Sendable, Equatable {
    public var supportsTextGeneration: Bool
    public var supportsTextStreaming: Bool
    public var supportsStructuredOutput: Bool
    public var supportsExactTokenEstimation: Bool
    public var supportsLocaleHints: Bool

    public init(
        supportsTextGeneration: Bool = true,
        supportsTextStreaming: Bool = false,
        supportsStructuredOutput: Bool = false,
        supportsExactTokenEstimation: Bool = false,
        supportsLocaleHints: Bool = false
    ) {
        self.supportsTextGeneration = supportsTextGeneration
        self.supportsTextStreaming = supportsTextStreaming
        self.supportsStructuredOutput = supportsStructuredOutput
        self.supportsExactTokenEstimation = supportsExactTokenEstimation
        self.supportsLocaleHints = supportsLocaleHints
    }
}

public struct RuntimeAvailability: Sendable, Equatable {
    public enum Status: Sendable, Equatable {
        case available
        case unavailable(reason: String)
    }

    public var status: Status
    public var capabilities: RuntimeCapabilities

    public init(
        status: Status,
        capabilities: RuntimeCapabilities = .init()
    ) {
        self.status = status
        self.capabilities = capabilities
    }
}

public enum RuntimeError: Error, Sendable, Equatable {
    case unavailable(String)
    case unsupportedCapability(String)
    case unsupportedLocale(String)
    case contextOverflow(String)
    case refusal(String)
    case generationFailed(String)
    case transportFailed(String)
}

public protocol InferenceBackend: Sendable {
    var backendID: String { get }

    func availability(for endpoint: ModelEndpoint) async -> RuntimeAvailability
    func makeSession(
        endpoint: ModelEndpoint,
        instructions: String?,
        locale: Locale?
    ) async throws -> any InferenceSession
    func contextWindowTokens(for endpoint: ModelEndpoint) async -> Int?
    func exactTokenEstimator(for endpoint: ModelEndpoint) async -> (any TokenEstimating)?
}

public protocol InferenceSession: Sendable {
    func generateText(
        prompt: String,
        options: TextGenerationOptions
    ) async throws -> TextGenerationResult

    func streamText(
        prompt: String,
        options: TextGenerationOptions
    ) async -> AsyncThrowingStream<TextStreamEvent, Error>
}

public protocol TokenEstimating: Sendable {
    func estimate(
        prompt: RenderedPrompt,
        reservedOutputTokens: Int
    ) async -> TokenEstimate?
}

public struct TextGenerationOptions: Sendable, Equatable {
    public var maximumResponseTokens: Int?
    public var temperature: Double?
    public var deterministic: Bool

    public init(
        maximumResponseTokens: Int? = nil,
        temperature: Double? = nil,
        deterministic: Bool = false
    ) {
        self.maximumResponseTokens = maximumResponseTokens
        self.temperature = temperature
        self.deterministic = deterministic
    }
}

public struct TextGenerationResult: Sendable, Equatable {
    public var text: String

    public init(text: String) {
        self.text = text
    }
}

public enum TextStreamEvent: Sendable, Equatable {
    case partial(String)
    case completed(TextGenerationResult)
}

public enum RenderedPromptSection: String, Codable, Sendable, Equatable, Hashable, CaseIterable {
    case instructions
    case durableMemory
    case retrievedMemory
    case recentTail
    case currentPrompt
    case schema
}

public struct RenderedPromptComponent: Codable, Sendable, Equatable {
    public var section: RenderedPromptSection
    public var text: String

    public init(section: RenderedPromptSection, text: String) {
        self.section = section
        self.text = text
    }
}

public struct RenderedPrompt: Codable, Sendable, Equatable {
    public var components: [RenderedPromptComponent]

    public init(components: [RenderedPromptComponent]) {
        self.components = components
    }

    public init(
        instructions: String?,
        durableMemory: [String],
        retrievedMemory: [String],
        recentTail: [String],
        currentPrompt: String,
        schemaText: String?
    ) {
        var built: [RenderedPromptComponent] = []
        if let instructions, instructions.isEmpty == false {
            built.append(.init(section: .instructions, text: instructions))
        }
        built.append(contentsOf: durableMemory.map { .init(section: .durableMemory, text: $0) })
        built.append(contentsOf: retrievedMemory.map { .init(section: .retrievedMemory, text: $0) })
        built.append(contentsOf: recentTail.map { .init(section: .recentTail, text: $0) })
        built.append(.init(section: .currentPrompt, text: currentPrompt))
        if let schemaText, schemaText.isEmpty == false {
            built.append(.init(section: .schema, text: schemaText))
        }
        self.components = built
    }

    public var text: String {
        components
            .map { component in
                "[\(component.section.rawValue)] \(component.text)"
            }
            .joined(separator: "\n")
    }
}

public struct TokenEstimate: Codable, Sendable, Equatable {
    public var inputTokens: Int
    public var breakdown: [RenderedPromptSection: Int]

    public init(
        inputTokens: Int,
        breakdown: [RenderedPromptSection: Int] = [:]
    ) {
        self.inputTokens = inputTokens
        self.breakdown = breakdown
    }
}

public actor RuntimeRegistry {
    private var backends: [String: any InferenceBackend]

    public init() {
        backends = [:]
    }

    public func register(_ backend: any InferenceBackend) async {
        backends[backend.backendID] = backend
    }

    public func backend(for backendID: String) async throws -> any InferenceBackend {
        guard let backend = backends[backendID] else {
            throw RuntimeError.unavailable("No backend registered for \(backendID)")
        }
        return backend
    }
}

package struct HTTPResponse: Sendable {
    package let response: HTTPURLResponse
    package let data: Data
}

package struct HTTPClient: Sendable {
    package let session: URLSession

    package init(session: URLSession = .shared) {
        self.session = session
    }

    package func data(for request: URLRequest) async throws -> HTTPResponse {
        do {
            let (data, response) = try await session.data(for: request)
            guard let http = response as? HTTPURLResponse else {
                throw RuntimeError.transportFailed("Invalid response type")
            }
            return HTTPResponse(response: http, data: data)
        } catch let error as RuntimeError {
            throw error
        } catch {
            throw RuntimeError.transportFailed(error.localizedDescription)
        }
    }

    package func bytes(for request: URLRequest) async throws -> (URLSession.AsyncBytes, HTTPURLResponse) {
        do {
            let (bytes, response) = try await session.bytes(for: request)
            guard let http = response as? HTTPURLResponse else {
                throw RuntimeError.transportFailed("Invalid response type")
            }
            return (bytes, http)
        } catch let error as RuntimeError {
            throw error
        } catch {
            throw RuntimeError.transportFailed(error.localizedDescription)
        }
    }
}

package struct SSEParser: AsyncSequence, Sendable {
    package typealias Element = String

    package struct AsyncIterator: AsyncIteratorProtocol {
        var iterator: URLSession.AsyncBytes.AsyncIterator

        package mutating func next() async throws -> String? {
            var buffer: [String] = []
            var lineBuffer: [UInt8] = []

            while let byte = try await iterator.next() {
                if byte == 0x0A {
                    let line = normalizedLine(from: lineBuffer)
                    lineBuffer.removeAll(keepingCapacity: true)

                    if let payload = flush(line: line, buffer: &buffer) {
                        return payload == "[DONE]" ? nil : payload
                    }
                    continue
                }
                lineBuffer.append(byte)
            }

            if lineBuffer.isEmpty == false,
               let payload = flush(line: normalizedLine(from: lineBuffer), buffer: &buffer) {
                return payload == "[DONE]" ? nil : payload
            }
            if buffer.isEmpty {
                return nil
            }

            let payload = buffer.joined(separator: "\n")
            return payload == "[DONE]" ? nil : payload
        }

        private func normalizedLine(from bytes: [UInt8]) -> String {
            var line = String(decoding: bytes, as: UTF8.self)
            if line.hasSuffix("\r") {
                line.removeLast()
            }
            return line
        }

        private func flush(
            line: String,
            buffer: inout [String]
        ) -> String? {
            if line.isEmpty {
                guard buffer.isEmpty == false else {
                    return nil
                }
                let payload = buffer.joined(separator: "\n")
                buffer.removeAll(keepingCapacity: true)
                return payload
            }

            guard line.hasPrefix("data:") else {
                return nil
            }

            buffer.append(String(line.dropFirst(5)).trimmingCharacters(in: .whitespaces))
            return nil
        }
    }

    let bytes: URLSession.AsyncBytes

    package init(bytes: URLSession.AsyncBytes) {
        self.bytes = bytes
    }

    package func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(iterator: bytes.makeAsyncIterator())
    }
}

extension URLSession.AsyncBytes {
    package var sseEvents: SSEParser {
        SSEParser(bytes: self)
    }
}

package enum RuntimeOptionParser {
    package static func bool(
        from options: [String: String],
        key: String,
        default defaultValue: Bool
    ) -> Bool {
        guard let value = options[key]?.lowercased() else {
            return defaultValue
        }
        switch value {
        case "1", "true", "yes", "on":
            return true
        case "0", "false", "no", "off":
            return false
        default:
            return defaultValue
        }
    }
}
