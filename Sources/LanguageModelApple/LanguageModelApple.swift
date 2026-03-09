#if canImport(FoundationModels)
import Foundation
import FoundationModels
import LanguageModelRuntime
import LanguageModelStructuredCore
import LanguageModelStructuredOutput

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
public final class AppleInferenceBackend: InferenceBackend {
    public let backendID: String

    public init(backendID: String = "apple") {
        self.backendID = backendID
    }

    public func availability(for endpoint: ModelEndpoint) async -> RuntimeAvailability {
        let model = makeModel(for: endpoint)
        return RuntimeAvailability(
            status: availabilityStatus(for: model.availability),
            capabilities: RuntimeCapabilities(
                supportsTextGeneration: true,
                supportsTextStreaming: true,
                supportsStructuredOutput: true,
                supportsExactTokenEstimation: false,
                supportsLocaleHints: true
            )
        )
    }

    public func makeSession(
        endpoint: ModelEndpoint,
        instructions: String?,
        locale: Locale?
    ) async throws -> any InferenceSession {
        let model = makeModel(for: endpoint)
        if let locale, model.supportsLocale(locale) == false {
            throw RuntimeError.unsupportedLocale("Locale \(locale.identifier) is unsupported")
        }

        let resolvedInstructions = LocaleInstructionBuilder.build(
            instructions: instructions,
            locale: locale
        )
        return AppleInferenceSession(
            session: LanguageModelSession(
                model: model,
                tools: [],
                instructions: resolvedInstructions
            )
        )
    }

    public func contextWindowTokens(for endpoint: ModelEndpoint) async -> Int? {
        endpoint.contextWindowOverride
    }

    public func exactTokenEstimator(for endpoint: ModelEndpoint) async -> (any TokenEstimating)? {
        _ = endpoint
        return nil
    }

    fileprivate func makeModel(for endpoint: ModelEndpoint) -> SystemLanguageModel {
        if let adapterName = endpoint.options["adapter"],
           adapterName.isEmpty == false,
           let adapter = try? SystemLanguageModel.Adapter(name: adapterName) {
            return SystemLanguageModel(adapter: adapter, guardrails: guardrails(for: endpoint))
        }

        switch endpoint.modelID {
        case "default":
            return SystemLanguageModel.default
        case "general":
            return SystemLanguageModel(useCase: .general, guardrails: guardrails(for: endpoint))
        case "contentTagging":
            return SystemLanguageModel(useCase: .contentTagging, guardrails: guardrails(for: endpoint))
        default:
            return SystemLanguageModel(useCase: .general, guardrails: guardrails(for: endpoint))
        }
    }

    private func guardrails(for endpoint: ModelEndpoint) -> SystemLanguageModel.Guardrails {
        switch endpoint.options["guardrails"] {
        case "permissiveContentTransformations":
            return .permissiveContentTransformations
        default:
            return .default
        }
    }

    private func availabilityStatus(
        for availability: SystemLanguageModel.Availability
    ) -> RuntimeAvailability.Status {
        switch availability {
        case .available:
            return .available
        case .unavailable(let reason):
            switch reason {
            case .deviceNotEligible:
                return .unavailable(reason: "Device not eligible for Foundation Models")
            case .appleIntelligenceNotEnabled:
                return .unavailable(reason: "Apple Intelligence is not enabled")
            case .modelNotReady:
                return .unavailable(reason: "Foundation Models assets are not ready")
            @unknown default:
                return .unavailable(reason: "Foundation Models is unavailable")
            }
        @unknown default:
            return .unavailable(reason: "Foundation Models is unavailable")
        }
    }
}

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
public final class AppleStructuredOutputBackend: StructuredOutputBackend {
    public let backendID: String

    public init(backendID: String = "apple") {
        self.backendID = backendID
    }

    public func generateStructured<Value>(
        endpoint: ModelEndpoint,
        instructions: String?,
        locale: Locale?,
        prompt: String,
        spec: StructuredOutputSpec<Value>,
        options: StructuredGenerationOptions
    ) async throws -> StructuredGenerationResult<Value> {
        let model = AppleInferenceBackend(backendID: backendID).makeModel(for: endpoint)
        if let locale, model.supportsLocale(locale) == false {
            throw RuntimeError.unsupportedLocale("Locale \(locale.identifier) is unsupported")
        }

        let schema = try AppleSchemaTranslator.translate(spec.schema)
        let session = LanguageModelSession(
            model: model,
            tools: [],
            instructions: LocaleInstructionBuilder.build(
                instructions: instructions,
                locale: locale
            )
        )

        do {
            let response = try await session.respond(
                to: prompt,
                schema: schema,
                includeSchemaInPrompt: true,
                options: makeOptions(options.maximumResponseTokens, deterministic: options.deterministic)
            )
            let data = Data(response.rawContent.jsonString.utf8)
            let value = try spec.decode(data)
            return StructuredGenerationResult(
                value: value,
                transcriptText: spec.transcriptRenderer(value)
            )
        } catch let error as LanguageModelSession.GenerationError {
            throw AppleErrorMapper.map(error)
        } catch let error as RuntimeError {
            throw error
        } catch {
            throw RuntimeError.generationFailed(error.localizedDescription)
        }
    }

    public func generateStructured<Value: Generable>(
        endpoint: ModelEndpoint,
        instructions: String?,
        locale: Locale?,
        prompt: String,
        spec: GenerableStructuredOutputSpec<Value>,
        options: StructuredGenerationOptions
    ) async throws -> StructuredGenerationResult<Value> {
        let model = AppleInferenceBackend(backendID: backendID).makeModel(for: endpoint)
        if let locale, model.supportsLocale(locale) == false {
            throw RuntimeError.unsupportedLocale("Locale \(locale.identifier) is unsupported")
        }

        let session = LanguageModelSession(
            model: model,
            tools: [],
            instructions: LocaleInstructionBuilder.build(
                instructions: instructions,
                locale: locale
            )
        )

        do {
            let response = try await session.respond(
                to: prompt,
                generating: Value.self,
                includeSchemaInPrompt: true,
                options: makeOptions(options.maximumResponseTokens, deterministic: options.deterministic)
            )
            return StructuredGenerationResult(
                value: response.content,
                transcriptText: spec.transcriptRenderer(response.content)
            )
        } catch let error as LanguageModelSession.GenerationError {
            throw AppleErrorMapper.map(error)
        } catch {
            throw RuntimeError.generationFailed(error.localizedDescription)
        }
    }

    private func makeOptions(
        _ maximumResponseTokens: Int?,
        deterministic: Bool
    ) -> GenerationOptions {
        if deterministic {
            return GenerationOptions(
                sampling: .greedy,
                temperature: 0,
                maximumResponseTokens: maximumResponseTokens
            )
        }
        return GenerationOptions(maximumResponseTokens: maximumResponseTokens)
    }
}

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
public struct GenerableStructuredOutputSpec<Value: Generable>: Sendable {
    public let transcriptRenderer: @Sendable (Value) -> String

    public init(
        transcriptRenderer: @escaping @Sendable (Value) -> String = { value in
            let generated = value.generatedContent
            return generated.jsonString
        }
    ) {
        self.transcriptRenderer = transcriptRenderer
    }
}

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
public extension StructuredOutputSpec where Value: Generable {
    static func appleGenerable(
        schema: OutputSchema,
        renderTranscript: @escaping @Sendable (Value) -> String = { value in
            value.generatedContent.jsonString
        }
    ) -> StructuredOutputSpec<Value> {
        StructuredOutputSpec(
            schema: schema,
            decode: { data in
                let json = String(decoding: data, as: UTF8.self)
                return try Value(GeneratedContent(json: json))
            },
            transcriptRenderer: renderTranscript
        )
    }
}

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
actor AppleInferenceSession: InferenceSession {
    private let session: LanguageModelSession

    init(session: LanguageModelSession) {
        self.session = session
    }

    func generateText(
        prompt: String,
        options: TextGenerationOptions
    ) async throws -> TextGenerationResult {
        do {
            let response = try await session.respond(
                to: prompt,
                options: makeOptions(from: options)
            )
            return TextGenerationResult(text: response.content)
        } catch let error as LanguageModelSession.GenerationError {
            throw AppleErrorMapper.map(error)
        } catch {
            throw RuntimeError.generationFailed(error.localizedDescription)
        }
    }

    func streamText(
        prompt: String,
        options: TextGenerationOptions
    ) async -> AsyncThrowingStream<TextStreamEvent, Error> {
        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    let stream = session.streamResponse(
                        to: prompt,
                        options: makeOptions(from: options)
                    )
                    for try await snapshot in stream {
                        continuation.yield(.partial(snapshot.content))
                    }
                    let collected = try await stream.collect()
                    continuation.yield(.completed(.init(text: collected.content)))
                    continuation.finish()
                } catch let error as LanguageModelSession.GenerationError {
                    continuation.finish(throwing: AppleErrorMapper.map(error))
                } catch {
                    continuation.finish(throwing: RuntimeError.generationFailed(error.localizedDescription))
                }
            }

            continuation.onTermination = { @Sendable _ in
                task.cancel()
            }
        }
    }

    private func makeOptions(from options: TextGenerationOptions) -> GenerationOptions {
        if options.deterministic {
            return GenerationOptions(
                sampling: .greedy,
                temperature: options.temperature ?? 0,
                maximumResponseTokens: options.maximumResponseTokens
            )
        }
        return GenerationOptions(
            temperature: options.temperature,
            maximumResponseTokens: options.maximumResponseTokens
        )
    }
}

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
enum AppleSchemaTranslator {
    static func translate(_ schema: OutputSchema) throws -> GenerationSchema {
        if case .optional = schema {
            throw RuntimeError.unsupportedCapability("Apple does not support top-level optional schemas")
        }
        return try GenerationSchema(
            root: translateDynamic(schema, fallbackName: "Result"),
            dependencies: []
        )
    }

    private static func translateDynamic(
        _ schema: OutputSchema,
        fallbackName: String
    ) throws -> DynamicGenerationSchema {
        switch schema {
        case .string(let constraints):
            if let minLength = constraints.minLength, minLength > 0 {
                throw RuntimeError.unsupportedCapability("Apple schema translation does not support string minLength")
            }
            if constraints.maxLength != nil {
                throw RuntimeError.unsupportedCapability("Apple schema translation does not support string maxLength")
            }
            if let regex = constraints.regex {
                return DynamicGenerationSchema(
                    type: String.self,
                    guides: [.pattern(try Regex(regex))]
                )
            }
            return DynamicGenerationSchema(type: String.self)
        case .integer(let constraints):
            return DynamicGenerationSchema(
                type: Int.self,
                guides: integerGuides(constraints)
            )
        case .number(let constraints):
            return DynamicGenerationSchema(
                type: Double.self,
                guides: doubleGuides(constraints)
            )
        case .boolean:
            return DynamicGenerationSchema(type: Bool.self)
        case .array(let constraints):
            let item = try translateDynamic(constraints.item, fallbackName: "\(fallbackName)Item")
            return DynamicGenerationSchema(
                arrayOf: item,
                minimumElements: constraints.minimumCount,
                maximumElements: constraints.maximumCount
            )
        case .object(let object):
            let properties = try object.properties.map { property in
                let translatedSchema: OutputSchema
                let isOptional: Bool
                if case .optional(let wrapped) = property.schema {
                    translatedSchema = wrapped
                    isOptional = true
                } else {
                    translatedSchema = property.schema
                    isOptional = property.isOptional
                }
                return DynamicGenerationSchema.Property(
                    name: property.name,
                    description: property.description,
                    schema: try translateDynamic(translatedSchema, fallbackName: property.name),
                    isOptional: isOptional
                )
            }
            return DynamicGenerationSchema(
                name: object.name,
                description: object.description,
                properties: properties
            )
        case .enumeration(let schema):
            return DynamicGenerationSchema(
                name: schema.name ?? fallbackName,
                anyOf: schema.cases
            )
        case .optional:
            throw RuntimeError.unsupportedCapability("Apple only supports optional fields inside object properties")
        }
    }

    private static func integerGuides(
        _ constraints: NumberConstraints<Int>
    ) -> [GenerationGuide<Int>] {
        var guides: [GenerationGuide<Int>] = []
        if let minimum = constraints.minimum {
            guides.append(.minimum(minimum))
        }
        if let maximum = constraints.maximum {
            guides.append(.maximum(maximum))
        }
        return guides
    }

    private static func doubleGuides(
        _ constraints: NumberConstraints<Double>
    ) -> [GenerationGuide<Double>] {
        var guides: [GenerationGuide<Double>] = []
        if let minimum = constraints.minimum {
            guides.append(.minimum(minimum))
        }
        if let maximum = constraints.maximum {
            guides.append(.maximum(maximum))
        }
        return guides
    }
}

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
enum AppleErrorMapper {
    static func map(_ error: LanguageModelSession.GenerationError) -> RuntimeError {
        switch error {
        case .exceededContextWindowSize(let context):
            return .contextOverflow(context.debugDescription)
        case .unsupportedLanguageOrLocale(let context):
            return .unsupportedLocale(context.debugDescription)
        case .refusal(let refusal, _):
            return .refusal(String(describing: refusal))
        case .unsupportedGuide(let context):
            return .unsupportedCapability(context.debugDescription)
        case .decodingFailure(let context):
            return .generationFailed(context.debugDescription)
        default:
            return .generationFailed(error.localizedDescription)
        }
    }
}

enum LocaleInstructionBuilder {
    static func build(
        instructions: String?,
        locale: Locale?
    ) -> String? {
        let localeHint = locale.map { "Target locale: \($0.identifier)." }
        switch (instructions, localeHint) {
        case let (instructions?, localeHint?):
            return "\(instructions)\n\n\(localeHint)"
        case let (instructions?, nil):
            return instructions
        case let (nil, localeHint?):
            return localeHint
        case (nil, nil):
            return nil
        }
    }
}
#else
import Foundation
import LanguageModelRuntime
import LanguageModelStructuredCore

public final class AppleInferenceBackend: InferenceBackend {
    public let backendID: String

    public init(backendID: String = "apple") {
        self.backendID = backendID
    }

    public func availability(for endpoint: ModelEndpoint) async -> RuntimeAvailability {
        _ = endpoint
        return RuntimeAvailability(
            status: .unavailable(reason: "FoundationModels is unavailable"),
            capabilities: RuntimeCapabilities()
        )
    }

    public func makeSession(
        endpoint: ModelEndpoint,
        instructions: String?,
        locale: Locale?
    ) async throws -> any InferenceSession {
        _ = endpoint
        _ = instructions
        _ = locale
        throw RuntimeError.unavailable("FoundationModels is unavailable")
    }

    public func contextWindowTokens(for endpoint: ModelEndpoint) async -> Int? {
        endpoint.contextWindowOverride
    }

    public func exactTokenEstimator(for endpoint: ModelEndpoint) async -> (any TokenEstimating)? {
        _ = endpoint
        return nil
    }
}

public final class AppleStructuredOutputBackend: StructuredOutputBackend {
    public let backendID: String

    public init(backendID: String = "apple") {
        self.backendID = backendID
    }

    public func generateStructured<Value>(
        endpoint: ModelEndpoint,
        instructions: String?,
        locale: Locale?,
        prompt: String,
        spec: StructuredOutputSpec<Value>,
        options: StructuredGenerationOptions
    ) async throws -> StructuredGenerationResult<Value> {
        _ = endpoint
        _ = instructions
        _ = locale
        _ = prompt
        _ = spec
        _ = options
        throw RuntimeError.unavailable("FoundationModels is unavailable")
    }
}
#endif
