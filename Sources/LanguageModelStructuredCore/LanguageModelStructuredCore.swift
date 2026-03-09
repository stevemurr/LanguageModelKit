import Foundation
import LanguageModelRuntime

public indirect enum OutputSchema: Sendable, Equatable, Codable {
    case string(StringConstraints = .init())
    case integer(NumberConstraints<Int> = .init())
    case number(NumberConstraints<Double> = .init())
    case boolean
    case array(ArrayConstraints)
    case object(ObjectSchema)
    case enumeration(EnumSchema)
    case optional(OutputSchema)

    private enum CodingKeys: String, CodingKey {
        case kind
        case string
        case integer
        case number
        case array
        case object
        case enumeration
        case optional
    }

    private enum Kind: String, Codable {
        case string
        case integer
        case number
        case boolean
        case array
        case object
        case enumeration
        case optional
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let kind = try container.decode(Kind.self, forKey: .kind)
        switch kind {
        case .string:
            self = .string(try container.decode(StringConstraints.self, forKey: .string))
        case .integer:
            self = .integer(try container.decode(NumberConstraints<Int>.self, forKey: .integer))
        case .number:
            self = .number(try container.decode(NumberConstraints<Double>.self, forKey: .number))
        case .boolean:
            self = .boolean
        case .array:
            self = .array(try container.decode(ArrayConstraints.self, forKey: .array))
        case .object:
            self = .object(try container.decode(ObjectSchema.self, forKey: .object))
        case .enumeration:
            self = .enumeration(try container.decode(EnumSchema.self, forKey: .enumeration))
        case .optional:
            self = .optional(try container.decode(OutputSchema.self, forKey: .optional))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .string(let constraints):
            try container.encode(Kind.string, forKey: .kind)
            try container.encode(constraints, forKey: .string)
        case .integer(let constraints):
            try container.encode(Kind.integer, forKey: .kind)
            try container.encode(constraints, forKey: .integer)
        case .number(let constraints):
            try container.encode(Kind.number, forKey: .kind)
            try container.encode(constraints, forKey: .number)
        case .boolean:
            try container.encode(Kind.boolean, forKey: .kind)
        case .array(let constraints):
            try container.encode(Kind.array, forKey: .kind)
            try container.encode(constraints, forKey: .array)
        case .object(let schema):
            try container.encode(Kind.object, forKey: .kind)
            try container.encode(schema, forKey: .object)
        case .enumeration(let schema):
            try container.encode(Kind.enumeration, forKey: .kind)
            try container.encode(schema, forKey: .enumeration)
        case .optional(let schema):
            try container.encode(Kind.optional, forKey: .kind)
            try container.encode(schema, forKey: .optional)
        }
    }
}

public struct StringConstraints: Sendable, Equatable, Codable {
    public var regex: String?
    public var minLength: Int?
    public var maxLength: Int?

    public init(
        regex: String? = nil,
        minLength: Int? = nil,
        maxLength: Int? = nil
    ) {
        self.regex = regex
        self.minLength = minLength
        self.maxLength = maxLength
    }
}

public struct NumberConstraints<T: Sendable & Equatable & Codable>: Sendable, Equatable, Codable {
    public var minimum: T?
    public var maximum: T?

    public init(
        minimum: T? = nil,
        maximum: T? = nil
    ) {
        self.minimum = minimum
        self.maximum = maximum
    }
}

public struct ArrayConstraints: Sendable, Equatable, Codable {
    public var item: OutputSchema
    public var minimumCount: Int?
    public var maximumCount: Int?

    public init(
        item: OutputSchema,
        minimumCount: Int? = nil,
        maximumCount: Int? = nil
    ) {
        self.item = item
        self.minimumCount = minimumCount
        self.maximumCount = maximumCount
    }
}

public struct ObjectSchema: Sendable, Equatable, Codable {
    public struct Property: Sendable, Equatable, Codable {
        public var name: String
        public var description: String?
        public var schema: OutputSchema
        public var isOptional: Bool

        public init(
            name: String,
            description: String? = nil,
            schema: OutputSchema,
            isOptional: Bool = false
        ) {
            self.name = name
            self.description = description
            self.schema = schema
            self.isOptional = isOptional
        }
    }

    public var name: String
    public var description: String?
    public var properties: [Property]

    public init(
        name: String,
        description: String? = nil,
        properties: [Property]
    ) {
        self.name = name
        self.description = description
        self.properties = properties
    }
}

public struct EnumSchema: Sendable, Equatable, Codable {
    public var name: String?
    public var cases: [String]

    public init(
        name: String? = nil,
        cases: [String]
    ) {
        self.name = name
        self.cases = cases
    }
}

public struct StructuredOutputSpec<Value: Sendable>: Sendable {
    public let schema: OutputSchema
    public let decode: @Sendable (Data) throws -> Value
    public let transcriptRenderer: @Sendable (Value) -> String

    public init(
        schema: OutputSchema,
        decode: @escaping @Sendable (Data) throws -> Value,
        transcriptRenderer: @escaping @Sendable (Value) -> String
    ) {
        self.schema = schema
        self.decode = decode
        self.transcriptRenderer = transcriptRenderer
    }
}

public protocol StructuredOutputBackend: Sendable {
    var backendID: String { get }

    func generateStructured<Value: Sendable>(
        endpoint: ModelEndpoint,
        instructions: String?,
        locale: Locale?,
        prompt: String,
        spec: StructuredOutputSpec<Value>,
        options: StructuredGenerationOptions
    ) async throws -> StructuredGenerationResult<Value>
}

public struct StructuredGenerationOptions: Sendable, Equatable {
    public var maximumResponseTokens: Int?
    public var deterministic: Bool

    public init(
        maximumResponseTokens: Int? = nil,
        deterministic: Bool = false
    ) {
        self.maximumResponseTokens = maximumResponseTokens
        self.deterministic = deterministic
    }
}

public struct StructuredGenerationResult<Value: Sendable>: Sendable {
    public let value: Value
    public let transcriptText: String

    public init(
        value: Value,
        transcriptText: String
    ) {
        self.value = value
        self.transcriptText = transcriptText
    }
}
