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

package protocol LiveStructuredGenerationSession: InferenceSession {
    func generateStructured<Value: Sendable>(
        prompt: String,
        spec: StructuredOutputSpec<Value>,
        options: StructuredGenerationOptions
    ) async throws -> StructuredGenerationResult<Value>
}

package enum PortableSchemaError: Error, Sendable, Equatable {
    case unsupported(String)
}

package indirect enum PortableJSONValue: Sendable, Equatable {
    case string(String)
    case number(Double)
    case integer(Int)
    case bool(Bool)
    case array([PortableJSONValue])
    case object([String: PortableJSONValue])
    case null

    package func asObject() -> Any {
        switch self {
        case .string(let value):
            return value
        case .number(let value):
            return value
        case .integer(let value):
            return value
        case .bool(let value):
            return value
        case .array(let values):
            return values.map { $0.asObject() }
        case .object(let values):
            return values.mapValues { $0.asObject() }
        case .null:
            return NSNull()
        }
    }
}

package enum PortableSchemaRenderer {
    package static func renderData(
        schema: OutputSchema,
        name: String = "result"
    ) throws -> Data {
        let object = try renderObject(schema: schema, name: name)
        return try JSONSerialization.data(withJSONObject: object.asObject(), options: [.sortedKeys])
    }

    package static func renderString(
        schema: OutputSchema,
        name: String = "result"
    ) throws -> String {
        let data = try renderData(schema: schema, name: name)
        guard let string = String(data: data, encoding: .utf8) else {
            throw PortableSchemaError.unsupported("Unable to encode JSON schema")
        }
        return string
    }

    package static func renderObject(
        schema: OutputSchema,
        name: String = "result"
    ) throws -> PortableJSONValue {
        try renderNode(schema: schema, fallbackName: name)
    }

    private static func renderNode(
        schema: OutputSchema,
        fallbackName: String
    ) throws -> PortableJSONValue {
        switch schema {
        case .string(let constraints):
            var object: [String: PortableJSONValue] = ["type": .string("string")]
            if let regex = constraints.regex {
                object["pattern"] = .string(regex)
            }
            if let minLength = constraints.minLength {
                object["minLength"] = .integer(minLength)
            }
            if let maxLength = constraints.maxLength {
                object["maxLength"] = .integer(maxLength)
            }
            return .object(object)
        case .integer(let constraints):
            var object: [String: PortableJSONValue] = ["type": .string("integer")]
            if let minimum = constraints.minimum {
                object["minimum"] = .integer(minimum)
            }
            if let maximum = constraints.maximum {
                object["maximum"] = .integer(maximum)
            }
            return .object(object)
        case .number(let constraints):
            var object: [String: PortableJSONValue] = ["type": .string("number")]
            if let minimum = constraints.minimum {
                object["minimum"] = .number(minimum)
            }
            if let maximum = constraints.maximum {
                object["maximum"] = .number(maximum)
            }
            return .object(object)
        case .boolean:
            return .object(["type": .string("boolean")])
        case .array(let constraints):
            var object: [String: PortableJSONValue] = [
                "type": .string("array"),
                "items": try renderNode(schema: constraints.item, fallbackName: "\(fallbackName)Item")
            ]
            if let minimumCount = constraints.minimumCount {
                object["minItems"] = .integer(minimumCount)
            }
            if let maximumCount = constraints.maximumCount {
                object["maxItems"] = .integer(maximumCount)
            }
            return .object(object)
        case .object(let schema):
            let properties = try schema.properties.reduce(into: [String: PortableJSONValue]()) { result, property in
                result[property.name] = try renderNode(schema: property.schema, fallbackName: property.name)
            }
            let required = schema.properties
                .filter { $0.isOptional == false }
                .map(\.name)
                .sorted()
            var object: [String: PortableJSONValue] = [
                "type": .string("object"),
                "title": .string(schema.name),
                "properties": .object(properties),
                "additionalProperties": .bool(false)
            ]
            if let description = schema.description {
                object["description"] = .string(description)
            }
            if required.isEmpty == false {
                object["required"] = .array(required.map(PortableJSONValue.string))
            }
            return .object(object)
        case .enumeration(let schema):
            var object: [String: PortableJSONValue] = [
                "type": .string("string"),
                "enum": .array(schema.cases.map(PortableJSONValue.string))
            ]
            if let name = schema.name {
                object["title"] = .string(name)
            }
            return .object(object)
        case .optional(let wrapped):
            return .object([
                "anyOf": .array([
                    try renderNode(schema: wrapped, fallbackName: fallbackName),
                    .object(["type": .string("null")])
                ])
            ])
        }
    }
}
