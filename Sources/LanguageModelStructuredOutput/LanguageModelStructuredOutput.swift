@_exported import LanguageModelRuntime
@_exported import LanguageModelStructuredCore
import Foundation

public enum StructuredOutput {
    public static func codable<Value: Codable & Sendable>(
        _ type: Value.Type,
        schema: OutputSchema,
        renderTranscript: @escaping @Sendable (Value) -> String = { value in
            let encoder = JSONEncoder()
            return (try? String(data: encoder.encode(value), encoding: .utf8)) ?? "{}"
        }
    ) -> StructuredOutputSpec<Value> {
        StructuredOutputSpec(
            schema: schema,
            decode: { data in
                try JSONDecoder().decode(Value.self, from: data)
            },
            transcriptRenderer: renderTranscript
        )
    }
}

package enum StructuredSchemaError: Error, Sendable, Equatable {
    case unsupported(String)
}

package indirect enum JSONValue: Sendable, Equatable {
    case string(String)
    case number(Double)
    case integer(Int)
    case bool(Bool)
    case array([JSONValue])
    case object([String: JSONValue])
    case null

    func asObject() -> Any {
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

package enum JSONSchemaRenderer {
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
            throw StructuredSchemaError.unsupported("Unable to encode JSON schema")
        }
        return string
    }

    package static func renderObject(
        schema: OutputSchema,
        name: String = "result"
    ) throws -> JSONValue {
        try renderNode(schema: schema, fallbackName: name)
    }

    private static func renderNode(
        schema: OutputSchema,
        fallbackName: String
    ) throws -> JSONValue {
        switch schema {
        case .string(let constraints):
            var object: [String: JSONValue] = ["type": .string("string")]
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
            var object: [String: JSONValue] = ["type": .string("integer")]
            if let minimum = constraints.minimum {
                object["minimum"] = .integer(minimum)
            }
            if let maximum = constraints.maximum {
                object["maximum"] = .integer(maximum)
            }
            return .object(object)
        case .number(let constraints):
            var object: [String: JSONValue] = ["type": .string("number")]
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
            var object: [String: JSONValue] = [
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
            let properties = try schema.properties.reduce(into: [String: JSONValue]()) { result, property in
                result[property.name] = try renderNode(schema: property.schema, fallbackName: property.name)
            }
            let required = schema.properties
                .filter { $0.isOptional == false }
                .map(\.name)
                .sorted()
            var object: [String: JSONValue] = [
                "type": .string("object"),
                "title": .string(schema.name),
                "properties": .object(properties),
                "additionalProperties": .bool(false)
            ]
            if let description = schema.description {
                object["description"] = .string(description)
            }
            if required.isEmpty == false {
                object["required"] = .array(required.map(JSONValue.string))
            }
            return .object(object)
        case .enumeration(let schema):
            var object: [String: JSONValue] = [
                "type": .string("string"),
                "enum": .array(schema.cases.map(JSONValue.string))
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
