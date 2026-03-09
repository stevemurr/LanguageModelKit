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

package enum JSONSchemaRenderer {
    package static func renderData(
        schema: OutputSchema,
        name: String = "result"
    ) throws -> Data {
        try PortableSchemaRenderer.renderData(schema: schema, name: name)
    }

    package static func renderString(
        schema: OutputSchema,
        name: String = "result"
    ) throws -> String {
        try PortableSchemaRenderer.renderString(schema: schema, name: name)
    }

    package static func renderObject(
        schema: OutputSchema,
        name: String = "result"
    ) throws -> PortableJSONValue {
        try PortableSchemaRenderer.renderObject(schema: schema, name: name)
    }
}
