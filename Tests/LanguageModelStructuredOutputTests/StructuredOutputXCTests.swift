import Foundation
import XCTest
@testable import LanguageModelStructuredOutput

final class StructuredOutputXCTests: XCTestCase {
    func testSchemaRoundTrip() throws {
        let schema = OutputSchema.object(
            ObjectSchema(
                name: "Task",
                description: "Portable schema round-trip",
                properties: [
                    .init(name: "title", schema: .string(.init(minLength: 1))),
                    .init(name: "priority", schema: .integer(.init(minimum: 1, maximum: 5))),
                    .init(name: "tags", schema: .array(.init(item: .string())))
                ]
            )
        )

        let data = try JSONEncoder().encode(schema)
        let decoded = try JSONDecoder().decode(OutputSchema.self, from: data)

        XCTAssertEqual(decoded, schema)
    }

    func testCodableHelperAndTranscriptRendering() throws {
        let spec = StructuredOutput.codable(
            Payload.self,
            schema: .object(
                ObjectSchema(
                    name: "Payload",
                    properties: [
                        .init(name: "message", schema: .string()),
                        .init(name: "count", schema: .integer())
                    ]
                )
            )
        )

        let value = try spec.decode(Data(#"{"message":"hello","count":2}"#.utf8))
        let transcript = spec.transcriptRenderer(value)

        XCTAssertEqual(value, Payload(message: "hello", count: 2))
        XCTAssertTrue(transcript.contains(#""message":"hello""#))
    }

    func testJSONSchemaTranslation() throws {
        let schema = OutputSchema.object(
            ObjectSchema(
                name: "Item",
                properties: [
                    .init(name: "name", schema: .string()),
                    .init(name: "nickname", schema: .optional(.string()), isOptional: true)
                ]
            )
        )

        let rendered = try JSONSchemaRenderer.renderString(schema: schema, name: "Item")

        XCTAssertTrue(rendered.contains(#""type":"object""#))
        XCTAssertTrue(rendered.contains(#""required":["name"]"#))
        XCTAssertTrue(rendered.contains(#""additionalProperties":false"#))
    }
}

private struct Payload: Codable, Sendable, Equatable {
    let message: String
    let count: Int
}
