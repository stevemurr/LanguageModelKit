#if canImport(FoundationModels)
import Foundation
import FoundationModels
import XCTest
@testable import LanguageModelApple
@testable import LanguageModelStructuredCore

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
@Generable
private struct AppleGeneratedPayload {
    var title: String
}

final class AppleXCTests: XCTestCase {
    func testSchemaTranslator() throws {
        guard #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) else {
            return
        }

        let schema = OutputSchema.object(
            ObjectSchema(
                name: "Item",
                properties: [
                    .init(name: "title", schema: .string()),
                    .init(name: "priority", schema: .integer(.init(minimum: 1, maximum: 5)))
                ]
            )
        )

        let translated = try AppleSchemaTranslator.translate(schema)
        XCTAssertTrue(translated.debugDescription.contains("Item"))
        XCTAssertTrue(translated.debugDescription.contains("title"))
    }

    func testGenerableHelper() throws {
        guard #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) else {
            return
        }

        let spec = StructuredOutputSpec<AppleGeneratedPayload>.appleGenerable(
            schema: .object(
                ObjectSchema(
                    name: "AppleGeneratedPayload",
                    properties: [.init(name: "title", schema: .string())]
                )
            )
        )
        let value = try spec.decode(Data(#"{"title":"hello"}"#.utf8))

        XCTAssertEqual(value.title, "hello")
    }
}
#else
import XCTest

final class AppleXCTests: XCTestCase {
    func testPlaceholder() {
        XCTAssertTrue(true)
    }
}
#endif
