#if canImport(FoundationModels)
import Foundation
import FoundationModels
import XCTest
@testable import LanguageModelApple
@testable import LanguageModelRuntime
@testable import LanguageModelStructuredCore

@available(iOS 26.0, macOS 26.0, visionOS 26.0, *)
@Generable
private struct AppleGeneratedPayload {
    var title: String
}

final class AppleXCTests: XCTestCase {
    func testUnknownModelIDIsRejected() async throws {
        guard #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) else {
            return
        }

        let backend = AppleInferenceBackend()
        let endpoint = ModelEndpoint(backendID: backend.backendID, modelID: "unknown-model")
        let availability = await backend.availability(for: endpoint)

        switch availability.status {
        case .available:
            XCTFail("Expected unavailable status")
        case .unavailable(let reason):
            XCTAssertTrue(reason.contains("Unsupported Apple modelID"))
        }

        do {
            _ = try await backend.makeSession(endpoint: endpoint, instructions: nil, locale: nil)
            XCTFail("Expected unavailable error")
        } catch let error as RuntimeError {
            XCTAssertEqual(
                error,
                .unavailable("Unsupported Apple modelID unknown-model. Supported values are default, general, and contentTagging.")
            )
        }
    }

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

    func testUnsupportedPortableAppleSchemas() throws {
        guard #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) else {
            return
        }

        XCTAssertThrowsError(try AppleSchemaTranslator.translate(.optional(.string()))) { error in
            XCTAssertEqual(
                error as? RuntimeError,
                .unsupportedCapability("Apple does not support top-level optional schemas")
            )
        }

        XCTAssertThrowsError(
            try AppleSchemaTranslator.translate(
                .object(
                    ObjectSchema(
                        name: "Item",
                        properties: [.init(name: "title", schema: .string(.init(minLength: 1, maxLength: 10)))]
                    )
                )
            )
        ) { error in
            XCTAssertEqual(
                error as? RuntimeError,
                .unsupportedCapability("Apple schema translation does not support string minLength")
            )
        }
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
