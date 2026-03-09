// swift-tools-version: 6.2
import PackageDescription

let package = Package(
    name: "LanguageModelKit",
    platforms: [
        .macOS(.v13),
        .iOS(.v16),
        .visionOS(.v1)
    ],
    products: [
        .library(name: "LanguageModelRuntime", targets: ["LanguageModelRuntime"]),
        .library(name: "LanguageModelStructuredOutput", targets: ["LanguageModelStructuredOutput"]),
        .library(name: "LanguageModelContextManagement", targets: ["LanguageModelContextManagement"]),
        .library(name: "LanguageModelApple", targets: ["LanguageModelApple"]),
        .library(name: "LanguageModelOpenAI", targets: ["LanguageModelOpenAI"]),
        .library(name: "LanguageModelVLLM", targets: ["LanguageModelVLLM"])
    ],
    targets: [
        .target(
            name: "LanguageModelRuntime",
            path: "Sources/LanguageModelRuntime"
        ),
        .target(
            name: "LanguageModelStructuredCore",
            dependencies: ["LanguageModelRuntime"],
            path: "Sources/LanguageModelStructuredCore"
        ),
        .target(
            name: "LanguageModelStructuredOutput",
            dependencies: [
                "LanguageModelRuntime",
                "LanguageModelStructuredCore"
            ],
            path: "Sources/LanguageModelStructuredOutput"
        ),
        .target(
            name: "LanguageModelContextManagement",
            dependencies: [
                "LanguageModelRuntime",
                "LanguageModelStructuredCore"
            ],
            path: "Sources/LanguageModelContextManagement"
        ),
        .target(
            name: "LanguageModelApple",
            dependencies: [
                "LanguageModelRuntime",
                "LanguageModelStructuredCore",
                "LanguageModelStructuredOutput"
            ],
            path: "Sources/LanguageModelApple"
        ),
        .target(
            name: "LanguageModelOpenAI",
            dependencies: [
                "LanguageModelRuntime",
                "LanguageModelStructuredCore",
                "LanguageModelStructuredOutput"
            ],
            path: "Sources/LanguageModelOpenAI"
        ),
        .target(
            name: "LanguageModelVLLM",
            dependencies: [
                "LanguageModelRuntime",
                "LanguageModelStructuredCore",
                "LanguageModelStructuredOutput"
            ],
            path: "Sources/LanguageModelVLLM"
        ),
        .testTarget(
            name: "LanguageModelRuntimeTests",
            dependencies: ["LanguageModelRuntime"],
            path: "Tests/LanguageModelRuntimeTests",
            sources: ["RuntimeXCTests.swift"]
        ),
        .testTarget(
            name: "LanguageModelStructuredOutputTests",
            dependencies: [
                "LanguageModelRuntime",
                "LanguageModelStructuredCore",
                "LanguageModelStructuredOutput"
            ],
            path: "Tests/LanguageModelStructuredOutputTests",
            sources: ["StructuredOutputXCTests.swift"]
        ),
        .testTarget(
            name: "LanguageModelContextManagementTests",
            dependencies: [
                "LanguageModelRuntime",
                "LanguageModelStructuredCore",
                "LanguageModelStructuredOutput",
                "LanguageModelContextManagement"
            ],
            path: "Tests/LanguageModelContextManagementTests",
            sources: ["ContextManagementXCTests.swift"]
        ),
        .testTarget(
            name: "LanguageModelAppleTests",
            dependencies: [
                "LanguageModelRuntime",
                "LanguageModelStructuredCore",
                "LanguageModelStructuredOutput",
                "LanguageModelApple"
            ],
            path: "Tests/LanguageModelAppleTests",
            sources: ["AppleXCTests.swift"]
        ),
        .testTarget(
            name: "LanguageModelOpenAITests",
            dependencies: [
                "LanguageModelRuntime",
                "LanguageModelStructuredCore",
                "LanguageModelStructuredOutput",
                "LanguageModelOpenAI"
            ],
            path: "Tests/LanguageModelOpenAITests",
            sources: ["OpenAIXCTests.swift"]
        ),
        .testTarget(
            name: "LanguageModelVLLMTests",
            dependencies: [
                "LanguageModelRuntime",
                "LanguageModelStructuredCore",
                "LanguageModelStructuredOutput",
                "LanguageModelVLLM"
            ],
            path: "Tests/LanguageModelVLLMTests",
            sources: ["VLLMXCTests.swift"]
        ),
        .testTarget(
            name: "LanguageModelIntegrationTests",
            dependencies: [
                "LanguageModelRuntime",
                "LanguageModelStructuredCore",
                "LanguageModelStructuredOutput",
                "LanguageModelContextManagement",
                "LanguageModelApple",
                "LanguageModelOpenAI",
                "LanguageModelVLLM"
            ],
            path: "Tests/LanguageModelIntegrationTests",
            sources: ["IntegrationXCTests.swift"]
        )
    ],
    swiftLanguageModes: [.v6]
)
