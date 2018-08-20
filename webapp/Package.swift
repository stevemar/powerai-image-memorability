// swift-tools-version:4.0
import PackageDescription

let package = Package(
    name: "MemExtract",
    dependencies: [
        .package(url: "https://github.com/IBM-Swift/Kitura.git", from: "2.3.0"),
        .package(url: "https://github.com/IBM-Swift/Kitura-MustacheTemplateEngine.git", from: "1.8.0")
    ],
    targets: [
        .target(name: "MemorabilityExtractor", dependencies: ["Kitura", "KituraMustache"])
    ]
)
