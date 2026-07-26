import AppKit
import Foundation
import Vision

struct OCRLine: Codable {
    let text: String
    let confidence: Float
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}

func fail(_ message: String, code: Int32 = 2) -> Never {
    FileHandle.standardError.write((message + "\n").data(using: .utf8)!)
    exit(code)
}

guard CommandLine.arguments.count >= 2 else {
    fail("usage: vision_ocr IMAGE_PATH [language]")
}

let imagePath = CommandLine.arguments[1]
let language = CommandLine.arguments.count >= 3 ? CommandLine.arguments[2] : "en-US"

guard let image = NSImage(contentsOfFile: imagePath) else {
    fail("cannot open image: \(imagePath)")
}

var imageRect = CGRect(origin: .zero, size: image.size)
guard let cgImage = image.cgImage(
    forProposedRect: &imageRect,
    context: nil,
    hints: nil
) else {
    fail("cannot convert image to CGImage: \(imagePath)")
}

let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.recognitionLanguages = [language]
request.usesLanguageCorrection = true
// Illustrated reference books often use very small italic captions.  Keeping
// this threshold low lets Vision return those lines instead of leaving a few
// English fragments behind.  Confidence filtering still happens in Python.
request.minimumTextHeight = 0.002

let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
do {
    try handler.perform([request])
} catch {
    fail("Vision OCR failed: \(error)")
}

let observations = request.results as? [VNRecognizedTextObservation] ?? []
let lines: [OCRLine] = observations.compactMap { observation in
    guard let candidate = observation.topCandidates(1).first else {
        return nil
    }
    let box = observation.boundingBox
    return OCRLine(
        text: candidate.string,
        confidence: candidate.confidence,
        x: Double(box.origin.x),
        y: Double(box.origin.y),
        width: Double(box.size.width),
        height: Double(box.size.height)
    )
}

do {
    let data = try JSONEncoder().encode(lines)
    FileHandle.standardOutput.write(data)
    FileHandle.standardOutput.write("\n".data(using: .utf8)!)
} catch {
    fail("cannot encode OCR result: \(error)")
}
