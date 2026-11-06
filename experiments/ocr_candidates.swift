// Candidate-only OCR proposals. Recognition is evidence, not verified identity.
import Foundation
import Vision
import AppKit

let args = CommandLine.arguments
guard args.count == 3, let image = NSImage(contentsOfFile: args[1]),
      let cg = image.cgImage(forProposedRect: nil, context: nil, hints: nil) else { exit(2) }
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.recognitionLanguages = ["zh-Hans", "en-US"]
request.usesLanguageCorrection = false
request.minimumTextHeight = 0.002
try VNImageRequestHandler(cgImage: cg, options: [:]).perform([request])
func box(_ b: CGRect) -> [Double] {
    [Double(b.minX)*Double(cg.width), Double(1-b.maxY)*Double(cg.height),
     Double(b.maxX)*Double(cg.width), Double(1-b.minY)*Double(cg.height)]
}
var rows: [[String: Any]] = []
for (i, observation) in (request.results ?? []).enumerated() {
    guard let candidate = observation.topCandidates(1).first else { continue }
    var chars: [[String: Any]] = []
    for index in candidate.string.indices {
        let end = candidate.string.index(after: index)
        let text = String(candidate.string[index..<end])
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { continue }
        if let bounds = try candidate.boundingBox(for: index..<end) {
            chars.append(["text": text, "bbox": box(bounds.boundingBox)])
        }
    }
    rows.append(["id": "ocr\(i)", "text": candidate.string,
                 "confidence": candidate.confidence, "bbox": box(observation.boundingBox), "chars": chars])
}
let output: [String: Any] = ["engine": "Apple Vision", "revision": request.revision,
                            "image_size": [cg.width, cg.height], "regions": rows]
try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys]).write(to: URL(fileURLWithPath: args[2]))
