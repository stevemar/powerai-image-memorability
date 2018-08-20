import Foundation
import Kitura
import KituraMustache
import CoreML
import Vision
import CoreImage
import Quartz
import QuartzCore

if #available(OSX 10.13, *) {
    let currentPath = FileManager.default.currentDirectoryPath
    print("Working in directory: \(currentPath)")
    
    var model: MLModel!
    model = try! MLModel(contentsOf: MLModel.compileModel(at: URL(fileURLWithPath: "\(currentPath)/lamem.mlmodel")))

    let visionModel = try! VNCoreMLModel(for: model)
    
    let router = Router()
    
    router.add(templateEngine: MustacheTemplateEngine())
    router.all("/", middleware: StaticFileServer())
    router.post(middleware: BodyParser())
    
    func saveFileFrom(body: ParsedBody) -> (Bool, String) {
        if let multiPart = body.asMultiPart?[0] {
            let filename = multiPart.filename

            let url = URL(fileURLWithPath: currentPath + "/public/" + filename)

            let data = multiPart.body.asRaw
            if let data = data {
                do {
                    try data.write(to: url)
                    return (true, filename)
                } catch {
                    return (false, filename)
                }
            }
        }
        return (false, "")
    }
    
    func imageFrom(body: ParsedBody) -> CIImage? {
        if let multiPart = body.asMultiPart?[0] {
            let data = multiPart.body.asRaw
            if let data = data {
                return CIImage(data: data)
            }
        }
        return nil
    }
    
    func visionRequest(completionHandler: @escaping (_ output: Double) -> Void) -> VNCoreMLRequest {
        let request = VNCoreMLRequest(model: visionModel, completionHandler: { (request, err) in
            let results = request.results as! [VNCoreMLFeatureValueObservation]
            completionHandler(Double(truncating: results[0].featureValue.multiArrayValue![0]))
        })
        request.imageCropAndScaleOption = .scaleFit
        return request
    }
    
    router.get("/") { request, response, next in
        try! response.render("UploadVideoView.mustache", context: [:])
        response.status(.OK)
        next()
    }
    
    router.post("/filereceived") { request, response, next in
        if let body = request.body {
            let fileSaved = saveFileFrom(body: body)
            let request = visionRequest(completionHandler: { (output) in
                try! response.render("MemorabilityView.mustache", context: ["img_memorability": Int(output * 100), "user_img": fileSaved.1])
                response.status(.OK)
                next()
            })
            let image = imageFrom(body: body)
            let handler = VNImageRequestHandler(ciImage: image!)
            try! handler.perform([request])
        } else {
            response.send("Sorry, you didn't upload a video file.")
            next()
        }
    }
    
    Kitura.addHTTPServer(onPort: 3333, with: router)
    Kitura.run()
} else {
    print("You must have macOS 10.13 (High Sierra) or higher to run this application.")
    exit(0)
}
