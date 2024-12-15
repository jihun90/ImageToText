#include <string>
#include <NvOnnxParser.h>
#include <NvInfer.h>
#include <memory>
#include <iostream>

using namespace std;

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Ignore INFO messages
        if (severity != Severity::kERROR && severity != Severity::kINTERNAL_ERROR) return;
        std::cerr << "TensorRT: " << msg << std::endl;
    }
};

class TensorRTModel {
public:
    TensorRTModel(string onnxPath, string trtPath);
    ~TensorRTModel();
    
    bool CreateEngine(const std::string& onnxModelPath, const std::string& engineFilePath);
    bool LoadEngine(const std::string& engineFilePath);

private:    
    Logger gLogger;   
    IRuntime* runtime;
    ICudaEngine* engine;
    IExecutionContext* context;
};


