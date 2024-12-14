#include <string>
#include <NvInfer.h>

class TensorRTModel {
public:
    TensorRTModel();
    ~TensorRTModel();

    // ONNX 파일을 불러와서 TensorRT 엔진 파일로 변환
    bool loadEngine(const std::string& onnxModelPath, const std::string& engineFilePath);

private:
    nvinfer1::ILogger* gLogger;         // TensorRT 로깅 객체
    nvinfer1::IBuilder* builder;        // TensorRT 빌더
    nvinfer1::INetworkDefinition* network; // 네트워크 정의
    nvinfer1::ICudaEngine* engine;     // CUDA 엔진
};


