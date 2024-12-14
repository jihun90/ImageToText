#include <string>
#include <NvInfer.h>

class TensorRTModel {
public:
    TensorRTModel();
    ~TensorRTModel();

    // ONNX ������ �ҷ��ͼ� TensorRT ���� ���Ϸ� ��ȯ
    bool loadEngine(const std::string& onnxModelPath, const std::string& engineFilePath);

private:
    nvinfer1::ILogger* gLogger;         // TensorRT �α� ��ü
    nvinfer1::IBuilder* builder;        // TensorRT ����
    nvinfer1::INetworkDefinition* network; // ��Ʈ��ũ ����
    nvinfer1::ICudaEngine* engine;     // CUDA ����
};


