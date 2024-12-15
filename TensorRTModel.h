#include <string>
#include <NvOnnxParser.h>
#include <NvInfer.h>
#include <memory>

using namespace std;

class TensorRTModel {
public:
    TensorRTModel(string onnxPath, string trtPath);
    ~TensorRTModel();
    
    bool CreateEngine(const std::string& onnxModelPath, const std::string& engineFilePath);
    bool LoadEngine(const std::string& engineFilePath);

private:    
    std::unique_ptr<nvinfer1::IBuilder> builder;
    std::unique_ptr<nvinfer1::INetworkDefinition> network;
    std::unique_ptr<nvinfer1::IBuilderConfig> config;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
};


