#include "pch.h"
#include <NvOnnxParser.h>
#include <NvInfer.h>
#include <iostream>
#include <fstream>
#include <memory>

class TensorRTModel {
public:
    TensorRTModel();
    ~TensorRTModel();

    bool loadEngine(const std::string& onnxModelPath, const std::string& engineFilePath);

private:
    std::unique_ptr<nvinfer1::ILogger> gLogger;
    std::unique_ptr<nvinfer1::IBuilder> builder;
    std::unique_ptr<nvinfer1::INetworkDefinition> network;
    std::unique_ptr<nvinfer1::IBuilderConfig> config;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
};

TensorRTModel::TensorRTModel() {
    gLogger = nullptr;
    builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(*gLogger));
    network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
}

TensorRTModel::~TensorRTModel() {
    
}

bool TensorRTModel::loadEngine(const std::string& onnxModelPath, const std::string& engineFilePath) {   
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, *gLogger));
    if (!parser || !parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR))) {
        std::cerr << "Failed to parse ONNX model file!" << std::endl;
        return false;
    }
        
    if (builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }

    
    engine = std::unique_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config));
    if (!engine) {
        std::cerr << "Failed to build CUDA engine!" << std::endl;
        return false;
    }
    
    auto serializedEngine = std::unique_ptr<nvinfer1::IHostMemory>(engine->serialize());
    std::ofstream engineFile(engineFilePath, std::ios::binary);
    if (!engineFile) {
        std::cerr << "Failed to save engine file!" << std::endl;
        return false;
    }
    engineFile.write(reinterpret_cast<const char*>(serializedEngine->data()), serializedEngine->size());

    return true;
}
