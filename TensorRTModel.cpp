#include "pch.h"
#include "TensorRTModel.h"
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace std;

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Ignore INFO messages
        if (severity != Severity::kERROR && severity != Severity::kINTERNAL_ERROR) return;
        std::cerr << "TensorRT: " << msg << std::endl;
    }
};

Logger gLogger;
TensorRTModel::TensorRTModel(string onnxPath, string trtPath) {
    builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());

    bool existTrtPath = std::filesystem::exists(trtPath);
    if (!existTrtPath)
    {
        bool isSucess = CreateEngine(onnxPath, trtPath);        
        if (isSucess)
        {
            //load
        }
    }
    else
    {
        //load
    }    
}

TensorRTModel::~TensorRTModel() {

}

bool TensorRTModel::CreateEngine(const std::string& onnxModelPath, const std::string& engineFilePath) {   
    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser || !parser->parseFromFile(onnxModelPath.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kERROR))) {
        std::cerr << "Failed to parse ONNX model file!" << std::endl;
        return false;
    }
    
    
    nvinfer1::IOptimizationProfile* profile = builder->createOptimizationProfile();    
    nvinfer1::ITensor* inputTensor = network->getInput(0);
    
    profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMIN, nvinfer1::Dims4{ 1, 3, 224, 224 }); 
    profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kOPT, nvinfer1::Dims4{ 8, 3, 224, 224 }); 
    profile->setDimensions(inputTensor->getName(), nvinfer1::OptProfileSelector::kMAX, nvinfer1::Dims4{ 32, 3, 224, 224 });

    /*string temp = inputTensor->getName();*/
    config->addOptimizationProfile(profile);     
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1 << 30);  


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

bool TensorRTModel::LoadEngine(const std::string& engineFilePath) {

    return true;
}

