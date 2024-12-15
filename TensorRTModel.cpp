#include "pch.h"
#include "TensorRTModel.h"
#include <iostream>
#include <fstream>
#include <filesystem>

using namespace std;
using namespace nvinfer1;

TensorRTModel::TensorRTModel(string onnxPath, string trtPath) {    
    context = nullptr;
    engine = nullptr;
    runtime = nullptr;

    bool existTrtPath = std::filesystem::exists(trtPath);
    if (!existTrtPath)
    {
        bool isSucess = CreateEngine(onnxPath, trtPath);        
        if (isSucess)
        {
            LoadEngine(trtPath);
        }
    }
    else
    {
        LoadEngine(trtPath);
    }    
}

TensorRTModel::~TensorRTModel() {
    
    if (context != nullptr) {
        delete context;
        context = nullptr;
    }

    if (engine != nullptr) {
        delete engine;
        engine = nullptr;
    }

    if (runtime != nullptr) {
        delete runtime;
        runtime = nullptr;
    }

}

bool TensorRTModel::CreateEngine(const std::string& onnxModelPath, const std::string& engineFilePath) 
{   
    std::unique_ptr<nvinfer1::IBuilder> builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    std::unique_ptr<nvinfer1::INetworkDefinition> network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(0));
    std::unique_ptr<nvinfer1::IBuilderConfig> config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    std::unique_ptr<nvinfer1::ICudaEngine> engine;

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

bool TensorRTModel::LoadEngine(const std::string& engineFilePath) 
{
    std::ifstream engineFileStream(engineFilePath, std::ios::binary);
    if (!engineFileStream.good()) {
        std::cerr << "Error opening .trt file." << std::endl;
        return -1;
    }

    std::vector<char> engineData((std::istreambuf_iterator<char>(engineFileStream)),
        std::istreambuf_iterator<char>());
    if (engineData.empty()) {
        std::cerr << "Error reading .trt file." << std::endl;
        return -1;
    }

    this->runtime = createInferRuntime(gLogger);
    if (!runtime) {
        std::cerr << "Failed to create inference runtime." << std::endl;
        return -1;
    }
    
    this->engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine) {
        std::cerr << "Failed to create the engine from the .trt file." << std::endl;
        return -1;
    }

    this->context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context." << std::endl;
        return -1;
    }

    
    delete context;
    delete engine;
    delete runtime;

    return true;
}

