#include "pch.h"
#include "TensorRTModel.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace nvinfer1;
using namespace cv;

TensorRTModel::TensorRTModel(string onnxPath, string trtPath) {        
    this->engine = nullptr;
    this->runtime = nullptr;

    bool existTrtPath = std::filesystem::exists(trtPath);
    if (!existTrtPath)
    {
        bool isSucess = CreateEngine(onnxPath, trtPath);                
    }    
}

TensorRTModel::~TensorRTModel() {        
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

    this->runtime = createInferRuntime(this->gLogger);
    if (!runtime) {
        std::cerr << "Failed to create inference runtime." << std::endl;
        return -1;
    }
    
    this->engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size());
    if (!engine) {
        std::cerr << "Failed to create the engine from the .trt file." << std::endl;
        return -1;
    }

    return true;
}

cv::Mat TensorRTModel::PreprocessImage(const std::string& imagePath, int inputHeight, int inputWidth)
{
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << imagePath << std::endl;
        exit(-1);
    }

    cv::resize(img, img, cv::Size(inputWidth, inputHeight));

    img.convertTo(img, CV_32F, 1.0 / 255.0);


    return img;
}

void TensorRTModel::infer(ICudaEngine* engine, const std::vector<float>& inputImage, int inputHeight, int inputWidth) {    
    IExecutionContext* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context" << std::endl;
        exit(-1);
    }
    
    float* inputData;
    size_t inputSize = inputHeight * inputWidth;
    cudaMalloc((void**)&inputData, inputSize * sizeof(float));    
    cudaMemcpy(inputData, inputImage.data(), inputSize * sizeof(float), cudaMemcpyHostToDevice);
    
    float* outputData;
    size_t outputSize = 1000; 
    cudaMalloc((void**)&outputData, outputSize * sizeof(float));
    
    void* buffers[] = { inputData, outputData };
    
    context->executeV2(buffers);
    
    std::vector<float> output(outputSize);
    cudaMemcpy(output.data(), outputData, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
    
    int maxIndex = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    std::cout << "Predicted label: " << maxIndex << std::endl;
    
    cudaFree(inputData);
    cudaFree(outputData);
    delete context;
}
