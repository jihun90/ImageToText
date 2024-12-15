#include "pch.h"
#include "TensorRTModel.h"
#include "ImageToText.h"


namespace ImageToText {

	string onnxPath = "D:\\space\\TestApp\\ocr_model.onnx";
	string trtPath = "D:\\space\\TestApp\\ocr_model.trt";

	TensorRTHandle CreateEngine()
	{
		return new TensorRTModel(onnxPath, trtPath);
	};

	bool LoadEngine(TensorRTHandle handle)
	{
		TensorRTModel* engineInstance = static_cast<TensorRTModel*>(handle);
		engineInstance->LoadEngine(trtPath);

		return true;
	}

	bool Run(TensorRTHandle handle, std::string& imagePath, int imageHeight, int imageWidth)
	{
		TensorRTModel* engineInstance = static_cast<TensorRTModel*>(handle);			
		/*engineInstance->infer(engineInstance->PreprocessImage(imagePath, imageHeight, imageWidth), imageHeight, imageWidth);*/
		engineInstance->infer(engineInstance->PreprocessImage(imagePath, 256, 256), 256, 256);

		return true;
	}
}
