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
}
