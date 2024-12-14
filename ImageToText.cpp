#include "pch.h"
#include "TensorRTModel.h"
#include "ImageToText.h"


namespace ImageToText {

	string onnxPath = "";
	string trtPath = "";

	TensorRTHandle CreateEngine()
	{
		return new TensorRTModel(onnxPath, trtPath);
	};
}
