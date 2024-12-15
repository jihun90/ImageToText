namespace ImageToText {
	typedef void* TensorRTHandle;
	
	extern "C" __declspec(dllexport) TensorRTHandle CreateEngine();
	extern "C" __declspec(dllexport) bool LoadEngine(TensorRTHandle handle);
	extern "C" __declspec(dllexport) bool Run(TensorRTHandle handle, std::string& imagePath, int imageHeight, int imageWidth);
}


