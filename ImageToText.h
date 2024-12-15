namespace ImageToText {
	typedef void* TensorRTHandle;
	
	extern "C" __declspec(dllexport) TensorRTHandle CreateEngine();
	extern "C" __declspec(dllexport) bool LoadEngine();
}


