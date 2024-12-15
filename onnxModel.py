import torch
import torchvision.models as models

# 예제 모델 (CRNN 또는 다른 OCR 모델)
model = models.resnet18(pretrained=True)
model.eval()

dummy_input = torch.randn(1, 3, 224, 224)  # 입력 텐서 크기 (채널, 높이, 너비)
torch.onnx.export(
    model,
    dummy_input,
    "ocr_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
