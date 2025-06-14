import torch
from torchvision import models
import onnx
from tqdm import tqdm
from vit_model import vit_base_patch16_224_in21k as create_model


def export_model_to_onnx(model_path, output_onnx_path):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # Load the pre-trained model 方式一
    # model = torch.load(model_path)
    # model = model.eval().to(device)

    # 方式二
    model = create_model(num_classes=3, has_logits=False).to(device)
    model_weight_path = model_path
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    # torch.load(model_weight_path, map_location=device)
    model.eval().to(device)

    # Construct an input image Tensor
    x = torch.rand(1, 3, 224, 224).to(device)
    print(x.shape)

    # Perform inference with a progress bar
    with torch.no_grad():
        pbar = tqdm(total=1, desc='Exporting ONNX')  # 进度条
        torch.onnx.export(
            model,                    # Model to export
            x,                        # Example input
            output_onnx_path,         # Output ONNX file name
            input_names=['input'],    # Input names (can be customized)
            output_names=['output'],  # Output names (can be customized)
            # opset_version=11,         # ONNX operator set version
        )
        pbar.update(1)
        pbar.close()

    # Validate the exported ONNX model
    onnx_model = onnx.load(output_onnx_path)
    # Check if the model format is correct
    onnx.checker.check_model(onnx_model)
    print('无报错，onnx模型导入成功!')

if __name__ == '__main__':
    # 调用函数进行模型导出
    model_path = r'/vision_transformer/weights_\best.pth'
    output_onnx_path = 'A.onnx'
    export_model_to_onnx(model_path, output_onnx_path)
    # 登录netron.app在线模型可视化网页打开可视化模型结构


