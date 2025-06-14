import torch
from vit_model import vit_base_patch16_224_in21k as create_model
from thop import profile

# 设备设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device for testing.")

num_classes=3
# 创建模型
model = create_model(num_classes=num_classes).to(device)

# 模拟输入数据，维度为（batch_size, channels, height, width）
input_tensor = torch.randn(1, 3, 224, 224).to(device)  # 假设输入是一个224x224的RGB图像

# 计算模型的参数量和FLOPs
macs, params = profile(model, inputs=(input_tensor,))

# 打印结果
print(f"Total number of parameters: {params / 1e6:.3f}M")
print(f"Total number of FLOPs: {macs / 1e9:.3f}G")
