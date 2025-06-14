import onnxruntime as ort
import numpy as np

# 加载 ONNX 模型
session = ort.InferenceSession("A.onnx")

# 构造输入张量
input_tensor = np.random.rand(1, 3, 224, 224).astype(np.float32)

# 推理
outputs = session.run(None, {'input': input_tensor})
print("ONNX inference output:", outputs)
