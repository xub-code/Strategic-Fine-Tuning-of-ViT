import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    """自定义预处理函数"""
    # 加载图片并调整大小
    img = Image.open(image_path).convert('RGB')
    img = img.resize((256, 256))  # 调整大小

    # 中心裁剪
    width, height = img.size
    new_width, new_height = 224, 224
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    img = img.crop((left, top, right, bottom))

    # 转换为 NumPy 数组并归一化
    img = np.array(img).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5  # 标准化到 [-1, 1]

    # 转置通道维度: HWC -> CHW
    img = np.transpose(img, (2, 0, 1))

    # 增加 batch 维度
    img = np.expand_dims(img, axis=0)

    return img


def main():
    # ONNX 推理会在 CPU 或 GPU 上运行
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else [
        'CPUExecutionProvider']
    print(providers)
    session = ort.InferenceSession("A.onnx", providers=providers)

    # 加载图片并预处理
    img_path = r"ADMCI/test/AD/AD002_S_0816a096.png"
    assert os.path.exists(img_path), f"file: '{img_path}' does not exist."

    # 显示原始图片
    original_img = Image.open(img_path).convert('RGB')
    plt.imshow(original_img)

    img = preprocess_image(img_path)

    # 加载类别索引
    json_path = './class_indices.json'
    assert os.path.exists(json_path), f"file: '{json_path}' does not exist."

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 推理
    inputs = {session.get_inputs()[0].name: img}
    outputs = session.run(None, inputs)
    predict = outputs[0][0]

    # 计算概率并获取预测结果
    def softmax(x):
        exp_x = np.exp(x - np.max(x))  # 减去最大值以防溢出
        return exp_x / np.sum(exp_x)

    predict_probs = softmax(predict)
    predict_cla = np.argmax(predict_probs)

    print_res = "class: {}   prob: {:.6}".format(class_indict[str(predict_cla)],
                                                 predict_probs[predict_cla])
    plt.title(print_res)
    for i in range(len(predict_probs)):
        print("class: {:10}   prob: {:.6}".format(class_indict[str(i)],
                                                  predict_probs[i]))
    plt.show()


if __name__ == '__main__':
    main()
