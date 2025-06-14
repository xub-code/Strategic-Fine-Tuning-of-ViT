import os
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt


def main():
    # ONNX 推理会在 CPU 或 GPU 上运行
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if ort.get_device() == 'GPU' else [
        'CPUExecutionProvider']
    session = ort.InferenceSession("A.onnx", providers=providers)

    # 定义数据预处理
    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.5, 0.5, 0.5],
                              [0.5, 0.5, 0.5])])

    # 加载图片
    img_path = r"ADMCI/test/AD/AD002_S_0816a096.png"
    assert os.path.exists(img_path), f"file: '{img_path}' does not exist."
    img = Image.open(img_path).convert('RGB')

    plt.imshow(img)

    # 预处理图片
    img = data_transform(img)
    img = img.unsqueeze(0).numpy()  # 将图片转为 NumPy 数组并增加 batch 维度

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
