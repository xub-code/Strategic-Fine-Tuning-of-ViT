import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import random

def augment_image(image_path, output_folder, num_augmentations=5):
    """
    对单张图像进行数据增强并保存增强后的图像。
    :param image_path: 输入图像路径
    :param output_folder: 增强图像保存文件夹
    :param num_augmentations: 每张图像生成的增强图像数量
    """
    # 打开图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

    for i in range(num_augmentations):
        # 随机选择增强方法
        enhanced_image = image.copy()

        # 1. 旋转
        angle = random.uniform(-15, 15)  # 旋转角度范围
        (h, w) = enhanced_image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        enhanced_image = cv2.warpAffine(enhanced_image, M, (w, h))

        # 2. 平移
        tx = random.randint(-10, 10)
        ty = random.randint(-10, 10)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        enhanced_image = cv2.warpAffine(enhanced_image, M, (w, h))

        # 3. 缩放
        scale = random.uniform(0.9, 1.1)
        enhanced_image = cv2.resize(enhanced_image, (0, 0), fx=scale, fy=scale)

        # 4. 添加噪声
        noise = np.random.normal(0, 10, enhanced_image.shape).astype(np.uint8)
        enhanced_image = cv2.add(enhanced_image, noise)

        # 5. 对比度调整
        enhanced_image = Image.fromarray(enhanced_image)
        enhancer = ImageEnhance.Contrast(enhanced_image)
        enhanced_image = enhancer.enhance(random.uniform(0.8, 1.2))

        # 6. 水平翻转（可选）
        if random.random() > 0.5:
            enhanced_image = enhanced_image.transpose(Image.FLIP_LEFT_RIGHT)

        # 保存增强后的图像
        filename = os.path.basename(image_path)
        output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}_aug_{i}.png")
        enhanced_image.save(output_path)

def batch_augment_images(input_folder, output_folder, num_augmentations=5):
    """
    批量处理文件夹中的图像并进行数据增强。
    :param input_folder: 输入图像文件夹
    :param output_folder: 增强图像保存文件夹
    :param num_augmentations: 每张图像生成的增强图像数量
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的图像
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            print(f"Processing image: {image_path}")
            augment_image(image_path, output_folder, num_augmentations)

# 示例用法
input_folder = r"D:\pythonProject\vision_transformer_my\ADMCI\train\MCI"  # 替换为你的训练集图像文件夹路径
output_folder = r"D:\pythonProject\vision_transformer_my\ADMCI\train\MCI"  # 替换为你希望保存增强图像的文件夹路径
batch_augment_images(input_folder, output_folder, num_augmentations=5)