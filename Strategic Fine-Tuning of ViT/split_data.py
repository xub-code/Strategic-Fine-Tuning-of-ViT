import os
from shutil import copy, rmtree
import random


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹再重新创建
        rmtree(file_path)
    os.makedirs(file_path)


def main():
    # 保证随机可复现
    random.seed(0)

    # 将数据集按照 8：1:1 划分为训练集、验证集和测试集
    train_rate = 0.7
    val_rate = 0.2
    test_rate = 0.1

    # 指向你解压后的flower_photos文件夹
    cwd = os.getcwd()
    data_root = os.path.join(cwd, "flower_data")
    origin_flower_path = os.path.join(data_root, "flower_photos")
    assert os.path.exists(origin_flower_path), "path '{}' does not exist.".format(origin_flower_path)

    flower_class = [cla for cla in os.listdir(origin_flower_path)
                    if os.path.isdir(os.path.join(origin_flower_path, cla))]

    # 建立保存训练集、验证集和测试集的文件夹
    train_root = os.path.join(data_root, "train")
    val_root = os.path.join(data_root, "val")
    test_root = os.path.join(data_root, "test")

    mk_file(train_root)
    mk_file(val_root)
    mk_file(test_root)

    for cla in flower_class:
        # 为每个类别建立对应的文件夹
        mk_file(os.path.join(train_root, cla))
        mk_file(os.path.join(val_root, cla))
        mk_file(os.path.join(test_root, cla))

    for cla in flower_class:
        cla_path = os.path.join(origin_flower_path, cla)
        images = os.listdir(cla_path)
        num = len(images)

        # 打乱数据
        random.shuffle(images)

        # 划分数据集
        num_train = int(num * train_rate)
        num_val = int(num * val_rate)
        num_test = num - num_train - num_val  # 剩余的是测试集

        # 分割数据
        train_images = images[:num_train]
        val_images = images[num_train:num_train + num_val]
        test_images = images[num_train + num_val:]

        # 将数据分别复制到对应的文件夹
        for image in train_images:
            image_path = os.path.join(cla_path, image)
            new_path = os.path.join(train_root, cla, image)
            copy(image_path, new_path)

        for image in val_images:
            image_path = os.path.join(cla_path, image)
            new_path = os.path.join(val_root, cla, image)
            copy(image_path, new_path)

        for image in test_images:
            image_path = os.path.join(cla_path, image)
            new_path = os.path.join(test_root, cla, image)
            copy(image_path, new_path)

        print(f"[{cla}] processed: train({len(train_images)}), val({len(val_images)}), test({len(test_images)})")

    print("Processing done!")


if __name__ == '__main__':
    main()
