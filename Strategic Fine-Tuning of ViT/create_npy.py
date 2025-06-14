import numpy as np

# 假设您有一个字典 idx_to_labels
idx_to_labels = {0: 'AD', 1: 'CN', 2: 'MCI'}

# 保存字典为 .npy 文件
npy_file_path = 'idx_to_labels.npy'  # 要保存的 .npy 文件路径
np.save(npy_file_path, idx_to_labels)

# 加载 .npy 文件并确保其内容是一个字典
loaded_dict = np.load(npy_file_path, allow_pickle=True).item()

# 打印加载的字典
print(loaded_dict)

# { "0": "AD", "1": "CN","2": "MCI"}