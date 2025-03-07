import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_ubyte
from skimage.util import random_noise

# 定义能量函数


def energy_function(image, denoised_image, beta):
    # 将图像转换为浮点数类型
    image = image.astype(np.float64)
    denoised_image = denoised_image.astype(np.float64)
    # 数据项
    data_term = np.sum((image - denoised_image) ** 2)
    # 平滑项
    rows, cols = denoised_image.shape
    smooth_term = 0
    for i in range(rows):
        for j in range(cols):
            if i > 0:
                smooth_term += (denoised_image[i, j] - denoised_image[i - 1, j]) ** 2
            if j > 0:
                smooth_term += (denoised_image[i, j] - denoised_image[i, j - 1]) ** 2
    energy = data_term + beta * smooth_term
    return energy

# 模拟退火算法进行图像去噪


def simulated_annealing(image, beta, initial_temperature=100, cooling_rate=0.95, num_iterations=100):
    denoised_image = image.copy()
    temperature = initial_temperature

    for _ in range(num_iterations):
        # 随机选择一个像素点
        row = np.random.randint(0, image.shape[0])
        col = np.random.randint(0, image.shape[1])

        # 保存当前像素值
        old_value = denoised_image[row, col]

        # 随机改变像素值
        new_value = old_value + np.random.randint(-10, 11)
        new_value = np.clip(new_value, 0, 255)
        denoised_image[row, col] = new_value

        # 计算能量变化
        old_energy = energy_function(image, denoised_image, beta)
        denoised_image[row, col] = old_value
        new_energy = energy_function(image, denoised_image, beta)
        delta_energy = new_energy - old_energy

        # 判断是否接受新状态
        if delta_energy < 0 or np.random.rand() < np.exp(-delta_energy / temperature):
            denoised_image[row, col] = new_value

        # 降温
        temperature *= cooling_rate

    return denoised_image


# 加载图像
image = img_as_ubyte(data.camera())

# 添加噪声
noisy_image = random_noise(image, mode='gaussian', var=0.01)
noisy_image = img_as_ubyte(noisy_image)

# 图像去噪
beta = 0.1
denoised_image = simulated_annealing(noisy_image, beta)

# 显示结果
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[1].imshow(noisy_image, cmap='gray')
axes[1].set_title('Noisy Image')
axes[2].imshow(denoised_image, cmap='gray')
axes[2].set_title('Denoised Image')

for ax in axes:
    ax.axis('off')

plt.show()
