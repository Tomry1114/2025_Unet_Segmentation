import cv2
import numpy as np

# 读取标签mask
mask = cv2.imread('/mnt/data1_hdd/wgk/tr-naturecommucation/unet-pytorch-region/VOCdevkit/VOC2007/SegmentationClass_2/000032_frame002.png', cv2.IMREAD_GRAYSCALE)

# 定义调色表：你可以自己改
# 这里举例 5 类（0~4）
colors = [
    (0, 0, 0),       # 0 - black
    (255, 0, 0),     # 1 - blue
    (0, 255, 0),     # 2 - green
    (0, 0, 255),     # 3 - red
    (255, 255, 0),   # 4 - cyan
]

# 创建彩色可视化图
vis = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

for idx, color in enumerate(colors):
    vis[mask == idx] = color

# 保存结果
cv2.imwrite('mask_visualization.png', vis)

print("Visualization saved as mask_visualization.png")
