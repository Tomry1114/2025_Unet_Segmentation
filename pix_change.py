import os
from PIL import Image
import numpy as np

# 你的标签文件夹
mask_folder = "/mnt/data1_hdd/wgk/tr-naturecommucation/unet-pytorch-region/VOCdevkit/VOC2007/SegmentationClass"

for filename in os.listdir(mask_folder):
    if filename.lower().endswith(".png"):
        path = os.path.join(mask_folder, filename)
        
        # 读取标签
        img = np.array(Image.open(path))
        
        # 替换
        img_converted = np.zeros_like(img)
        img_converted[img == 1] = 0
        img_converted[img == 2] = 1
        
        # 保存覆盖
        Image.fromarray(img_converted.astype(np.uint8)).save(path)
        
        print(f"Converted and saved: {path}")

print("✅ All masks converted and overwritten!")
