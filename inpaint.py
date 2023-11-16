import cv2
import os
import importlib
import numpy as np
from glob import glob 

import torch
from torchvision.transforms import ToTensor

from utils.option import args

# python demo.py --dir_image /Users/hwangseho/Downloads/AOT-GAN-for-Inpainting-master/examples/logos/image --pre_train /Users/hwangseho/Downloads/AOT-GAN-for-Inpainting-master/G0000000.pt --painter bbox

logo_img='/Users/hwangseho/Downloads/AOT-GAN-for-Inpainting-master/examples/logos/image/252027220.jpg'
logo_mask='/Users/hwangseho/Downloads/AOT-GAN-for-Inpainting-master/examples/logos/mask/252027220.png'
logo_mask=cv2.imread(logo_mask,cv2.IMREAD_GRAYSCALE)
print(logo_mask)

def postprocess(image):
    image = torch.clamp(image, -1., 1.)
    image = (image + 1) / 2.0 * 255.0
    image = image.permute(1, 2, 0)
    image = image.cpu().numpy().astype(np.uint8)
    return image

def demo():
    # Model and version
    net = importlib.import_module('model.'+args.model)
    model = net.InpaintGenerator(args)
    model.load_state_dict(torch.load('/Users/hwangseho/Documents/GitHub/pyqt/G0000000.pt', map_location='cpu'))
    model.eval()

    filename = logo_img
    orig_img = cv2.resize(cv2.imread(filename, cv2.IMREAD_COLOR), (512, 512))
    img_tensor = (ToTensor()(orig_img) * 2.0 - 1.0).unsqueeze(0)
    mask = logo_mask

    # print('[**] inpainting ... ')
    with torch.no_grad():
        mask_tensor = (ToTensor()(mask)).unsqueeze(0)
        masked_tensor = (img_tensor * (1 - mask_tensor).float()) + mask_tensor
        pred_tensor = model(masked_tensor, mask_tensor)
        comp_tensor = (pred_tensor * mask_tensor + img_tensor * (1 - mask_tensor))

        comp_np = postprocess(comp_tensor[0])

        cv2.imwrite('/Users/hwangseho/Documents/GitHub/pyqt/images/res/0.jpg', comp_np)


if __name__ == '__main__':
    demo()
