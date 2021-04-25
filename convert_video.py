#convert_video.py

import os
import sys
import glob
import argparse
import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
import cv2 as cv
import torch
from torchvision import transforms, datasets
import networks


def convert_video(image):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model_path = ('models/mono_640x192')
    #print("-> Loading model from ", model_path)
    encoder_path = os.path.join(model_path, "encoder.pth")
    depth_decoder_path = os.path.join(model_path, "depth.pth")

    # LOADING PRETRAINED MODEL
    #print("   Loading pretrained encoder")
    encoder = networks.ResnetEncoder(18, False)
    loaded_dict_enc = torch.load(encoder_path, map_location=device)

    # extract the height and width of image that this model was trained with
    feed_height = loaded_dict_enc['height']
    feed_width = loaded_dict_enc['width']
    filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
    encoder.load_state_dict(filtered_dict_enc)
    encoder.to(device)
    encoder.eval()

    #print("   Loading pretrained decoder")
    depth_decoder = networks.DepthDecoder(
        num_ch_enc=encoder.num_ch_enc, scales=range(4))

    loaded_dict = torch.load(depth_decoder_path, map_location=device)
    depth_decoder.load_state_dict(loaded_dict)

    depth_decoder.to(device)
    depth_decoder.eval()

    with torch.no_grad():
        converted = cv.cvtColor(image,cv.COLOR_BGR2RGB)
        input_image = pil.fromarray(converted)
        #input_image = pil.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # Load image and preprocess
        #input_image = pil.open('assets/test_image.jpg').convert('RGB')
        original_width, original_height = input_image.size
        resized = input_image.resize((feed_width, feed_height), pil.LANCZOS)
        transform = transforms.ToTensor()(resized).unsqueeze(0)
        features = encoder(transform)
        outputs = depth_decoder(features)
        disp = outputs[("disp", 0)]
        disp_resized = torch.nn.functional.interpolate(
            disp, (original_height, original_width), mode="bilinear", align_corners=False)
        # Saving colormapped depth image
        disp_resized_np = disp_resized.squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)
        img = cv.cvtColor(np.array(im), cv.COLOR_RGB2BGR)
        #out.write(img)
        #cv2.imshow ('image' , img)
        #if cv.waitKey(1) & 0xFF == ord('q'):
        #break
        #cap.release()
        #out.release()
        return img

#image = cv.imread('image.jpg')
#img = convert_video(image)
#cv.imshow('img',img)
