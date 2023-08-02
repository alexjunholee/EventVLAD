#!/usr/bin/env python

import os
import cv2
import numpy as np
import torch
from networks import EventDenoiser
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage import img_as_float, img_as_ubyte, img_as_float32
from utils import load_state_dict_cpu
from matplotlib import pyplot as plt
import time
import png

use_gpu = True
dep_U = 4

testpath = 'sample'
test_imgs = ['d0', 'd1', 'd2']
truth_edge = 'd_gt'

# load the pretrained model
print('Loading the Model')
checkpoint = torch.load('./denoiser_carla')

net = EventDenoiser(3, slope=0.2, dep_U=5, dep_S=5)
if use_gpu:
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(checkpoint)
else:
    load_state_dict_cpu(net, checkpoint)
net.eval()

# load images
im_gt_path = os.path.join(testpath, truth_edge+'.png')
im_gt = cv2.imread(im_gt_path,0)

H, W = im_gt.shape
if H % 2**dep_U != 0:
    H -= H % 2**dep_U
if W % 2**dep_U != 0:
    W -= W % 2**dep_U
im_gt = im_gt[:H, :W]
im_gt = cv2.resize(im_gt, (256, 256))
im_gt = img_as_float32(im_gt[:,:,np.newaxis])
im_gt = torch.from_numpy(im_gt.transpose((2,0,1)))[np.newaxis,]

im_0 = cv2.imread(os.path.join(testpath, test_imgs[0]+'.png'), 0)
im_1 = cv2.imread(os.path.join(testpath, test_imgs[1]+'.png'), 0)
im_2 = cv2.imread(os.path.join(testpath, test_imgs[2]+'.png'), 0)

im_0 = cv2.resize(im_0, (256, 256))
im_1 = cv2.resize(im_1, (256, 256))
im_2 = cv2.resize(im_2, (256, 256))

im_0 = img_as_float32(im_0[:,:,np.newaxis])
im_1 = img_as_float32(im_1[:,:,np.newaxis])
im_2 = img_as_float32(im_2[:,:,np.newaxis])

im_0 = cv2.rotate(im_0, cv2.ROTATE_180)[:,:,np.newaxis]
im_1 = cv2.rotate(im_1, cv2.ROTATE_180)[:,:,np.newaxis]
im_2 = cv2.rotate(im_2, cv2.ROTATE_180)[:,:,np.newaxis]

im_0 = torch.from_numpy(im_0.transpose((2,0,1)))
im_1 = torch.from_numpy(im_1.transpose((2,0,1)))
im_2 = torch.from_numpy(im_2.transpose((2,0,1)))

im_noisy = torch.cat((im_0,im_1,im_2),0)[np.newaxis,]
if use_gpu:
    im_noisy = im_noisy.cuda()
    print('Begin Testing on GPU')
else:
    print('Begin Testing on CPU')
with torch.autograd.set_grad_enabled(False):
    torch.cuda.synchronize()
    tic = time.perf_counter()
    img_estim = net(im_noisy)
    torch.cuda.synchronize()
    toc = time.perf_counter()
    outimg = img_estim.cpu().numpy()
if use_gpu:
    im_noisy = im_noisy.cpu().numpy()
else:
    im_noisy = im_noisy.numpy()
#im_noisy = im_noisy[:,1,]
im_noisy = np.mean(im_noisy,1)
im_denoise = outimg[:,0,]
im_denoise = np.transpose(im_denoise.squeeze(), (0,1))
im_denoise = cv2.rotate(img_as_ubyte(im_denoise.clip(0,1)),cv2.ROTATE_180)

im_noisy = np.transpose(im_noisy.squeeze(), (0,1))
im_noisy = cv2.rotate(img_as_ubyte(im_noisy.clip(0,1)),cv2.ROTATE_180)
im_gt = img_as_ubyte(im_gt.squeeze())
psnr_val = peak_signal_noise_ratio(im_gt, im_denoise, data_range=255)
ssim_val = structural_similarity(im_gt, im_denoise, data_range=255, multichannel=False)

print('PSNR={:5.2f}, SSIM={:7.4f}, time={:.4f}'.format(psnr_val, ssim_val, toc-tic))
plt.subplot(131)
plt.imshow(im_gt)
plt.title('Groundtruth')
plt.subplot(132)
plt.imshow(im_noisy)
plt.title('Noisy Image')
plt.subplot(133)
plt.imshow(im_denoise)
plt.title('Denoised Image')
plt.show()

