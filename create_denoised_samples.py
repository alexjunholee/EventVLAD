#!/usr/bin/env python

import sys
sys.path.append('./')
import cv2
import numpy as np
import torch
from networks import EventDenoiser
from skimage.measure import compare_psnr, compare_ssim
from skimage import img_as_float, img_as_ubyte, img_as_float32
from pathlib import Path
from utils import peaks, sincos_kernel, generate_gauss_kernel_mix, load_state_dict_cpu
from matplotlib import pyplot as plt
import time
import png

use_gpu = True
case = 3
dep_U = 4
testimg = '124.065664.png'
testseq = 'base'
#testseq = 'tran'

# load the pretrained model
print('Loading the Model')
checkpoint = torch.load('./EventDenoiser_210227')

imdir = '/home/jhlee/data/carla/'
net = EventDenoiser(3, slope=0.2, dep_U=5, dep_S=5)
if use_gpu:
    net = torch.nn.DataParallel(net).cuda()
    net.load_state_dict(checkpoint)
else:
    load_state_dict_cpu(net, checkpoint)
net.eval()

im_gt_path = str(imdir+'img/'+testimg)
if testseq == 'base':
    im_gt_raw_path = str(imdir+'raw/'+testimg)
else : 
    im_gt_raw_path = str(imdir+'raw_night/'+testimg)
    
im_name = im_gt_path.split('/')[-1]
im_gt = cv2.imread(im_gt_path,0)
im_gt_raw = cv2.imread(im_gt_raw_path,0)

H, W = im_gt.shape
if H % 2**dep_U != 0:
    H -= H % 2**dep_U
if W % 2**dep_U != 0:
    W -= W % 2**dep_U
im_gt = im_gt[:H, :W]

# Generate the sigma map
if case == 1:
    # Test case 1
    sigma = peaks(256)
elif case == 2:
    # Test case 2
    sigma = sincos_kernel()
elif case == 3:
    # Test case 3
    sigma = generate_gauss_kernel_mix(32, 32)
else:
    sys.exit('Please input the corrected test case: 1, 2 or 3')

im_n_path_0 = str(imdir+testseq+'/0/'+testimg)
im_n_path_1 = str(imdir+testseq+'/1/'+testimg)
im_n_path_2 = str(imdir+testseq+'/2/'+testimg)
im_0 = cv2.imread(im_n_path_0,0)
im_1 = cv2.imread(im_n_path_1,0)
im_2 = cv2.imread(im_n_path_2,0)

im_gt = cv2.resize(im_gt, (256, 256))
im_gt_raw = cv2.resize(im_gt_raw, (256, 256))
im_0 = cv2.resize(im_0, (256, 256))
im_1 = cv2.resize(im_1, (256, 256))
im_2 = cv2.resize(im_2, (256, 256))

im_0 = img_as_float32(im_0[:,:,np.newaxis])
im_1 = img_as_float32(im_1[:,:,np.newaxis])
im_2 = img_as_float32(im_2[:,:,np.newaxis])

im_gt = img_as_float32(im_gt[:,:,np.newaxis])
im_gt = torch.from_numpy(im_gt.transpose((2,0,1)))[np.newaxis,]

im_gt_raw = img_as_float32(im_gt_raw[:,:,np.newaxis])
im_gt_raw = torch.from_numpy(im_gt_raw.transpose((2,0,1)))[np.newaxis,]

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

im_denoise_mask = outimg[:,0,] * outimg[:,1,]
im_denoise_mask = np.transpose(im_denoise_mask.squeeze(), (0,1))
im_denoise_mask = cv2.rotate(img_as_ubyte(im_denoise_mask.clip(0,1)),cv2.ROTATE_180)

im_noisy = np.transpose(im_noisy.squeeze(), (0,1))
im_noisy = cv2.rotate(img_as_ubyte(im_noisy.clip(0,1)),cv2.ROTATE_180)
im_gt = img_as_ubyte(im_gt.squeeze())
im_gt_raw = img_as_ubyte(im_gt_raw.squeeze())
psnr_val = compare_psnr(im_gt, im_denoise, data_range=255)
ssim_val = compare_ssim(im_gt, im_denoise, data_range=255, multichannel=False)

filename_raw = '/home/jhlee/Desktop' + "/raw_" + testimg
#filename_rae = '/home/jhlee/Desktop' + "/rawedge_" + testimg
filename_evt = '/home/jhlee/Desktop' + "/evt_" + testimg
filename_rec = '/home/jhlee/Desktop' + "/edge_" + testimg
filename_rem = '/home/jhlee/Desktop' + "/mask_" + testimg
png.from_array(im_gt_raw.astype(np.int8), mode="L").save(filename_raw)
#png.from_array(im_gt.astype(np.int8), mode="L").save(filename_rae)
png.from_array(im_noisy.astype(np.int8), mode="L").save(filename_evt)
png.from_array(im_denoise.astype(np.int8), mode="L").save(filename_rec)
png.from_array(im_denoise_mask.astype(np.int8), mode="L").save(filename_rem)

print('Image name: {:s}, PSNR={:5.2f}, SSIM={:7.4f}, time={:.4f}'.format(im_name, psnr_val,
                                                                                 ssim_val, toc-tic))
plt.subplot(141)
plt.imshow(im_gt)
plt.title('Groundtruth')
plt.subplot(142)
plt.imshow(im_noisy)
plt.title('Noisy Image')
plt.subplot(143)
plt.imshow(im_denoise)
plt.title('Denoised Image')
plt.subplot(144)
plt.imshow(im_denoise_mask)
plt.title('Masked Denoised')
plt.show()

