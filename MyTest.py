import torch
import cv2
import torch.nn.functional as F
import numpy as np
import os, argparse
from scipy import misc
import imageio
import shutil
from lib.EAMNet import Network
from utils.sdy_data_val import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--pth_path', type=str, default='./snapshot/EAMNet/Net_epoch_best.pth')
opt = parser.parse_args()
 
for _data_name in ['CAMO', 'COD10K','NC4K']:
#for _data_name in ['CAMO', 'COD10K', 'CHAMELEON','NC4K']:
    data_path = './Dataset/TestDataset/{}/'.format(_data_name)

    save_path = './res/{}/{}/preds/'.format(opt.pth_path.split('/')[-2], _data_name)
    #tex_save_path = './res/{}/{}/tex/'.format(opt.pth_path.split('/')[-2], _data_name)
    edge_save_path = './res/{}/{}/edge/'.format(opt.pth_path.split('/')[-2], _data_name)
    sen2_save_path = './res/{}/{}/sen_22/'.format(opt.pth_path.split('/')[-2], _data_name)



    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = Network()
    model.load_state_dict(torch.load(opt.pth_path))
    model.cuda()
    model.eval()
 
    os.makedirs(save_path, exist_ok=True)
    #os.makedirs(tex_save_path, exist_ok=True)
    os.makedirs(edge_save_path, exist_ok=True)
    os.makedirs(sen2_save_path, exist_ok=True)


    image_root = '{}/Imgs/'.format(data_path)
    gt_root = '{}/GT/'.format(data_path)
    test_loader = test_dataset(image_root, gt_root, opt.testsize)



    for i in range(test_loader.size):
        image, gt, name, _ = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()

        out_1, out_final ,out_3,edge,edge2,edg_final=  model(image)

        res = out_final
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        #print('> {} - {}'.format(_data_name, name))
        res = 255 * res
        res = res.astype(np.uint8)
        cv2.imwrite(save_path + name, res)

        res = edg_final
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        #print('> {} - {}'.format(_data_name, name))
        res = 255 * res
        res = res.astype(np.uint8)
        cv2.imwrite(edge_save_path + name, res)


    print('{} Finish!'.format(_data_name))
