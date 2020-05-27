import torch,sys
from util import util
from models import pretrained_networks as pn
from models import dist_model as dm
from IPython import embed
import os
import numpy as np

class LIPIS():
    def __init__(self):
        mode = 'net-lin'  # ['net', 'net-lin'] 'net-lin' for Linearly calibrated models, 'net' for Off-the-shelf uncalibrated networks
        net = 'alex'   # ['squeeze', 'alex', 'vgg']
        use_gpu = True # if cuda.is_avaliable() else False
        mode_low = 'l2'  # ['l2', 'ssim']
        colorspace = 'RGB' # ['Lab', 'RGB']

        ## modify configuration
        self.folder_root = '../Results'
        self.dataset = 'Set5'
        self.methods = {'FASRGAN', 'Fs-SRGAN', 'FA+Fs-SRGAN'}

        self.model = dm.DistModel()
        self.model.initialize(model=mode, net=net, use_gpu=use_gpu, colorspace= colorspace)
        


    def cal_lipis(self):
        for method in self.methods:
            img_path_GT = os.path.join(self.folder_root, 'HR', self.dataset)
            img_path_SR = os.path.join(self.folder_root, 'SR', self.dataset, method)
            lipis_List = []

            img_paths_GT = os.listdir(img_path_GT)
            img_paths_SR = os.listdir(img_path_SR)
            for path_GT, path_SR in zip(img_paths_GT, img_paths_SR):
                assert path_GT == path_SR, "Images with different name"
                img_GT = util.im2tensor(util.load_image(os.path.join(img_path_GT, path_GT)))
                img_SR = util.im2tensor(util.load_image(os.path.join(img_path_SR, path_SR)))
                name = path_SR[:-4]

                lipis = self.model.forward(img_GT, img_SR)[0]
                print("Image {} LIPIS result: {}".format(name, lipis))
                lipis_List.append(lipis)
            aver_lipis = np.mean(np.asarray(lipis_List))
            print("Average LIPIS result of {} in {} dataset: {}".format(method, self.dataset, aver_lipis))
            print("End")

if __name__ == "__main__":
    Mylipis = LIPIS()
    Mylipis.cal_lipis()