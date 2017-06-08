import cv2,os
from util.AttrData import AttrDataSet as DataSet
from AttrNet import AttrNet
from easydict import EasyDict as edict
import mxnet as mx
import numpy as np

def param_parse(filename):
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    return yaml_cfg

def train_boxnet():
	params = param_parse("cfgs/AttrNet.yml")
	trainds = DataSet(params.datalist_file,params.path_base, (3,224,224), params.batch_size)
	testds = DataSet(params.test_datalist_file,params.path_base, (3,224,224), params.batch_size)
	net = AttrNet(params)
	net.set_dataset(trainds,testds)		
	#net.loadfromnpy("ckpt_npy/attrnet.npy",sess)
	net.load("ckpt/attr",10)
	#net.load("/home/arcthing/qiaohan/mx_pretrained/vgg/vgg16",0)
	net.train()
if __name__=='__main__':
	train_boxnet()
	#test_boxnet()
