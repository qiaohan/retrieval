import cv2
import os,lmdb,shutil
from util.dataset import DataSet
from RankNet import RankNet
from easydict import EasyDict as edict
import mxnet as mx
import numpy as np
from random import choice

def param_parse(filename):
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))
    return yaml_cfg

def createdb():
	params = param_parse("cfgs/RankNet.yml")
	params.batch_size = 1
	params.test = True
	#trainds = DataSet(params.datalist_file,params.path_base, (3,224,224), params.batch_size)
	net = RankNet(params)
	#net.set_dataset(trainds,testds)		
	#net.loadfromnpy("ckpt_npy/attrnet.npy",sess)
	net.load("ckpt/ltr",10)
	#net.load("/home/arcthing/qiaohan/mx_pretrained/vgg/vgg16",0)
	#net.train()
	basepath = "benckmark/"
	env = lmdb.open("ltr_fts", map_size=int(1e12))
	with env.begin(write=True) as txn:
		for f in open('testfile.txt'):
			f = f.strip()
			print f
			im = cv2.imread(f)
			ft = net.predict(im).reshape([-1,])
			txn.put(f, ft.tostring())
def test_retrieval():
	params = param_parse("cfgs/RankNet.yml")
	params.batch_size = 1
	params.test = True
	#trainds = DataSet(params.datalist_file,params.path_base, (3,224,224), params.batch_size)
	net = RankNet(params)
	#net.set_dataset(trainds,testds)		
	#net.loadfromnpy("ckpt_npy/attrnet.npy",sess)
	net.load("ckpt/ltr",10)
	#net.load("/home/arcthing/qiaohan/mx_pretrained/vgg/vgg16",0)
	#net.train()
	#retrieval(net,"benckmark/10/TB2Wr.pqRNkpuFjy0FaXXbRCVXa_!!0-rate.jpg_400x400.jpg")
	imgs = {}
	for line in open("testfile.txt"):
		line = line.strip()
		ls = line.split('/')
		if int(ls[1])-1 in imgs.keys():
			imgs[int(ls[1])-1].append(line)
		else:
			imgs[int(ls[1])-1] = [line]
			
	for i in range(10):
		#print imgs[i]
		im = choice(imgs[i])
		#print im
		retrieval(net,im)

def retrieval(net,im_file):
	print im_file
	im = cv2.imread(im_file)
	feat = net.predict(im).reshape([-1,])

	env = lmdb.open("ltr_fts", map_size=int(1e12))
	items = []
	with env.begin(write=True) as txn:
		cursor = txn.cursor()
		for key, value in cursor:
			db_feat = np.fromstring(value, dtype=np.float32)
			item = edict()
			item.dist = np.sum( np.power(feat-db_feat,2) )
			item.name = key
			items.append(item)
	topk = sorted(items, key=lambda y: y.dist)
	topk = topk[:10]
	topk = [k.name for k in topk]
	prefix = "res/"+im_file.split('/')[1]
	os.makedirs(prefix)
	i=0
	for fname in topk:
		i+=1
		shutil.copy(fname, prefix+'/res_'+str(i)+'_'.join(fname.split('/')))
if __name__=='__main__':
	test_retrieval()
	#createdb()
