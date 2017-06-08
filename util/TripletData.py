import numpy as np
from random import shuffle
from random import choice
import os,cv2
import mxnet as mx

class TripletDataSet(object):
	def __init__(self, lstfile, base, imshape, batchsize):
		infos = [l for l in open(lstfile)]
		self.imgs = {}
		self.imgcls = {}
		self.allimgs = []
		self.bboxes = {}
		shuffle(infos)
		for l in infos:
			ll = l.split()
			img,ctype,stype = ll[:3]
			bbox = [int(k) for k in ll[-4:] ]
			self.allimgs.append(img)
			k,name = os.path.split(img)
			if k in self.imgs.keys():
				self.imgs[k].append(img)
			else:
				self.imgs[k] = [img]
			cl,_ = os.path.split(k)
			if cl in self.imgcls.keys():
				self.imgcls[cl].append(img)
			else:
				self.imgcls[cl] = [img]
			self.bboxes[img] = bbox
		self.pathbase = base
		self.imgshape = imshape
		self.itptr = 0
		batchsize /= 3
		self.num_batches = len(self.allimgs)/batchsize
		self.batchsize = batchsize
		
	def reset(self):
		self.itptr = 0
	def getimg(self,imgs):
		theimgs=[]
		for i in range(len(imgs)):
			#print self.pathbase+imgs[i]
			bbox = self.bboxes[imgs[i]]
			im = cv2.imread(self.pathbase+imgs[i])
			imh,imw,_ = im.shape
			bbox[0] = bbox[0] if bbox[0]>0 else 0
			bbox[1] = bbox[1] if bbox[1]>0 else 0
			bbox[2] = bbox[2] if bbox[2]<imw else imw-1
			bbox[3] = bbox[3] if bbox[3]<imh else imh-1
			im = im[bbox[1]:bbox[3],bbox[0]:bbox[2],:]
			imh,imw,_ = im.shape
			newh,neww = self.imgshape[-2:]
			if newh*1.0/neww > imh*1.0/imw:
				imw_resize = neww
				imh_resize = int(imw_resize*imh*1.0/imw)
				scale = imw_resize*1.0/imw
				blank = (newh-imh_resize)
				innerh = [0, newh - blank]
				innerw = [0,neww]
			else:
				imh_resize = newh
				imw_resize = int(imh_resize*imw*1.0/imh)
				scale = imw_resize*1.0/imw
				blank = (neww - imw_resize)
				innerw = [0, neww - blank]
				innerh = [0,newh]
			try:
				im = cv2.resize(im,(imw_resize,imh_resize))
				im = (im.transpose([2,0,1])-128)/255.0
			except Exception as e:
				im = np.zeros(self.imgshape)
				innerw = [0, neww]
				innerh = [0, newh]
			#im = cv2.resize(im,(innerw[1] - innerw[0], innerh[1] - innerh[0]))
			img = np.zeros(self.imgshape)
			#print im.shape
			#print img[:,innerh[0]:innerh[1],innerw[0]:innerw[1]].shape
			img[:,innerh[0]:innerh[1],innerw[0]:innerw[1]]= im
			theimgs.append(img)
		return theimgs
	def next(self):
		#idxs = [ k+self.itptr for k in range(self.batchsize)] 
		anchors = self.allimgs[self.itptr:self.itptr+self.batchsize]
		pos = []
		neg = []
		for a in anchors:
			k,name = os.path.split(a)
			posimg = choice(self.imgs[k])
			pos.append(posimg)

			cl,_ = os.path.split(k)
			nk = k
			negimg = None
			i=0
			while nk==k or i>10:
				i+=i
				negimg = choice(self.imgcls[cl])
				nk,_ = os.path.split(negimg)
			neg.append(negimg)
		self.itptr += self.batchsize
		imgs_ = self.getimg(anchors+pos+neg)
		data = [mx.nd.array(imgs_)]
		#label = [mx.nd.array(labels)]
		return mx.io.DataBatch(data)