import numpy as np
from random import shuffle
import os,cv2
import mxnet as mx

class AttrDataSet(mx.io.DataIter):
	def __init__(self, lstfile, base, imshape, batchsize):
		self.imgshape = imshape
		infos = [l for l in open(lstfile)]
		self.imgs = []
		self.labels = []
		shuffle(infos)
		for l in infos:
			lls = l.split()
			self.imgs.append(lls[0])
			lab = lls[0].split('/')[-2]
			self.labels.append(lab)
		self.pathbase = base
		self.itptr = 0
		self.num_batches = len(self.imgs)/batchsize
		self.batchsize = batchsize
		self.readlabelmap()
	def readlabelmap(self):
		self.labelmap = {}
		for line in open("datalist/list_attr_items.txt"):
			lls = line.strip().split()
			self.labelmap[lls[0]] = [int((int(i)+1)/2) for i in lls[1:]]
	def reset(self):
		self.itptr = 0
	def __iter__(self):
		return self
	def __next__(self):
		return self.next()

	@property
	def provide_data(self):
		return [('data',(self.batchsize,) + self.imgshape)]

	@property
	def provide_label(self):
		return [('bboxgt',(self.batchsize, 4))]
	def getimg(self,imgs):
		theimgs=[]
		for i in range(len(imgs)):
			#print self.pathbase+imgs[i]
			im = cv2.imread(self.pathbase+imgs[i])
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
			im = cv2.resize(im,(imw_resize,imh_resize))
			#im = cv2.resize(im,(innerw[1] - innerw[0], innerh[1] - innerh[0]))
			im = (im.transpose([2,0,1])-128)/255.0
			img = np.zeros(self.imgshape)
			#print im.shape
			#print img[:,innerh[0]:innerh[1],innerw[0]:innerw[1]].shape
			img[:,innerh[0]:innerh[1],innerw[0]:innerw[1]]= im
			theimgs.append(img)
		return theimgs
	def getimg2(self,imgs,bboxes):
		thebboxes=[]
		theimgs=[]
		for i in range(len(imgs)):
			#print self.pathbase+imgs[i]
			im = cv2.imread(self.pathbase+imgs[i])
			imh,imw,_ = im.shape
			x1 = bboxes[i][0] if bboxes[i][0]>0 else 0
			y1 = bboxes[i][1] if bboxes[i][1]>0 else 0
			x2 = bboxes[i][2] if bboxes[i][2]<imw else imw
			y2 = bboxes[i][3] if bboxes[i][3]<imh else imh
			bbox = [x1,y1,x2,y2]
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
			bbox = map(lambda x: x*scale, bbox)
			#bbox = [bbox[0]/neww, bbox[1]/newh, bbox[2]/neww, bbox[3]/newh]
			im = cv2.resize(im,(imw_resize,imh_resize))
			#im = cv2.resize(im,(innerw[1] - innerw[0], innerh[1] - innerh[0]))
			im = (im.transpose([2,0,1])-128)/255.0
			img = np.zeros(self.imgshape)
			#print im.shape
			#print img[:,innerh[0]:innerh[1],innerw[0]:innerw[1]].shape
			img[:,innerh[0]:innerh[1],innerw[0]:innerw[1]]= im
			theimgs.append(img)
			thebboxes.append(bbox)
		return theimgs,thebboxes
	def next(self):
		if self.itptr + self.batchsize < len(self.imgs):
			imgs = self.imgs[self.itptr:self.itptr+self.batchsize]
			labels = [self.labelmap[k] for k in self.labels[self.itptr:self.itptr+self.batchsize]]
			self.itptr += self.batchsize
			imgs_ = self.getimg(imgs)
			data = [mx.nd.array(imgs_)]
			label = [mx.nd.array(labels)]
			return mx.io.DataBatch(data, label)
		else:
			raise StopIteration