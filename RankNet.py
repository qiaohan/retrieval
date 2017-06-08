import cv2
import logging
import numpy as np
logging.getLogger().setLevel(logging.INFO)
import mxnet as mx
from util.cnn import CNN
from tqdm import tqdm

class RankNet(CNN):
    def build_loss(self):
        #self.params.test = None
        if self.params.test is None:
            assert self.batch_size%3==0, "Batch Size must divide by 3 for Triplet Loss..."
        conv_feats_reshape = mx.symbol.Flatten(data=self.conv_feats)
        self.feat_embbed = mx.symbol.FullyConnected(data=conv_feats_reshape, num_hidden=4096, name='feat_fc')
        self.feat_embbed = mx.sym.L2Normalization(self.feat_embbed)
        if self.params.test is None:
            self.triplet_margin = self.params.margin
            self.loss = mx.symbol.MakeLoss(self.triplet())
        else:
            self.loss = mx.symbol.MakeLoss(self.feat_embbed)
        self.ctx = mx.cpu() if self.params.gpu<0 else mx.gpu(self.params.gpu)
        #mx.Context.default_ctx = self.ctx
        self.input_names = ['data']
        self.input_types = {'data':'float32'}
        self.input_shapes = {'data':(self.params.batch_size,3,self.params.scale_shape.h,self.params.scale_shape.w)}
    def triplet(self):
        anchor_output, positive_output, negative_output = mx.symbol.split(data=self.feat_embbed, num_outputs=3, axis=0)
        print "triplet loss margin:",self.triplet_margin
        d_pos = mx.symbol.sum((anchor_output - positive_output)**2, axis=1)
        d_neg = mx.symbol.sum((anchor_output - negative_output)**2, axis=1)
        loss = mx.sym.maximum(0., self.triplet_margin + d_pos - d_neg)
        #loss = mx.symbol.sum(loss)
        return loss

    def set_dataset(self,traind,testd):
        self.train_dataset = traind
        self.test_dataset = testd
    
    def train(self):
        print("Training the model...")
        params = self.params
        arg_names = self.loss.list_arguments()
        #aux_names = self.loss.list_auxiliary_states()
        #output_names = self.loss.list_outputs()
        exec_params = self.arg_params
        exec_params['data'] = mx.nd.ones([params.batch_size, 3, params.scale_shape.h, params.scale_shape.w])
        #exec_params['attrgt'] = mx.nd.ones([params.batch_size, 1000])
        #exec_params['labelgt'] = mx.nd.ones([params.batch_size,])
        exec_grads = {}
        for k,v in exec_params.items():
            exec_params[k] = v.copyto(self.ctx)
            exec_grads[k] = v.copyto(self.ctx)
        self.im = exec_params['data']
        #self.attr = exec_params['attrgt']
        #self.label = exec_params['labelgt']
        self.exe = self.loss.bind(ctx = self.ctx, args = exec_params, args_grad = exec_grads, aux_states = self.aux_params)
        self.opt = mx.optimizer.SGD( sym=self.loss, #param_idx2name=idx2name,
                                     learning_rate=self.params.lr, momentum = 0.1)
        args = self.exe.arg_arrays
        grads = self.exe.grad_arrays
        states = []
        for i,argn in enumerate(arg_names):
            if argn in self.input_names:
                states.append(None)
            else:
                states.append(self.opt.create_state(i,args[i]))
        for epoch_no in tqdm(list(range(params.num_epochs)), desc='epoch'): 
            self.train_dataset.reset()
            #data_iter = iter(self.train_dataset)
            #end_of_batch = False
            for idx in range(self.train_dataset.num_batches): 
                data_batch = self.train_dataset.next()
                data_batch.data[0].copyto(self.im)
                #data_batch.label[0].copyto(self.label)
                l = self.exe.forward(is_train=True)
                self.exe.backward()
                for i,argn in enumerate(arg_names):
                    if argn in self.input_names:
                        continue
                    #print args[i], grads[i], states[i]
                    self.opt.update(i, args[i], grads[i], states[i])
                ll = l[0].asnumpy()
                print "iter:",idx,'zero loss number:',np.where(ll==0.0)[0].shape[0],'mean loss:',ll[np.where(ll!=0.0)].mean()
                print ll
                #if (idx+1) % params.test_period == 0:
                #   self.test(sess)
                params.save_period = self.train_dataset.num_batches/5
                if (idx+1) % params.save_period == 0:
                    self.save()
        print("Model trained.")
    
    def predict(self,cv_im):
        im = cv_im.copy()
        imh,imw,c = im.shape
        newh = self.params.scale_shape.h
        neww = self.params.scale_shape.w
        if newh*1.0/neww > imh*1.0/imw:
            imw_resize = neww
            imh_resize = int(imw_resize*imh*1.0/imw)
            blank = (newh-imh_resize)
            innerh = [0, newh - blank]
            innerw = [0,neww]
            scale = imw*1.0/imw_resize
        else:
            imh_resize = newh
            imw_resize = int(imh_resize*imw*1.0/imh)
            blank = (neww - imw_resize)
            innerw = [0, neww - blank]
            innerh = [0,newh]
            scale = imw*1.0/imw_resize
        im = cv2.resize(im,(imw_resize,imh_resize))
        im = (im.transpose([2,0,1])-128)/255.0
        img = np.zeros((3,newh,neww))
        img[:,innerh[0]:innerh[1],innerw[0]:innerw[1]]= im

        self.arg_params["data"] = mx.nd.array(img.reshape((1,)+img.shape))
        ex = self.feat_embbed.bind(ctx=mx.cpu(), args=self.arg_params)
        #print ex.grad_arrays
        #print ex.arg_arrays
        return ex.forward()[0].asnumpy()
