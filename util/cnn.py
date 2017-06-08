import math
import numpy as np
import mxnet as mx
from tqdm import tqdm
import time,os
from easydict import EasyDict as edict
from symbols import *

class CNN(object): 
    def __init__(self, params):
        self.params = params
        self.batch_size = params.batch_size
        self.batch_norm = params.batch_norm
        self.basic_model = params.basic_model

        self.label = self.basic_model + '/'
        self.save_dir = params.save_dir

        self.global_step = 0
        self.conv_feats = None
        self.build() 
        self.loss = None
        self.build_loss()
        #self.saver = tf.train.Saver(max_to_keep = 100) 
    def build_loss(self):
        raise NotImplementedError()
    def build(self):
        if self.basic_model=='vgg16':
            self.build_basic_vgg16()

        elif self.basic_model=='resnet50':
            self.build_basic_resnet50()

        elif self.basic_model=='resnet101':
            self.build_basic_resnet101()

        else:
            self.build_basic_resnet152()

    def build_basic_vgg16(self):
        print("Building the basic VGG16 net...")
        bn = self.batch_norm
        data = mx.symbol.Variable(name="data")
        is_train = mx.symbol.Variable(name="istrain")

        self.conv_feats, self.conv_scale = get_vgg16(bn,is_train,data)

        self.img_data = data
        self.is_train = is_train
        print("Basic VGG16 net built.")

    def load(self, model_prefix, step):
        print("Loading model...") 
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, step)
        initer = mx.init.Xavier()
        self._param_names = [x for x in self.loss.list_arguments() if x not in self.input_names]
        self._aux_names = self.loss.list_auxiliary_states()

        arg_shapes, _, aux_shapes = self.loss.infer_shape(**self.input_shapes)
        arg_types, _, aux_types = self.loss.infer_type(**self.input_types)

        self.arg_params = {}
        self.aux_params = {}

        arg_name2idx = {}
        for i,x in enumerate(self.loss.list_arguments()):
            arg_name2idx[x] = i
        aux_name2idx = {}
        for i,x in enumerate(self.loss.list_auxiliary_states()):
            aux_name2idx[x] = i
        for name in self._param_names:
            if name in arg_params.keys():
                self.arg_params[name] = arg_params[name].copy()
                continue
            desc = mx.init.InitDesc(name)
            w = mx.nd.zeros(arg_shapes[arg_name2idx[name]], dtype=arg_types[arg_name2idx[name]])
            initer(desc,w)
            self.arg_params[name] = w

        for name, arr in self._aux_names:
            if name in aux_params.keys():
                self.aux_params[name] = aux_params[name].copy()
                continue
            desc = mx.init.InitDesc(name)
            w = mx.nd.zeros(aux_shapes[aux_name2idx[name]], dtype=aux_types[aux_name2idx[name]])
            initer(desc,w)
            self.aux_params[name] = w

        #self.init_params(initializer=mx.initializer.Xavier(), arg_params=self.arg_params, aux_params=self.aux_params)
    def loadfromnpy(self, data_path, session, ignore_missing=True):
        print("Loading basic model from %s..." %data_path)
        for v in tf.trainable_variables():
            print v.name
        data_dict = np.load(data_path).item()
        count = 0
        miss_count = 0
        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].iteritems():
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                        count += 1
                        print("Variable %s:%s loaded" %(op_name, param_name))
                    except ValueError,e:
                        miss_count += 1
                        print("Variable %s:%s missed" %(op_name, param_name))
                        print(e)
            if not ignore_missing:
                raise
        print("%d variables loaded. %d variables missed." %(count, miss_count))
    def save_params(self, fname):
        arg_params = self.exe.arg_dict
        aux_params = self.exe.aux_dict
        #self._param_names
        #self._aux_names
        save_dict = {('arg:%s' % k) : v.as_in_context(mx.cpu()) for k, v in arg_params.items()}
        save_dict.update({('aux:%s' % k) : v.as_in_context(mx.cpu()) for k, v in aux_params.items()})
        for n in self.input_names:
            save_dict.pop('arg:%s' % n)
        mx.ndarray.save(fname, save_dict)
    def save(self, npyfile=None):
        assert self.exe is not None
        prefix = self.save_dir
        print("Saving model to %s" %prefix)
        self.loss.save('%s-symbol.json'%prefix)
        param_name = '%s-%04d.params' % (prefix, 10)
        self.save_params(param_name)
        print("Saved checkpoint to %s" %param_name)
        if npyfile is not None:
            self.savenpy(npyfile)
    def savenpy(self, npyfile):
        print("saving the model params to .npy file:"+npyfile)
        data_dict = {}
        op_name = []
        for v in tf.trainable_variables():
            opname, vname = v.name.split('/')[-2:]
            vname = vname.split(':')[0]
            vdata = v.eval(session = sess)
            print opname,vname,vdata.shape
            if opname in op_name:
                data_dict[opname][vname] = vdata.copy()
            else:
                data_dict[opname] = {vname:vdata.copy()}
                op_name.append(opname)
        np.save(npyfile,data_dict)
    def eval_feat(self, sess, imf):
        return sess.run([self.conv_feats], feed_dict={self.is_train:False, self.img_files:imf})
