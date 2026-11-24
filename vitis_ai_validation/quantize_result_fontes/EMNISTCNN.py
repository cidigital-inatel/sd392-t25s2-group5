# GENETARED BY NNDCT, DO NOT EDIT!

import torch
from torch import tensor
import pytorch_nndct as py_nndct

class EMNISTCNN(py_nndct.nn.NndctQuantModel):
    def __init__(self):
        super(EMNISTCNN, self).__init__()
        self.module_0 = py_nndct.nn.Input() #EMNISTCNN::input_0(EMNISTCNN::nndct_input_0)
        self.module_1 = py_nndct.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #EMNISTCNN::EMNISTCNN/Sequential[conv1]/Conv2d[0]/ret.3(EMNISTCNN::nndct_conv2d_1)
        self.module_2 = py_nndct.nn.ReLU(inplace=False) #EMNISTCNN::EMNISTCNN/Sequential[conv1]/ReLU[2]/ret.7(EMNISTCNN::nndct_relu_2)
        self.module_3 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #EMNISTCNN::EMNISTCNN/Sequential[conv1]/MaxPool2d[3]/523(EMNISTCNN::nndct_maxpool_3)
        self.module_4 = py_nndct.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1], dilation=[1, 1], groups=1, bias=True) #EMNISTCNN::EMNISTCNN/Sequential[conv2]/Conv2d[0]/ret.9(EMNISTCNN::nndct_conv2d_4)
        self.module_5 = py_nndct.nn.ReLU(inplace=False) #EMNISTCNN::EMNISTCNN/Sequential[conv2]/ReLU[2]/ret.13(EMNISTCNN::nndct_relu_5)
        self.module_6 = py_nndct.nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2], padding=[0, 0], dilation=[1, 1], ceil_mode=False) #EMNISTCNN::EMNISTCNN/Sequential[conv2]/MaxPool2d[3]/567(EMNISTCNN::nndct_maxpool_6)
        self.module_7 = py_nndct.nn.Module('nndct_shape') #EMNISTCNN::EMNISTCNN/570(EMNISTCNN::nndct_shape_7)
        self.module_8 = py_nndct.nn.Module('nndct_reshape') #EMNISTCNN::EMNISTCNN/ret.17(EMNISTCNN::nndct_reshape_8)
        self.module_9 = py_nndct.nn.Linear(in_features=3136, out_features=128, bias=True) #EMNISTCNN::EMNISTCNN/Sequential[fc]/Linear[0]/ret.19(EMNISTCNN::nndct_dense_9)
        self.module_10 = py_nndct.nn.ReLU(inplace=False) #EMNISTCNN::EMNISTCNN/Sequential[fc]/ReLU[1]/ret.21(EMNISTCNN::nndct_relu_10)
        self.module_11 = py_nndct.nn.Linear(in_features=128, out_features=62, bias=True) #EMNISTCNN::EMNISTCNN/Sequential[fc]/Linear[3]/ret(EMNISTCNN::nndct_dense_11)

    @py_nndct.nn.forward_processor
    def forward(self, *args):
        output_module_0 = self.module_0(input=args[0])
        output_module_0 = self.module_1(output_module_0)
        output_module_0 = self.module_2(output_module_0)
        output_module_0 = self.module_3(output_module_0)
        output_module_0 = self.module_4(output_module_0)
        output_module_0 = self.module_5(output_module_0)
        output_module_0 = self.module_6(output_module_0)
        output_module_7 = self.module_7(input=output_module_0, dim=0)
        output_module_8 = self.module_8(input=output_module_0, shape=[output_module_7,-1])
        output_module_8 = self.module_9(output_module_8)
        output_module_8 = self.module_10(output_module_8)
        output_module_8 = self.module_11(output_module_8)
        return output_module_8
