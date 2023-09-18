import torch
import numpy as np 
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from torch.optim import lr_scheduler

input_size=(5000,1)
lr=0.01
EPOCH=2000
max_order=10
num_coefficient=max_order

def GenerateData1D(start,end,input_size,function=None):           
    num_input=input_size[0]
    dim=input_size[1]
    
    data=np.linspace(start,end,num_input)
    target=1/(1+25*np.power(data,2))

    return data,target

data1,target1=GenerateData1D(-1,1,input_size)

plt.scatter(data1,target1,color='blue',marker='.',s=0.5)
plt.legend(['y=exp(x)'],loc='best',fontsize=12,markerscale=5)
plt.show()

input_size1=(10,1)
data2,target2=GenerateData1D(-1,1,input_size1)

data=torch.from_numpy(data2)
data=data.type(torch.float)

target=torch.from_numpy(target2)
target=target.type(torch.float)

n_hidden1=10
n_hidden2=128
n_hidden3=64
n_hidden4=32
n_hidden5=16
n_hidden6=num_coefficient

class EncoderX(torch.nn.Module):
    def __init__(self,n_hidden1,n_hidden2,n_hidden3,n_hidden4,n_hidden5,n_hidden6):
        super(EncoderX,self).__init__()
        self.n_hidden1=n_hidden1
        self.n_hidden2=n_hidden2
        self.n_hidden3=n_hidden3
        self.n_hidden4=n_hidden4
        self.n_hidden5=n_hidden5
        self.n_hidden6=n_hidden6


        self.weights1=Parameter(torch.rand([self.n_hidden1,self.n_hidden2]))
        self.weights2=Parameter(torch.rand([self.n_hidden2,self.n_hidden3]))
        self.weights3=Parameter(torch.rand([self.n_hidden3,self.n_hidden4]))
        self.weights4=Parameter(torch.rand([self.n_hidden4,self.n_hidden5]))
        self.weights5=Parameter(torch.rand([self.n_hidden5,self.n_hidden6]))

        
        self.bias1=Parameter(torch.rand([self.n_hidden2]))
        self.bias2=Parameter(torch.rand([self.n_hidden3]))
        self.bias3=Parameter(torch.rand([self.n_hidden4]))
        self.bias4=Parameter(torch.rand([self.n_hidden5]))
        self.bias5=Parameter(torch.rand([self.n_hidden6]))

        
    def forward(self,x):
        layer_1=torch.relu(torch.add(torch.matmul(x,self.weights1),self.bias1))
        layer_1_BN=(torch.nn.LayerNorm(n_hidden2,eps=1e-05))(layer_1)
        layer_2=torch.relu(torch.add(torch.matmul(layer_1_BN,self.weights2),self.bias2))
        layer_2_BN=(torch.nn.LayerNorm(n_hidden3,eps=1e-05))(layer_2)
        layer_3=torch.relu(torch.add(torch.matmul(layer_2_BN,self.weights3),self.bias3))
        layer_3_BN=(torch.nn.LayerNorm(n_hidden4,eps=1e-05))(layer_3)
        layer_4=torch.relu(torch.add(torch.matmul(layer_3_BN,self.weights4),self.bias4))
        layer_4_BN=(torch.nn.LayerNorm(n_hidden5,eps=1e-05))(layer_4)
        layer_5=torch.add(torch.matmul(layer_4_BN,self.weights5),self.bias5)
        

        action=[layer_5[z] for z in range(num_coefficient-1)]
        bias=layer_5[-1]
        pred=torch.zeros([10])
        nums=0
        for order_num in range(num_coefficient-1):
            temp=action[order_num]*np.power(data,(order_num+1))
            pred=pred+temp
        pred=pred+bias
        return layer_5,pred

net=EncoderX(n_hidden1,n_hidden2,n_hidden3,n_hidden4,n_hidden5,n_hidden6)

optimizer=torch.optim.Adam(net.parameters(),lr=lr)
loss_function=torch.nn.MSELoss()
loss1_=[]
for epoch in range(EPOCH):
    coefficients1,pred1=net(target)

    loss1=loss_function(target,pred1)

    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()

    print('|Epoch:',epoch,'|Loss:',loss1.data.numpy())
    loss1_.append(loss1.data.numpy())
    print('Taylor coefficients:',coefficients1.data.numpy())

data=torch.from_numpy(data1)
data=data.type(torch.float)

target=torch.from_numpy(target1)
target=target.type(torch.float)

action=[coefficients1[z] for z in range(num_coefficient-1)]
bias=coefficients1[-1]
pred=torch.zeros([5000])
nums=0
for order_num in range(num_coefficient-1):
    temp=action[order_num]*np.power(data,(order_num+1))
    pred=pred+temp
pred=pred+bias

loss2=loss_function(target,pred)

print('|Loss2:',loss2.data.numpy())

plt.figure()
plt.scatter(data1,target1,color='blue',marker='.',s=0.5)
plt.scatter(data1,pred.data.numpy(),color='red',marker='.',s=0.5)
plt.legend(['y=exp(x)','Encoder X'],loc='best',fontsize=12,markerscale=5)
plt.show()