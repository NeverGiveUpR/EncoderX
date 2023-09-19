import torch
import numpy as np 
from torch.nn.parameter import Parameter 
import matplotlib.pyplot as plt 
import csv 
#from sympy.parsing.sympy_parser import parse_expr 
#from sympy import plot_implicit
import math
#from mpl_toolkits.mplot3d import Axes3D
#from sympy import *
from torch.optim import lr_scheduler

def data_loader(filename):
    data=[]
    i=0
    with open(filename,'r') as f:
        reader=csv.reader(f)
        for row in reader:
            if row != [] :
                row[0]=int(float(row[0]))
                row[1]=int(float(row[1]))
                data.append(row)
            i+=1
        return data

data2=data_loader(r'./butterfly0.csv')
data2=np.array(data2)

x1=np.mean(data2,axis=0)
data_size=data2.shape[0]

data2=(data2-x1)/100
data1=data2*0.97
data3=data2*1.03

plt.figure()
plt.scatter(data2[:,0]/476,data2[:,1]/409,color='red',s=0.5)
plt.show()

data1=torch.from_numpy(data1)
data1=data1.type(torch.float)

data2=torch.from_numpy(data2)
data2=data2.type(torch.float)

data3=torch.from_numpy(data3)
data3=data3.type(torch.float)


data=torch.cat([data1,data2],0)
data=torch.cat([data,data3],0).cuda()


#fig=plt.figure()
#ax=Axes3D(fig)
#ax.scatter(data[:,0]/476,data[:,1]/409,target,c='red',marker='.',s=15)
#ax.set_zlim(-2,2)
#plt.show()

#hyper parameters
EPOCH=1
max_order=8
LR1=0.06


num_coefficient=0
for num in range(1,max_order+1):
    num_coefficient+=num
num_coefficient=num_coefficient+max_order+1


n_hidden1=data_size*3
n_hidden2=4000
n_hidden3=3500
n_hidden4=3000
n_hidden5=2500
n_hidden6=2000
n_hidden7=1500
n_hidden8=1000
n_hidden9=800
n_hidden10=600
n_hidden11=400
n_hidden12=200
n_hidden13=100
n_hidden14=50
n_hidden15=num_coefficient


class IPEncoder(torch.nn.Module):
    def __init__(self,n_hidden1,n_hidden2,n_hidden3,n_hidden4,n_hidden5,n_hidden6,n_hidden7,n_hidden8,n_hidden9,n_hidden10,n_hidden11,n_hidden12,n_hidden13,n_hidden14,n_hidden15):
        super(IPEncoder,self).__init__()
        self.n_hidden1=n_hidden1
        self.n_hidden2=n_hidden2
        self.n_hidden3=n_hidden3
        self.n_hidden4=n_hidden4
        self.n_hidden5=n_hidden5
        self.n_hidden6=n_hidden6
        self.n_hidden7=n_hidden7
        self.n_hidden8=n_hidden8
        self.n_hidden9=n_hidden9
        self.n_hidden10=n_hidden10
        self.n_hidden11=n_hidden11
        self.n_hidden12=n_hidden12
        self.n_hidden13=n_hidden13
        self.n_hidden14=n_hidden14
        self.n_hidden15=n_hidden15


        self.weights1_1=Parameter(torch.rand([self.n_hidden1,self.n_hidden2]).cuda())
        self.weights1_2=Parameter(torch.rand([self.n_hidden1,self.n_hidden2]).cuda())
        self.weights2=Parameter(torch.rand([self.n_hidden2,self.n_hidden3]).cuda())
        self.weights3=Parameter(torch.rand([self.n_hidden3,self.n_hidden4]).cuda())
        self.weights4=Parameter(torch.rand([self.n_hidden4,self.n_hidden5]).cuda())
        self.weights5=Parameter(torch.rand([self.n_hidden5,self.n_hidden6]).cuda())
        self.weights6=Parameter(torch.rand([self.n_hidden6,self.n_hidden7]).cuda())
        self.weights7=Parameter(torch.rand([self.n_hidden7,self.n_hidden8]).cuda())
        self.weights8=Parameter(torch.rand([self.n_hidden8,self.n_hidden9]).cuda())
        self.weights9=Parameter(torch.rand([self.n_hidden9,self.n_hidden10]).cuda())
        self.weights10=Parameter(torch.rand([self.n_hidden10,self.n_hidden11]).cuda())
        self.weights11=Parameter(torch.rand([self.n_hidden11,self.n_hidden12]).cuda())
        self.weights12=Parameter(torch.rand([self.n_hidden12,self.n_hidden13]).cuda())
        self.weights13=Parameter(torch.rand([self.n_hidden13,self.n_hidden14]).cuda())
        self.weights14=Parameter(torch.rand([self.n_hidden14,self.n_hidden15]).cuda())

        
        self.bias1_1=Parameter(torch.rand([self.n_hidden2]).cuda())
        self.bias1_2=Parameter(torch.rand([self.n_hidden2]).cuda())
        self.bias2=Parameter(torch.rand([self.n_hidden3]).cuda())
        self.bias3=Parameter(torch.rand([self.n_hidden4]).cuda())
        self.bias4=Parameter(torch.rand([self.n_hidden5]).cuda())
        self.bias5=Parameter(torch.rand([self.n_hidden6]).cuda())
        self.bias6=Parameter(torch.rand([self.n_hidden7]).cuda())
        self.bias7=Parameter(torch.rand([self.n_hidden8]).cuda())
        self.bias8=Parameter(torch.rand([self.n_hidden9]).cuda())
        self.bias9=Parameter(torch.rand([self.n_hidden10]).cuda())
        self.bias10=Parameter(torch.rand([self.n_hidden11]).cuda())
        self.bias11=Parameter(torch.rand([self.n_hidden12]).cuda())
        self.bias12=Parameter(torch.rand([self.n_hidden13]).cuda())
        self.bias13=Parameter(torch.rand([self.n_hidden14]).cuda())
        self.bias14=Parameter(torch.rand([self.n_hidden15]).cuda())


        
    def forward(self,x,y):
        layer_1=torch.relu(torch.add(torch.matmul(x,self.weights1_1),self.bias1_1)+torch.add(torch.matmul(y,self.weights1_2),self.bias1_2))
        layer_1_BN=(torch.nn.LayerNorm(n_hidden2,eps=1e-05).cuda())(layer_1)
        layer_2=torch.relu(torch.add(torch.matmul(layer_1_BN,self.weights2),self.bias2))
        layer_2_BN=(torch.nn.LayerNorm(n_hidden3,eps=1e-05).cuda())(layer_2)
        layer_3=torch.relu(torch.add(torch.matmul(layer_2_BN,self.weights3),self.bias3))
        layer_3_BN=(torch.nn.LayerNorm(n_hidden4,eps=1e-05).cuda())(layer_3)
        layer_4=torch.relu(torch.add(torch.matmul(layer_3_BN,self.weights4),self.bias4))
        layer_4_BN=(torch.nn.LayerNorm(n_hidden5,eps=1e-05).cuda())(layer_4)
        layer_5=torch.relu(torch.add(torch.matmul(layer_4_BN,self.weights5),self.bias5))
        layer_5_BN=(torch.nn.LayerNorm(n_hidden6,eps=1e-05).cuda())(layer_5)
        layer_6=torch.relu(torch.add(torch.matmul(layer_5_BN,self.weights6),self.bias6))
        layer_6_BN=(torch.nn.LayerNorm(n_hidden7,eps=1e-05).cuda())(layer_6)
        layer_7=torch.relu(torch.add(torch.matmul(layer_6_BN,self.weights7),self.bias7))
        layer_7_BN=(torch.nn.LayerNorm(n_hidden8,eps=1e-05).cuda())(layer_7)
        layer_8=torch.relu(torch.add(torch.matmul(layer_7_BN,self.weights8),self.bias8))
        layer_8_BN=(torch.nn.LayerNorm(n_hidden9,eps=1e-05).cuda())(layer_8)
        layer_9=torch.relu(torch.add(torch.matmul(layer_8_BN,self.weights9),self.bias9))
        layer_9_BN=(torch.nn.LayerNorm(n_hidden10,eps=1e-05).cuda())(layer_9)
        layer_10=torch.relu(torch.add(torch.matmul(layer_9_BN,self.weights10),self.bias10))
        layer_10_BN=(torch.nn.LayerNorm(n_hidden11,eps=1e-05).cuda())(layer_10)
        layer_11=torch.relu(torch.add(torch.matmul(layer_10_BN,self.weights11),self.bias11))
        layer_11_BN=(torch.nn.LayerNorm(n_hidden12,eps=1e-05).cuda())(layer_11)
        layer_12=torch.relu(torch.add(torch.matmul(layer_11_BN,self.weights12),self.bias12))
        layer_12_BN=(torch.nn.LayerNorm(n_hidden13,eps=1e-05).cuda())(layer_12)
        layer_13=torch.relu(torch.add(torch.matmul(layer_12_BN,self.weights13),self.bias13))
        layer_13_BN=(torch.nn.LayerNorm(n_hidden14,eps=1e-05).cuda())(layer_13)
        layer_14=torch.add(torch.matmul(layer_13_BN,self.weights14),self.bias14)
        

        action=[layer_14[z] for z in range(num_coefficient-1)]
        bias=layer_14[-1]
        pred=torch.zeros([data_size*3]).cuda()
        nums=0
        for order_num in range(1,max_order+1):
                for num in range(order_num+1):
                        temp=action[nums]*torch.pow(x,num)*torch.pow(y,order_num-num)
                        nums+=1
                        pred=pred+temp.float()
        pred=pred+bias

        return layer_14,pred

net=IPEncoder(n_hidden1,n_hidden2,n_hidden3,n_hidden4,n_hidden5,n_hidden6,n_hidden7,n_hidden8,n_hidden9,n_hidden10,n_hidden11,n_hidden12,n_hidden13,n_hidden14,n_hidden15)

optimizer1=torch.optim.Adam(net.parameters(),lr=LR1)

scheduler1=lr_scheduler.StepLR(optimizer1,step_size=800,gamma=0.85)
loss_func=torch.nn.MSELoss()
#loss_func=torch.nn.L1Loss()
#loss_func1=torch.nn.SmoothL1Loss()

class Loss_Func(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,target,pred):
        return torch.dist(target,pred,p=2)

loss_func2=Loss_Func()

target1=0.03*torch.ones([data_size])
target2=torch.zeros([data_size])
target3=-0.03*torch.ones([data_size])

target=torch.cat([target1,target2],0)
target=torch.cat([target,target3],0).cuda()


#fig=plt.figure()

for epoch in range(EPOCH):
    scheduler1.step()


    coefficients,pred=net(data[:,0],data[:,1])

    loss=loss_func(target,pred)


    optimizer1.zero_grad()
    loss.backward()
    optimizer1.step()

    print('|Epoch:',epoch,'|Loss:%6f'%loss.cpu().data.numpy())
    #print('pred:',pred.cpu().data.numpy())
    #print('coefficients:',coefficients.data.numpy())
    #print('output:',output.data.numpy())


    #ax=Axes3D(fig)
    #ax.scatter(data[:,0],data[:,1],output.data.numpy(),c='red',marker='.',s=15)
    #ax.scatter(data[:,0],data[:,1],target,c='blue',marker='.',s=15)
    #ax.scatter(data[:,0],data[:,1],pred.data.numpy(),c='green',marker='.',s=15)
    #ax.set_zlim(-2.0,2.0)
    #plt.pause(0.3)

    action=[coefficients[x] for x in range(num_coefficient-1)]
    bias=coefficients[-1]

    nums=0
    pred='0'
    for order_num in range(1,max_order+1):
        for num in range(order_num+1):
            temp='+('+str(action[nums].cpu().data.numpy())+')*(x**'+str(num)+')*(y**'+str(order_num-num)+')'
            pred=pred+temp
            nums+=1
    pred=pred+'+('+str(bias.cpu().data.numpy())+')'
    expression=str(pred)
    print(expression)

#x,y=symbols('x y')
#plot_implicit(parse_expr(expression),(x,0,1.5),(y,0,1.5))
