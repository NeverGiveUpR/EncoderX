import torch
import numpy as np 
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from torch.optim import lr_scheduler

input_size=(300,1)
lr=0.01
EPOCH=5000
max_order=6
num_coefficient=max_order

def GenerateData1D(start,end,input_size,function=None):           
    num_input=input_size[0]
    dim=input_size[1]
    
    data=np.random.uniform(start, end, (num_input))
    #target=function(data)
    #target = np.exp(-data**2/2)
    #target = 0.3 * np.exp(-data) * np.sin(data)
    #target = np.exp(data)
    #target = np.exp(-data**2/2)/(2*3.1314)
    x = data
    target = 3.2+0.1*x+0.78*x**2+0.23*x**3+1.2*x**4

    return data,target

data1,target1=GenerateData1D(-5,5,input_size,function=np.exp)

#plt.scatter(data1,target1,color='blue',marker='.',s=0.5)
#plt.legend(['y=exp(x)'],loc='best',fontsize=12,markerscale=5)
#plt.show()

data=torch.from_numpy(data1)
data=data.type(torch.float)

target=torch.from_numpy(target1)
target=target.type(torch.float)
'''
n_hidden1=300
n_hidden2=4000
n_hidden3=3000
n_hidden4=2000
n_hidden5=1000
n_hidden6=512
n_hidden7=256
n_hidden8=128
n_hidden9=64
n_hidden10=32
n_hidden11=16
n_hidden12=num_coefficient

class EncoderX(torch.nn.Module):
    def __init__(self,n_hidden1,n_hidden2,n_hidden3,n_hidden4,n_hidden5,n_hidden6,n_hidden7,n_hidden8,n_hidden9,n_hidden10,n_hidden11,n_hidden12):
        super(EncoderX,self).__init__()
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


        self.weights1=Parameter(torch.rand([self.n_hidden1,self.n_hidden2]))
        self.weights2=Parameter(torch.rand([self.n_hidden2,self.n_hidden3]))
        self.weights3=Parameter(torch.rand([self.n_hidden3,self.n_hidden4]))
        self.weights4=Parameter(torch.rand([self.n_hidden4,self.n_hidden5]))
        self.weights5=Parameter(torch.rand([self.n_hidden5,self.n_hidden6]))
        self.weights6=Parameter(torch.rand([self.n_hidden6,self.n_hidden7]))
        self.weights7=Parameter(torch.rand([self.n_hidden7,self.n_hidden8]))
        self.weights8=Parameter(torch.rand([self.n_hidden8,self.n_hidden9]))
        self.weights9=Parameter(torch.rand([self.n_hidden9,self.n_hidden10]))
        self.weights10=Parameter(torch.rand([self.n_hidden10,self.n_hidden11]))
        self.weights11=Parameter(torch.rand([self.n_hidden11,self.n_hidden12]))

        
        self.bias1=Parameter(torch.rand([self.n_hidden2]))
        self.bias2=Parameter(torch.rand([self.n_hidden3]))
        self.bias3=Parameter(torch.rand([self.n_hidden4]))
        self.bias4=Parameter(torch.rand([self.n_hidden5]))
        self.bias5=Parameter(torch.rand([self.n_hidden6]))
        self.bias6=Parameter(torch.rand([self.n_hidden7]))
        self.bias7=Parameter(torch.rand([self.n_hidden8]))
        self.bias8=Parameter(torch.rand([self.n_hidden9]))
        self.bias9=Parameter(torch.rand([self.n_hidden10]))
        self.bias10=Parameter(torch.rand([self.n_hidden11]))
        self.bias11=Parameter(torch.rand([self.n_hidden12]))

        
    def forward(self,x):
        layer_1=torch.relu(torch.add(torch.matmul(x,self.weights1),self.bias1))
        layer_1_BN=(torch.nn.LayerNorm(n_hidden2,eps=1e-05))(layer_1)
        layer_2=torch.relu(torch.add(torch.matmul(layer_1_BN,self.weights2),self.bias2))
        layer_2_BN=(torch.nn.LayerNorm(n_hidden3,eps=1e-05))(layer_2)
        layer_3=torch.relu(torch.add(torch.matmul(layer_2_BN,self.weights3),self.bias3))
        layer_3_BN=(torch.nn.LayerNorm(n_hidden4,eps=1e-05))(layer_3)
        layer_4=torch.relu(torch.add(torch.matmul(layer_3_BN,self.weights4),self.bias4))
        layer_4_BN=(torch.nn.LayerNorm(n_hidden5,eps=1e-05))(layer_4)
        layer_5=torch.relu(torch.add(torch.matmul(layer_4_BN,self.weights5),self.bias5))
        layer_5_BN=(torch.nn.LayerNorm(n_hidden6,eps=1e-05))(layer_5)
        layer_6=torch.relu(torch.add(torch.matmul(layer_5_BN,self.weights6),self.bias6))
        layer_6_BN=(torch.nn.LayerNorm(n_hidden7,eps=1e-05))(layer_6)
        layer_7=torch.relu(torch.add(torch.matmul(layer_6_BN,self.weights7),self.bias7))
        layer_7_BN=(torch.nn.LayerNorm(n_hidden8,eps=1e-05))(layer_7)
        layer_8=torch.relu(torch.add(torch.matmul(layer_7_BN,self.weights8),self.bias8))
        layer_8_BN=(torch.nn.LayerNorm(n_hidden9,eps=1e-05))(layer_8)
        layer_9=torch.relu(torch.add(torch.matmul(layer_8_BN,self.weights9),self.bias9))
        layer_9_BN=(torch.nn.LayerNorm(n_hidden10,eps=1e-05))(layer_9)
        layer_10=torch.relu(torch.add(torch.matmul(layer_9_BN,self.weights10),self.bias10))
        layer_10_BN=(torch.nn.LayerNorm(n_hidden11,eps=1e-05))(layer_10)
        layer_11=torch.add(torch.matmul(layer_10_BN,self.weights11),self.bias11)
        

        action=[layer_11[z] for z in range(num_coefficient-1)]
        bias=layer_11[-1]
        pred=torch.zeros([n_hidden1])
        nums=0
        for order_num in range(num_coefficient-1):
            temp=action[order_num]*np.power(data,(order_num+1))
            pred=pred+temp
        pred=pred+bias
        return layer_11,pred
'''

n_hidden1 = 300
n_hidden2 = 1024
n_hidden3 = 512
n_hidden4 = 256
n_hidden5 = 128
n_hidden6 = 64
n_hidden7 = num_coefficient


class EncoderX(torch.nn.Module):
    def __init__(self, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5, n_hidden6, n_hidden7):
        super(EncoderX, self).__init__()
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.n_hidden3 = n_hidden3
        self.n_hidden4 = n_hidden4
        self.n_hidden5 = n_hidden5
        self.n_hidden6 = n_hidden6
        self.n_hidden7 = n_hidden7

        self.weights1 = Parameter(torch.rand([self.n_hidden1, self.n_hidden2]))
        self.weights2 = Parameter(torch.rand([self.n_hidden2, self.n_hidden3]))
        self.weights3 = Parameter(torch.rand([self.n_hidden3, self.n_hidden4]))
        self.weights4 = Parameter(torch.rand([self.n_hidden4, self.n_hidden5]))
        self.weights5 = Parameter(torch.rand([self.n_hidden5, self.n_hidden6]))
        self.weights6 = Parameter(torch.rand([self.n_hidden6, self.n_hidden7]))

        self.bias1 = Parameter(torch.rand([self.n_hidden2]))
        self.bias2 = Parameter(torch.rand([self.n_hidden3]))
        self.bias3 = Parameter(torch.rand([self.n_hidden4]))
        self.bias4 = Parameter(torch.rand([self.n_hidden5]))
        self.bias5 = Parameter(torch.rand([self.n_hidden6]))
        self.bias6 = Parameter(torch.rand([self.n_hidden7]))

    def forward(self, x):
        layer_1 = torch.relu(torch.add(torch.matmul(x, self.weights1), self.bias1))
        layer_1_BN = (torch.nn.LayerNorm(n_hidden2, eps=1e-05))(layer_1)
        layer_2 = torch.relu(torch.add(torch.matmul(layer_1_BN, self.weights2), self.bias2))
        layer_2_BN = (torch.nn.LayerNorm(n_hidden3, eps=1e-05))(layer_2)
        layer_3 = torch.relu(torch.add(torch.matmul(layer_2_BN, self.weights3), self.bias3))
        layer_3_BN = (torch.nn.LayerNorm(n_hidden4, eps=1e-05))(layer_3)
        layer_4 = torch.relu(torch.add(torch.matmul(layer_3_BN, self.weights4), self.bias4))
        layer_4_BN = (torch.nn.LayerNorm(n_hidden5, eps=1e-05))(layer_4)
        layer_5 = torch.relu(torch.add(torch.matmul(layer_4_BN, self.weights5), self.bias5))
        layer_5_BN = (torch.nn.LayerNorm(n_hidden6, eps=1e-05))(layer_5)
        layer_6 = torch.add(torch.matmul(layer_5_BN, self.weights6), self.bias6)

        action = [layer_6[z] for z in range(num_coefficient - 1)]
        bias = layer_6[-1]
        pred = torch.zeros([n_hidden1])
        nums = 0
        for order_num in range(num_coefficient - 1):
            temp = action[order_num] * np.power(data, (order_num + 1))
            pred = pred + temp
        pred = pred + bias
        return layer_6, pred
net=EncoderX(n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5, n_hidden6, n_hidden7)

optimizer = torch.optim.Adam(net.parameters(), lr=lr)
loss_function = torch.nn.MSELoss()

t = 0
pre_loss = 0
while True:
    coefficients1, pred1 = net(target)

    loss1 = loss_function(target, pred1)

    #if abs(pre_loss-loss1.item()) < 0.0001 and t > 100:
    if loss1.item() < 0.00005:
        #print(abs(pre_loss - loss1.item()))
        break
    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()
    t += 1
    pre_loss = loss1.item()
    print(t, loss1.item())
print("final loss:", loss1.item())
print("coef:", coefficients1)

'''
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
'''
plt.figure()
plt.scatter(data1,target1,color='blue',marker='.',s=0.5)
plt.scatter(data1,pred1.data.numpy(),color='red',marker='.',s=0.5)
plt.legend(['y=exp(x)','Encoder X'],loc='best',fontsize=12,markerscale=5)
plt.show()
