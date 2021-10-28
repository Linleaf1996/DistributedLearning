import math
import torch
import copy
PI=math.pi
    #NodesAggregation:the nodes opt their arguments referring to neighbors
class GaussianRff:#the random_fourier_feature is only for gaussian kernel approximation
    def __init__(self,dim,gamma,net,D=10):#dim is the dimension of data,D is the number of the features
        self.W=torch.randn(D,dim)*torch.sqrt(torch.Tensor([2*gamma]))
        self.dim=dim
        self.b=torch.FloatTensor(D).uniform_(0,2*PI) 
        self.theta=torch.zeros(D)
        self.D=D #number of features
        self.grad=torch.zeros(D)
        self.net=net
        self.loss=torch.tensor([0])
    def GenRFF(self,X):#return a tensor(n_smaples,D)
        n_samples,dim=X.size()
        X_feature=torch.sqrt(torch.Tensor([2/dim]))* torch.cos(torch.matmul(self.W,X.t())+self.b.repeat(n_samples,1).t())
        return X_feature.t()
    def Fit(self,X):
        ff=self.GenRFF(X)
        Y=torch.matmul(ff,self.theta)
        return Y
    def SolveTheta(self,X,Y):#solve parameters of RFF by linear system
        X=self.GenRFF(X)
        X=torch.pinverse(X)
        self.theta=torch.matmul(X,Y)
        return self.theta
    #interface between GL and Calcualte function
    def Calculate(self,x,y,opt):
        if opt=='grad':
            self.LocalGradient(x,y,lr=0.01)
            return self.net.weight.grad.numpy()
        else:
            print('Cal developing>>>')
            return []
    #interface between GL and Update function
    def Update(self,recv_info,opt):
        if opt=='grad':
            # print('recv_info:{}'.format(recv_info))
            self.LocalStep(recv_info,ratio=1,lr=0.01)
            return self.net.weight.data.numpy()
        else:
            print('Up developing>>>')
            return []

###########opt paras by gradient###########################################

    def LocalGradient(self,X,Y,lr=0.01):
        criterion=torch.nn.MSELoss()#+torch.matmul(Net.weight.data,Net.weight.data.t())
        opt=torch.optim.SGD(self.net.parameters(),lr,weight_decay=0.001)
        xf=self.GenRFF(X)
        yhat=self.net(xf).squeeze(-1)
        # print('net Y:{}'.format(yhat))
        loss=criterion(yhat,Y)
        self.loss=torch.tensor([loss.data])
        # print('loss:{},'.format(loss))
        opt.zero_grad()
        loss.backward()
        self.grad=self.net.weight.grad.data.squeeze()
        # print('w:{}'.format(self.net.weight.data))
        # print('grad:{}'.format(net.weight.grad))
    def LocalStep(self,recv_info,ratio=1,lr=0.01):
        d=len(recv_info)
        g_agg=torch.zeros(1,self.D)
        for recv_i in recv_info:
            g=recv_i['info']
            # print('vex_{} grad:{},selfgrad{}'.format(recv_i['from_vex'],g[0,1:2],self.net.weight.grad[0,1:2]))
            g_agg+=torch.from_numpy(g/d)
        ratio=1/(d+1)
        # print('ratio={}'.format(ratio))
        # pregrad=copy.deepcopy(self.net.weight.grad[0,1:2])
        # print('pre_weight:{} '.format(self.net.weight.data[0,1:2]))
        # print('grad:{} and g_agg:{}'.format(self.net.weight.grad[0,1:2],g_agg[0,1:2]))
        # print(d)
        self.net.weight.grad=self.net.weight.grad*ratio+g_agg*(1-ratio)#self.net.weight.grad*ratio+g_agg*(1-ratio)
        # print('AGGR grad:{},pregrad:{}'.format(self.net.weight.grad[0,1:2]),pregrad)
        opt=torch.optim.SGD(self.net.parameters(),lr,weight_decay=0.001)
        opt.step()
        # print('step_weight:{} '.format(self.net.weight.data[0,1:2]))
        self.theta=self.net.weight.data.squeeze()

