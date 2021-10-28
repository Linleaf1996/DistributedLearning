import torch
import math
from rff import GaussianRff
PI=math.pi

class MultiKernel:
    def __init__(self,rffs,net=[]):
        self.dim=rffs[0].dim
        self.rffs=rffs
        self.num_kernel=len(rffs)
        self.Ws=[]# a list including parameters of features for different kernels
        self.bs=[]# a list including parameters of features for different kernels
        self.omega=torch.ones(self.num_kernel)/torch.Tensor([self.num_kernel])
        self.vec_D=0
        self.vec_loss=torch.zeros(self.num_kernel)
        self.list_grad=[]
        self.list_weight=[]
        self.net=[]
        for k in rffs:
            self.vec_D+=k.D
            self.Ws.append(k.W)
            self.bs.append(k.b)
        self.grad_vec=torch.zeros(self.vec_D)
        self.theta_vec=torch.zeros(self.vec_D)
    def GenMKF(self,X):
        mkf=[]
        for ker in self.rffs:
            mkf.append(ker.GenRFF(X))
        return mkf
    def Fit(self,X):
        Y=torch.zeros(X.shape[0])
        sum_omega=torch.sum(self.omega)
        nor_omega=self.omega/sum_omega
        for i in range(0,self.num_kernel):
            Y+=(self.rffs[i].Fit(X)*nor_omega[i])
        return Y
    def GenMkfVec(self,X):#concat features from different kernel
        mkf=self.GenMKF(self,X)
        mkf_vec=mkf[0]*self.omega[0]
        i=1
        for i in range(1,len(mkf)):
            mkf_vec=torch.cat((mkf_vec,mkf[i]*self.omega[i],1))
        return mkf_vec

    def OmegaUpdate(self,recv_loss,lr=0.01):
        if recv_loss.shape[0]==0:
            # print('recv_loss={}'.format(recv_loss))
            loss=self.vec_loss.unsqueeze(0)
        else:
            loss=torch.cat((recv_loss,self.vec_loss.unsqueeze(0)),0)
        adj=torch.tensor(recv_loss.shape[0])
        self.omega=self.omega*torch.exp(-torch.sum(loss,0)*torch.tensor(lr)/(adj+1))
        # print(self.omega)
############interface between Graphlearning and different solvers####################
    def Calculate(self,x,y,opt,lr=0.01):
        
        if opt=='grad':
            self.list_grad=[]
            for k in range(0,len(self.rffs)):
                self.rffs[k].LocalGradient(x,y)
                self.vec_loss[k]=self.rffs[k].loss
                self.list_grad.append(self.rffs[k].net.weight.grad.numpy())
            # print('mkl:{} and {}'.format(self.vec_loss.numpy(),self.list_grad))
            return {'grad':self.list_grad,'loss':self.vec_loss.numpy()}

        elif opt=='Mkgrad':
            self.MkLocalGradient(x,y,lr=0.01)
            return self.net.weight.grad.numpy()
            
        else:
            print('Cal developing>>>')
            return []
    #interface between GL and Update function
    def Update(self,recv_info,opt):
        if opt=='grad':
            # print('gradient descend')
            recv_loss=torch.tensor([])
            self.list_weight=[]
            # print('recv_info:{}\n'.format(recv_info))
            recv=[]
            for k in range(0,self.num_kernel):
                recv_k=[]
                for recv_i in recv_info:
                    recv_k.append({'recv_from':recv_i['recv_from'],'info':recv_i['info']['grad'][k]})
                recv.append(recv_k)
            
            
            k=0
            for recv_k in recv:
                self.rffs[k].LocalStep(recv_k)
                
                self.list_weight.append(self.rffs[k].net.weight.data.numpy()) 
                k+=1
            # print({'weight':self.list_weight,'omega':self.omega.numpy()})
            recv_loss=torch.tensor([])
            for recv_i in recv_info:
                recv_loss=torch.cat((recv_loss,torch.from_numpy(recv_i['info']['loss']).unsqueeze(0)),0) 
            # print({'weight':self.list_weight,'omega':self.omega.numpy()})
            self.OmegaUpdate(recv_loss,lr=0.01)
            # print({'weight':self.list_weight,'omega':self.omega.numpy()})
            return {'weight':self.list_weight,'omega':self.omega.numpy()}
        elif opt=='Mkgrad':
            self.LocalStep(recv_info)
            return self.net.weight.data.numpy()
        else:
            print('Up developing>>>')
            return []  
###########opt paras by gradient of mkl loss fun###########################################
    def MkLocalGradient(self,X,Y,lr=0.01):
        criterion=torch.nn.MSELoss()#+torch.matmul(Net.weight.data,Net.weight.data.t())
        opt=torch.optim.SGD(self.net.parameters(),lr)
        mkf_vec=self.GenMkfVec(X)
        yhat=self.net(mkf_vec).squeeze(-1)
        # print('net Y:{}'.format(yhat))
        loss=criterion(yhat,Y)
        opt.zero_grad()
        loss.backward()
        self.grad_vec=self.net.weight.grad.data.squeeze()
        # print('grad:{}'.format(net.weight.grad))
    def MkLocalStep(self,recv_info,ratio=1,lr=0.01):
        d=len(recv_info)
        g_agg=torch.zeros(1,self.D)
        for recv_i in recv_info:
            g=recv_i['info']
            g_agg+=torch.from_numpy(g/(d+1))
        ratio=1/(d+1)
        # print('ratio={}'.format(ratio))
        # print('grad:{} and g_agg:{}'.format(self.net.weight.grad,g_agg))
        self.net.weight.grad=self.net.weight.grad*ratio+g_agg*(1-ratio)
        # print('grad1:{}'.format(net.weight.grad))
        # tweight=copy.deepcopy(net.weight)
        # net.weight.grad=net.weight.grad+10000
        opt=torch.optim.SGD(self.net.parameters(),lr)
        opt.step()
        # print('step_weight:{}---{} '.format(net.weight,tweight))
        self.theta_vec=self.net.weight.data.squeeze()