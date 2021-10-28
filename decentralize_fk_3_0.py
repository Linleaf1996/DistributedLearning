from multiprocessing import Process,Queue,Manager
import torch
import time
import numpy as np
import networkx as nx
import copy
'''
The mini-framework is for commmunication and update of parameters among nodes in a graph by Multiprocesses at simple computer
Made by Lin Yifu from Beihang University in Beijing 
If u have any question or advice, please be free send me mail(E-mail:lyf1996leaf@163.com)
'''
class GraphLearning:
    def __init__(self,graph):
        self.graph=graph
        self.num_nodes=graph.number_of_nodes()
        self.ques=[]
        self.mques=[]# communication between main proc and child proc
        # self.send=[]
        # self.recv=[]
        self.share_space= Manager().dict()
###################   New Mode  ####################
    ############## Upate Topology   ###############################
    def AdjUpdate(self):
        #record adj for v=every node
        adjmatrix=np.array(nx.adjacency_matrix(self.graph).todense())
        for v in range(0,self.num_nodes):
            adj_list_r=[]#send to who
            adj_list_c=[]#receive from who
            for j in range(0,self.num_nodes):
                if adjmatrix[v,j]!=0:
                    adj_list_r.append(j)
                if adjmatrix[j,v]!=0:
                    adj_list_c.append(j)
            self.share_space['send_to_adj']+=[adj_list_r]
            self.share_space['recv_fr_adj']+=[adj_list_c]
        # print(self.share_space['send_to_adj'],self.share_space['recv_fr_adj'])
    ###############  Communication  ##############################
    def SendMessage(self,ques,m,towho):
        ques[towho].put(m)
    def RecvMessage(self,ques,recv_id,from_len):
        #recv_id is the node who is receiving message
        self.share_space['vex_{}_recv'.format(recv_id)]=[]
        t0=time.time()
        i=0
        m=[]
        while(i<from_len):
            t1=time.time()
            T=2
            if t1-t0>T:#if time is over, break 
                print('Communication time is over {}s'.format(T))
                break
            if ques[recv_id].qsize()==0:
                continue
            mi=ques[recv_id].get()
            m+=[mi]
            i+=1
        return m
    #################   TRAIN PARAMETERS   ##########################
    def NodeLocalCalculate(self,t,batch,vex,opt='grad'):
        # here we can modify the code by Dataloader()
        x=self.graph.nodes[vex]['data']['x'][t:t+batch,:]
        y=self.graph.nodes[vex]['data']['y'][t:t+batch]
        cal_para=self.graph.nodes[vex]['learner'].Calculate(x,y,opt)
        return cal_para
    def NodeUpdate(self,vex,recv_info,opt='grad'):
        update_para=self.graph.nodes[vex]['learner'].Update(recv_info,opt)
        return update_para


    def Gplot(self):
        pass
    



    #################   Node and Graph Learning    ###############################
    def NodeLearning(self,mques,ques,v,T):
        #ques[v] includes (time t, adj_list) received from server and info received from other nodes
        # note: the server doesnt take part in calculating but sends info about time and adj to learners as a manager
        # recv info of big world which means this class:
        # because of this child process, if we want info at real time we need to look up reletive varibles in share space
        t=0
        para=[]
        while self.share_space['LearnerRun'] and t<T:# continue running or not
            # print('t{},vex_{},time:{}'.format(t,v,self.share_space['time']))
            t=mques[v][0].get()
            #cal#####################################
            batch=self.graph.nodes[v]['batch']
            cal_para=self.NodeLocalCalculate(t,batch,v,opt='grad')            
            #send###################################
            s_to_ids=self.share_space['send_to_adj'][v]#this operation at every itr is designed for topology that is dynamic at real time
            s_m={'recv_from':v,'time':t,'info':cal_para}
            for s_i in s_to_ids:
                self.SendMessage(ques,s_m,s_i)
            #recv######################################
            r_fr_ids=self.share_space['recv_fr_adj'][v]#this operation at every itr is designed for topology that is dynamic at real time
            r_m=self.RecvMessage(ques,v,len(r_fr_ids))
            #update####################################
            up_para=self.NodeUpdate(v,r_m,opt='grad')
            # for rmi in r_m:
            #     print('t{},vex_{},recv from:{} at {},recv:{}!\n'.format(t,v,rmi['recv_from'],
            #     rmi['time'],rmi['info'][0,0:2]))
            para.append({'cal_para':cal_para,'up_para':up_para})
            # print('t{},vex_{},recv:{}\n'.format(t,v,r_m))
            #################################################
            mques[v][1].put(t)
            t+=1
            
                
        self.share_space['vex_{}_para'.format(v)]=para
        print('node {}: train is over!!!!!!!!!\n'.format(v))     

    #Graph learning
    def GLearning(self,T,G_dynamic=False):
        print('Glearning start!')
        learners=[]
        for v in range(0,self.num_nodes):
            self.ques.append(Queue())
            self.mques.append([Queue(),Queue()])
        for v in range(0,self.num_nodes):
            
            learner=Process(target=self.NodeLearning,args=(self.mques,self.ques,v,T))
            learners.append(learner)    
        #############   Init Siganal     ##############
        self.share_space['send_to_adj']=[]
        self.share_space['recv_fr_adj']=[]
        self.AdjUpdate()
        self.share_space['LearnerRun']=True
        ############# Start Node Learning############## 
        for v in range(0,self.num_nodes):
            learners[v].start()
        ############   Train    #############################
        for t in range(0,T):
            # print('#################  {}th Update     ###################################\n'.format(t)) 
            time.sleep(0.001)
            for v in range(0,self.num_nodes):#tell node to start next itr
                self.mques[v][0].put(t)
            ######################
            # confirm if update finished for every node
            for v in range(0,self.num_nodes):
                # print('waiting for vex_{} updating at time{} ...\n'.format(v,t))
                if self.mques[v][1].get()==t:
                    # print('t{}, v{} update finished!!!\n'.format(t,v)) 
                    pass
            
            if G_dynamic:     
                self.AdjUpdate() # means topology is dynamic      
        time.sleep(0.01)
        self.share_space['LearnerRun']=False
        for v in range(0,self.num_nodes):
            learners[v].join()
        for v in range(0,self.num_nodes):
            self.graph.nodes[v]['para']=self.share_space['vex_{}_para'.format(v)]
        
        self.ques=[]
        self.mques=[]
        