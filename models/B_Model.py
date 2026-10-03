from importlib import import_module
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from gnnmodels.BatchProgramClassifier import BatchProgramClassifier
from ParameterConfig import ParameterConfig
from gnnmodels.OptimizedGATClassifier import OptimizedGATClassifier
from gnnmodels.DPCNN import Config,Model
#import gnnmodels.TextCNN as x
import numpy as np

'''class SelfAttention(nn.Module):
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.concat_dense = nn.Linear(2 * in_dim, 100)
        self.final_dropout = nn.Dropout(0.5)
        self.prediction = nn.Linear(100, 1)
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        x = self.concat_dense(x)
        x = self.relu(x)
        x = self.final_dropout(x)
        x = self.prediction(x).sigmoid()
        return x

out12 = th.cat([out1,out2], dim=1)
out12 = SelfAttention(out12.size(1))(out12)'''

class B_Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, HEAD_NUM, emb_ast, n_classes, tokens_size, emb_cfg,
                 device=None, layer_num=2,activation=F.relu):
        super(B_Model, self).__init__()
        self.n_classes = n_classes
        self.device = device
        # self.embedding_matrix = embedding_matrix
        self.layers = nn.ModuleList()
        self.emb_ast = emb_ast
        #node_vec_stg='TextCNN'
        node_vec_stg = 'mean'
        self.model1 = OptimizedGATClassifier(ParameterConfig.GAT_HIDDEN_DIM, ParameterConfig.GAT_HIDDEN_DIM, HEAD_NUM, # 
                                              self.n_classes, emb_cfg, node_vec_stg=node_vec_stg).to(device)
        self.model2 = OptimizedGATClassifier(ParameterConfig.GAT_HIDDEN_DIM, ParameterConfig.GAT_HIDDEN_DIM, HEAD_NUM,
                                              self.n_classes, emb_cfg, node_vec_stg=node_vec_stg,device = device).to(device)
        #OPCode
        self.config1 = Config(emb_ast, hidden_dim, device)
        self.model3 = Model(self.config1).to(device)
        #self.model3 = nn.LSTM(in_dim, hidden_dim, batch_first=True) #Model(self.config1).to(device)
        #self.model3 = nn.TransformerEncoderLayer(d_model=in_dim, nhead=2)
        self.out_2 = nn.Linear(4, 2)
        self.out_3 = nn.Linear(6, 2)
        self.fc1 = nn.Linear(ParameterConfig.GAT_HIDDEN_DIM, 32)
        self.fc2 = nn.Linear(32, 2)
        self.fd1 = nn.Linear(ParameterConfig.GAT_HIDDEN_DIM, 32)
        self.fd2 = nn.Linear(32, 2)
        self.factor = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        self.probabilities = [[],[],[],[],[],[],[]]
    def factor_sum(self,labels,out,num):
        for i in range(len(labels)):
            if out[i] == labels[i]:
                if out[i] == 0:
                    self.factor[num][0] += 1
                else:
                    self.factor[num][3] += 1
            else:
                if out[i] == 0:
                    self.factor[num][1] += 1
                else:
                    self.factor[num][2] += 1
    def factor_max(self,num):
        m = 0
        t = []
        for type in self.factor:
            m = max(self.factor[type][num],m)
        for type in self.factor:
            if self.factor[type][num] == m:
                t.append(type)
        return t
    def forward(self,op, edges_c,edges_d,labels,flag):
        #op.long()
        #out1, _ = self.model3(nodes)
        out1 = self.model3(op)
        #out1 = out1[:, -1, :]
        out1 = self.fc1(out1)
        out1 = self.fc2(out1)
        out2 = self.model2(edges_c)        
        out2 = self.fc1(out2)
        out2 = self.fc2(out2)
        out3 = self.model1(edges_d)
        out3 = self.fc1(out3)
        out3 = self.fc2(out3)
        out12 = th.cat([out1,out2], dim=1)
        out12 = self.out_2(out12)
        out13 = th.cat([out1,out3], dim=1)
        out13 = self.out_2(out13)
        out23 = th.cat([out2,out3], dim=1)
        out23 = self.out_2(out23)
        out123 = th.cat([out1,out2,out3], dim=1)
        out123 = self.out_3(out123)
        Out = [out1,out2,out3,out12,out13,out23,out123]
        if flag == ParameterConfig.EPOCHES-1:
            with th.no_grad():  
                i = 0      
                for result in Out:
                    pred = np.argmax(result.detach().cpu().numpy(), axis=-1).tolist()
                    self.factor_sum(labels,pred,i)
                    i += 1
        elif flag == -1:
            with th.no_grad():
                j = 0
                for result in Out:
                    probability = []
                    for i in range(len(labels)):
                        probability.append(F.softmax(result[i], dim=0).tolist())
                    self.probabilities[j].extend(probability)
                    j += 1
        ps = []
        for i in range(len(labels)):
            ps.append([])
        with th.no_grad():
            for result in Out:
                for i in range(len(labels)):
                    ps[i].extend(F.softmax(result[i], dim=0).tolist())
        return Out,ps
    
class S_Model(nn.Module):
    def __init__(self, in_dim, hidden_dim, HEAD_NUM, emb_ast, n_classes, tokens_size, emb_cfg,
                 device=None, layer_num=2,activation=F.relu):
        super(S_Model, self).__init__()
        self.n_classes = n_classes
        self.device = device
        # self.embedding_matrix = embedding_matrix
        self.layers = nn.ModuleList()
        self.emb_ast = emb_ast
        node_vec_stg = 'mean'
        self.model1 = OptimizedGATClassifier(ParameterConfig.GAT_HIDDEN_DIM, ParameterConfig.GAT_HIDDEN_DIM, HEAD_NUM, # 
                                              self.n_classes, emb_cfg, node_vec_stg=node_vec_stg).to(device)
        self.model2 = OptimizedGATClassifier(ParameterConfig.GAT_HIDDEN_DIM, ParameterConfig.GAT_HIDDEN_DIM, HEAD_NUM,
                                              self.n_classes, emb_cfg, node_vec_stg=node_vec_stg,device = device).to(device)
        self.model3 = BatchProgramClassifier(embedding_dim=len(emb_ast[0]), hidden_dim=100, vocab_size=tokens_size+1, encode_dim=128,
                                             label_size=ParameterConfig.GAT_HIDDEN_DIM, batch_size=ParameterConfig.BATCH_SIZE,
                                             use_gpu=True, pretrained_weight=emb_ast).to(device)
        self.out_2 = nn.Linear(4, 2)
        self.out_3 = nn.Linear(6, 2)

        self.fc1 = nn.Linear(ParameterConfig.GAT_HIDDEN_DIM, 32)
        self.fc2 = nn.Linear(32, 2)

        self.fd1 = nn.Linear(ParameterConfig.GAT_HIDDEN_DIM, 32)
        self.fd2 = nn.Linear(32, 2)

        self.factor = [[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
        self.probabilities = [[],[],[],[],[],[],[]]
   
    def factor_sum(self,labels,out,num):
        # 00 01 10 11
        for i in range(len(labels)):
            if out[i] == labels[i]:
                if out[i] == 0:
                    self.factor[num][0] += 1
                else:
                    self.factor[num][3] += 1
            else:
                if out[i] == 0:
                    self.factor[num][1] += 1
                else:
                    self.factor[num][2] += 1

    def factor_max(self,num):
        m = 0
        t = []
        for type in self.factor:
            m = max(self.factor[type][num],m)
        for type in self.factor:
            if self.factor[type][num] == m:
                t.append(type)
        return t


    def forward(self,nodes,edges_c,edges_d, labels, flag):
        out1 = self.model1(edges_d)
        out1 = self.fc1(out1)
        out1 = self.fc2(out1)
        out2 = self.model2(edges_c)        
        out2 = self.fc1(out2)
        out2 = self.fc2(out2)
        out3 = self.model3(nodes)
        out3 = self.fc1(out3)
        out3 = self.fc2(out3)
        out12 = th.cat([out1,out2], dim=1)
        out12 = self.out_2(out12)
        out13 = th.cat([out1,out3], dim=1)
        out13 = self.out_2(out13)
        out23 = th.cat([out2,out3], dim=1)
        out23 = self.out_2(out23)
        out123 = th.cat([out1,out2,out3], dim=1)
        out123 = self.out_3(out123)
        Out = [out1,out2,out3,out12,out13,out23,out123]
        if flag == ParameterConfig.EPOCHES-1:
            with th.no_grad():  
                i = 0      
                for result in Out:
                    pred = np.argmax(result.detach().cpu().numpy(), axis=-1).tolist()
                    self.factor_sum(labels,pred,i)
                    i += 1
        elif flag == -1:
            with th.no_grad():
                j = 0
                for result in Out:
                    probability = []
                    for i in range(len(labels)):
                        probability.append(F.softmax(result[i], dim=0).tolist())
                    self.probabilities[j].extend(probability)
                    j += 1
        ps = []
        for i in range(len(labels)):
            ps.append([])
        with th.no_grad():
            for result in Out:
                for i in range(len(labels)):
                    ps[i].extend(F.softmax(result[i], dim=0).tolist())
            return Out,ps
        for i in range(len(Out)):
            ps.append([])
        with th.no_grad():
            j = 0
            for result in Out:
                probability = []
                for i in range(len(labels)):
                    probability.append(F.softmax(result[i], dim=0).tolist())
                ps[j].extend(probability)
                j += 1
        #self.factor_sum(labels,cd,"cd")
        #c = np.argmax(c, axis=-1).tolist()
        #d = np.argmax(d, axis=-1).tolist()
        #cd = np.argmax(cd, axis=-1).tolist()#用于返回一个numpy数组中最大值的索引值
        with th.no_grad(): 
            tfnp = []
            for i in range(4):
                tfnp.append(self.factor_max(i))
            l = []
            for num in range(len(labels)):
                if c[num][0] > c[num][1]: #00 01
                    ii = 0
                    i = (c[num][0] - c[num][1])#*(self.factor["c"][0]- self.factor["c"][1])# 
                else:
                    ii = 1
                    i = (c[num][1] - c[num][0])#*(self.factor["c"][3] - self.factor["c"][2])#
                if d[num][0] > d[num][1]: #00 01
                    jj = 0
                    j = (d[num][0] - d[num][1])#*(self.factor["d"][0] - self.factor["d"][1])#
                else:
                    jj = 1
                    j = (d[num][1] - d[num][0])#*(self.factor["d"][3] - self.factor["d"][2])#
                if cd[num][0] > cd[num][1]: #00 01
                    kk = 0
                    k = (cd[num][0] - cd[num][1])#*(self.factor["cd"][0] - self.factor["cd"][1])#
                else:
                    kk = 1
                    k = (cd[num][1] - cd[num][0])*(self.factor["cd"][3] - self.factor["cd"][2])#
                
                if i>k:
                    if i>j:
                        l.append(ii)
                    else:
                        l.append(jj)
                elif k>j:
                    l.append(kk)
                else:
                    l.append(jj)
            print(labels,c,d)