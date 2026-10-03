import torch
import dgl
import math
import torch as th
import torch.nn as nn
import torch.optim as optim
from ParameterConfig import ParameterConfig
from torch.utils.data import DataLoader
import torch.optim as optim
from sklearn.tree import DecisionTreeClassifier
from gnnmodels.S_Model import S_Model
from gnnmodels.B_Model import B_Model
device = torch.device('cuda:0')
num_classes = 2
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def collate(samples):
    labels,opcode,edges_c,edges_d,ast,c,d  = map(list, zip(*samples))
    label = th.tensor(labels).to(device)
    edges_c = dgl.batch(edges_c)
    edges_d = dgl.batch(edges_d)
    c = dgl.batch(c).to(device)
    d = dgl.batch(d).to(device)
    opcode = th.tensor(opcode, dtype=th.float32).to(device)
    return label,opcode,edges_c,edges_d,ast,c,d
def collates(samples):
    labels, _,_,_,ast,edges_c,edges_d = map(list, zip(*samples))
    label = th.tensor(labels).to(device)
    edges_c = dgl.batch(edges_c).to(device)
    edges_d = dgl.batch(edges_d).to(device)
    return label, ast, edges_c, edges_d
def Smodel_train(trainset, model, loss_func, optimizer, scheduler, validationset=None):
    data_loader = DataLoader(trainset, batch_size=ParameterConfig.BATCH_SIZE, shuffle=True,
                             collate_fn=collates, pin_memory=ParameterConfig.PIN_MEM)
    modelFusion = DecisionTreeClassifier()
    for epoch in range(ParameterConfig.EPOCHES):
        trains(data_loader, loss_func, model, optimizer, epoch,modelFusion)
        scheduler.step()
    if validationset is not None:
        evaluate_loss_and_accs(model, validationset, loss_func,modelFusion)
def Bmodel_train(trainset, bmodel, smodel, validationset=None):
    loss_func = nn.CrossEntropyLoss().to(device)
    loss_kd = nn.KLDivLoss(reduction="batchmean")
    optimizer = optim.Adam(bmodel.parameters(), lr=ParameterConfig.lr, weight_decay=1e-4 )
    scheduler = optim.lr_scheduler.StepLR(optimizer, 2, gamma=ParameterConfig.lr_decay)
    data_loader = DataLoader(trainset, batch_size=ParameterConfig.BATCH_SIZE, shuffle=True,
                             collate_fn=collate, pin_memory=ParameterConfig.PIN_MEM)
    modelFusion = DecisionTreeClassifier()
    for epoch in range(ParameterConfig.EPOCHES):
        train(data_loader, loss_func, loss_kd ,bmodel, smodel,optimizer, epoch,modelFusion)
        scheduler.step()
    if validationset is not None:
        evaluate_loss_and_acc(bmodel, validationset, loss_func,modelFusion)
def compute_distill_loss(loss_func,loss_kd,S_prediction,B_prediction,labels,epoch):
    T_max = 10.0
    k = 0.05
    temperature = T_max * math.exp(-k * epoch)
    alpha_min, alpha_max = 0.1, 0.9
    alpha = alpha_min + (alpha_max - alpha_min) * (epoch / ParameterConfig.EPOCHES)
    out = nn.Linear(4, 2).to(device)
    loss = 0.0
    for i in range(7):
        loss_ce = loss_func(B_prediction[i], labels)
        #loss_k = loss_kd(F.log_softmax(B_prediction[i] / temperature, dim=1),F.softmax(S_prediction[i] / temperature, dim=1))
        #loss += alpha * loss_ce + (1 - alpha)*loss_k*(temperature ** 2)
        loss += alpha * loss_ce + (1 - alpha)*loss_func(out(th.cat([B_prediction[i],S_prediction[2]], dim=1)), labels)
    return loss
def compute_distill_loss_new(loss_func,loss_kd,S_prediction,B_prediction,labels,epoch):
    T_max = 10.0
    k = 0.05
    temperature = T_max * math.exp(-k * epoch)
    alpha_min, alpha_max = 0.1, 0.9
    alpha = alpha_min + (alpha_max - alpha_min) * (epoch / ParameterConfig.EPOCHES)
    '''losses = []
    losses2 = []
    for i in range(7):
        losses.append(loss_func(B_prediction[i], labels))
        losses2.append(loss_kd(F.log_softmax(B_prediction[i] / temperature, dim=1),F.softmax(S_prediction[i] / temperature, dim=1)))
    loss_ce = losses[0]+losses[1]+losses[2]+losses[3]+losses[4]+losses[5]+losses[6]
    loss_k = losses2[0]+losses2[1]+losses2[2]+losses2[3]+losses2[4]+losses2[5]+losses2[6]
    loss = alpha * loss_ce + (1 - alpha)*loss_k'''
    loss = 0.0
    for i in range(7):
        loss_ce = loss_func(B_prediction[i], labels)
        for j in range(7):
            loss_k = loss_kd(F.log_softmax(B_prediction[i] / temperature, dim=1),F.softmax(S_prediction[j] / temperature, dim=1))
        loss += alpha * loss_ce + (1 - alpha)*loss_k*(temperature ** 2)/7
    '''loss_ce = losses[0]+losses[1]+losses[2]+losses[3]+losses[4]+losses[5]+losses[6]
    loss_k = losses2[0]+losses2[1]+losses2[2]+losses2[3]+losses2[4]+losses2[5]+losses2[6]
    loss = alpha * loss_ce + (1 - alpha)*loss_k'''
    return loss
def train(data_loader, loss_func, loss_kd ,bmodel, smodel, optimizer,epoch,modelFusion):
    smodel.eval()  # 冻结教师模型
    bmodel.train()
    bmodel.is_train_mode = True
    all_predictions = [[],[],[],[],[],[],[]]
    all_labels = []
    outlen = 7
    for iter, (labels,o,c,d,a,cc,dd) in enumerate(data_loader):
        with torch.no_grad():
            S_prediction,_ = smodel(a,cc,dd, labels, -1)
        prediction,ps = bmodel(o,c,d, labels, epoch)
        # 结构蒸馏损失方法 2
        '''attention_weights_b = F.adaptive_avg_pool1d(attention_weights_b.unsqueeze(0).unsqueeze(1), output_size=128).squeeze(1).squeeze(0)
        attention_weights_s = F.adaptive_avg_pool1d(attention_weights_s.unsqueeze(0).unsqueeze(1), output_size=128).squeeze(1).squeeze(0)
        loss2 = F.mse_loss(attention_weights_b, attention_weights_s)'''
        # 特征(128)蒸馏损失方法
        '''criterion = nn.MSELoss()
        loss3 = criterion(tb,ts) #但可能要128长度#prediction[0], S_prediction[0]'''
        loss = compute_distill_loss(loss_func,loss_kd,S_prediction,prediction,labels,epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        labels = labels.detach().cpu().numpy().tolist()
        
        i = 0
        for out in prediction:
            num = out.detach().cpu().numpy()
            li = np.argmax(num, axis=-1).tolist()
            all_predictions[i].extend(li)
            i += 1
            num = num.tolist()

        all_labels.extend(labels)
        if epoch==ParameterConfig.EPOCHES -1 and sum(labels)!=0 and sum(labels)!=len(labels):#
            modelFusion.fit(ps,labels)

    for i in range(outlen):
        acc = accuracy_score(all_predictions[i], all_labels)
        prec = precision_score(all_predictions[i], all_labels)
        recall = recall_score(all_predictions[i], all_labels)
        f1 = f1_score(all_predictions[i], all_labels)
        print(acc,prec,recall,f1)
def trains(data_loader, loss_func, model, optimizer,epoch,modelFusion):
    model.train()
    model.is_train_mode = True
    all_predictions = [[],[],[],[],[],[],[]]
    all_labels = []
    outlen = 7
    for iter, (labels, a,c,d) in enumerate(data_loader):
        prediction,ps= model(a,c,d,labels, epoch)
        losses = []
        for out in prediction:
            losses.append(loss_func(out, labels))
        loss = losses[0]+losses[1]+losses[2]+losses[3]+losses[4]+losses[5]+losses[6]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        labels = labels.detach().cpu().numpy().tolist()
        i = 0
        X = [[]*len(labels)]
        for out in prediction:
            num = out.detach().cpu().numpy()
            li = np.argmax(num, axis=-1).tolist()
            all_predictions[i].extend(li)
            i += 1
            num = num.tolist()


        all_labels.extend(labels)
        if epoch==ParameterConfig.EPOCHES -1 and sum(labels)!=0 and sum(labels)!=len(labels):#
            modelFusion.fit(ps,labels)


    for i in range(outlen):
        acc = accuracy_score(all_predictions[i], all_labels)
        prec = precision_score(all_predictions[i], all_labels)
        recall = recall_score(all_predictions[i], all_labels)
        f1 = f1_score(all_predictions[i], all_labels)
        print([acc,prec,recall,f1])

def evaluate_loss_and_acc(model, dataset, loss_func,modelFusion):
    data_loader = DataLoader(dataset, batch_size=ParameterConfig.BATCH_SIZE, collate_fn=collate,
                             pin_memory=ParameterConfig.PIN_MEM)
    model.eval()
    model.is_train_mode = False
    total_loss = 0.
    with th.no_grad():
        all_predictions = [[],[],[],[],[],[],[]]
        pp = []
        all_labels = []
        outlen = 7
        for iter, (labels,o,c,d,_,_,_) in enumerate(data_loader):
            prediction ,ps = model(o,c,d, labels, -1)

            losses = []
            for out in prediction:
                losses.append(loss_func(out, labels))
            loss = losses[0]+losses[1]+losses[2]+losses[3]+losses[4]+losses[5]+losses[6]
            total_loss += loss.item()

            labels = labels.detach().cpu().numpy().tolist()
            all_labels.extend(labels)
            i = 0
            for out in prediction:
                num = out.detach().cpu().numpy()
                li = np.argmax(num, axis=-1).tolist()
                all_predictions[i].extend(li)
                i += 1
                num = num.tolist()
            try:
                pp.extend(modelFusion.predict(ps).tolist())
            except:
                1

        if len(pp)>0:
            acc = accuracy_score(pp, all_labels)
            f1 = f1_score(pp, all_labels)

        all_predictions = []
        for j in range(len(all_labels)):
            m = 0
            n = 0
            for i in range(outlen):
                m += model.probabilities[i][j][0]
                n += model.probabilities[i][j][1]
            if m>n:
                all_predictions.append(0)
            else:
                all_predictions.append(1)

        acc = accuracy_score(all_predictions, all_labels)
        f1 = f1_score(all_predictions, all_labels)
        prec = precision_score(all_predictions, all_labels)
        recall = recall_score(all_predictions, all_labels)
        print('Loss: %0.4f\tAccuracy: %0.5f\tPrecision: %0.5f\tRecall: %0.5f\tF1: %0.5f' % (loss, acc, prec, recall, f1))

def evaluate_loss_and_accs(model, dataset, loss_func,modelFusion):
    data_loader = DataLoader(dataset, batch_size=ParameterConfig.BATCH_SIZE, collate_fn=collates,
                             pin_memory=ParameterConfig.PIN_MEM)
    model.eval()
    model.is_train_mode = False
    total_loss = 0.
    with th.no_grad():
        all_predictions = [[],[],[],[],[],[],[]]
        pp = []
        all_labels = []
        outlen = 7
        for iter, (labels, a,c,d) in enumerate(data_loader):
            prediction ,ps = model(a,c,d, labels, -1)

            losses = []
            for out in prediction:
                losses.append(loss_func(out, labels))
            loss = losses[0]+losses[1]+losses[2]+losses[3]+losses[4]+losses[5]+losses[6]
            total_loss += loss.item()

            labels = labels.detach().cpu().numpy().tolist()
            all_labels.extend(labels)
            i = 0
            for out in prediction:
                num = out.detach().cpu().numpy()
                li = np.argmax(num, axis=-1).tolist()
                all_predictions[i].extend(li)
                i += 1
                num = num.tolist()
            pp.extend(modelFusion.predict(ps).tolist())
        if len(pp)>0:
            acc = accuracy_score(pp, all_labels)
            f1 = f1_score(pp, all_labels)

        all_predictions = []
        for j in range(len(all_labels)):
            m = 0
            n = 0
            for i in range(outlen):
                m += model.probabilities[i][j][0]
                n += model.probabilities[i][j][1]
                #pre0.append(model.probabilities[i][j][0]  - model.factor[i][1])#*()model.factor[i][0]
                #pre1.append(model.probabilities[i][j][1]  - model.factor[i][2])#*()model.factor[i][3]
            
            if m>n:#max(pre0) > max(pre1):
                all_predictions.append(0)
            else:
                all_predictions.append(1)
                
        acc = accuracy_score(all_predictions, all_labels)
        f1 = f1_score(all_predictions, all_labels)
        prec = precision_score(all_predictions, all_labels)
        recall = recall_score(all_predictions, all_labels)
        print('Loss: %0.4f\tAccuracy: %0.5f\tPrecision: %0.5f\tRecall: %0.5f\tF1: %0.5f' % (loss, acc, prec, recall, f1))


class Distillation:
    def __init__(self,embs,embb):
        tokens_size1 = len(embs) - 1
        tokens_size2 = len(embb) - 1
        self.smodel = S_Model(ParameterConfig.GAT_HIDDEN_DIM, ParameterConfig.GAT_HIDDEN_DIM,
                    ParameterConfig.HEAD_NUM, embs,num_classes, tokens_size1, embs,
                    device=device, layer_num=ParameterConfig.GAT_Layer_Num).to(device)
        self.bmodel = B_Model(ParameterConfig.GAT_HIDDEN_DIM, ParameterConfig.GAT_HIDDEN_DIM,
                    ParameterConfig.HEAD_NUM, embb,num_classes, tokens_size2, embb,
                    device=device, layer_num=ParameterConfig.GAT_Layer_Num).to(device)
        self.model = B_Model(ParameterConfig.GAT_HIDDEN_DIM, ParameterConfig.GAT_HIDDEN_DIM,
                    ParameterConfig.HEAD_NUM, embb,num_classes, tokens_size2, embb,
                    device=device, layer_num=ParameterConfig.GAT_Layer_Num).to(device)
    def model_(self,train_dataset, test_dataset):
        loss_func_s = nn.CrossEntropyLoss().to(device)
        optimizer_s = optim.Adam(self.smodel.parameters(), lr=ParameterConfig.lr, weight_decay=1e-4)
        scheduler_s = optim.lr_scheduler.StepLR(optimizer_s, 2, gamma=ParameterConfig.lr_decay)
        Smodel_train(train_dataset, self.smodel, loss_func_s, optimizer_s, scheduler_s, test_dataset)
        Bmodel_train(train_dataset, self.bmodel, self.smodel, test_dataset)
    def no_D(self, train_dataset, test_dataset):
        loss_func = nn.CrossEntropyLoss().to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=ParameterConfig.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 2, gamma=ParameterConfig.lr_decay)
        #model_train(self.logger,train_dataset, self.model, loss_func, optimizer, scheduler, test_dataset)