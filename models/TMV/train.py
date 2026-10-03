import os
import torch
import torch.optim as optim
from torch.autograd import Variable
#from torch.utils.data import DataLoader
from gnnmodels.TMV.model import TMC
#from data import Multi_view_data
import warnings
warnings.filterwarnings("ignore")
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Fusion():
    def __init__(self):
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch-size', type=int, default=200, metavar='N',
                            help='input batch size for training [default: 100]')
        parser.add_argument('--epochs', type=int, default=10, metavar='N',   #500
                            help='number of epochs to train [default: 500]')
        parser.add_argument('--lambda-epochs', type=int, default=1, metavar='N',
                            help='gradually increase the value of lambda from 0 to 1')
        parser.add_argument('--lr', type=float, default=0.0003, metavar='LR',
                            help='learning rate')
        args = parser.parse_args()
        

        args.dims = [[2], [2],[2]]
        self.correct_num = 0
        self.data_num = 0

        args.views = len(args.dims)
        self.model = TMC(2, args.views, args.dims, args.lambda_epochs) # 2是分类数
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=1e-5)
        self.model.cuda()

    def train_fusion(self,data, target,epoch):
        self.model.train()
        loss_meter = AverageMeter()
        #print(data, target)

        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].cuda()) 
        target = Variable(target.long().cuda())
        # refresh the optimizer
        self.optimizer.zero_grad()
        evidences, evidence_a, loss = self.model(data, target, epoch)
        loss.backward()
        self.optimizer.step()
        loss_meter.update(loss.item())

    def test_fusion(self,data, target,epoch):
        self.model.eval()
        loss_meter = AverageMeter()
        
        #print(data[0].size(), target.size())#200*240,200
        #print(data)
        for v_num in range(len(data)):
            data[v_num] = Variable(data[v_num].cuda())
        self.data_num += target.size(0) 
        with torch.no_grad():
            target = Variable(target.long().cuda())
            
            evidences, evidence_a, loss = self.model(data, target, epoch)
            _, predicted = torch.max(evidence_a.data, 1)
            self.correct_num += (predicted == target).sum().item()
            loss_meter.update(loss.item())

        print('====> acc: {:.4f}'.format(self.correct_num/self.data_num))



    

#    test_loss, acc = test(epoch)
#    print('====> acc: {:.4f}'.format(acc))
