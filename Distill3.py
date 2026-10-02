'''import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertModel, BertTokenizer

# 教师模型
class TeacherModel(nn.Module):
    def __init__(self, bytecode_feature_size, hidden_size, num_classes):
        super(TeacherModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.requires_grad = False  # 可以选择冻结BERT参数
        self.bytecode_fc = nn.Linear(bytecode_feature_size, hidden_size)
        self.fc = nn.Linear(768 + hidden_size, num_classes)  # 768是BERT输出的隐藏层维度

    def forward(self, src_text, bytecode_features):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        inputs = tokenizer(src_text, return_tensors='pt', padding=True, truncation=True)
        inputs = {k: v.to(self.bert.device) for k, v in inputs.items()}
        bert_output = self.bert(**inputs).last_hidden_state.mean(dim=1)  # 取平均池化后的特征
        bytecode_output = torch.relu(self.bytecode_fc(bytecode_features))
        combined = torch.cat((bert_output, bytecode_output), dim=1)
        output = self.fc(combined)
        return output

# 学生模型
class StudentModel(nn.Module):
    def __init__(self, bytecode_feature_size, hidden_size, num_classes):
        super(StudentModel, self).__init__()
        self.bytecode_fc = nn.Linear(bytecode_feature_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, bytecode_features):
        bytecode_output = torch.relu(self.bytecode_fc(bytecode_features))
        output = self.fc(bytecode_output)
        return output

def distillation_loss(student_output, teacher_output, labels, alpha=0.5, temperature=2):
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(student_output / temperature, dim=1),
        nn.functional.softmax(teacher_output / temperature, dim=1)
    ) * (alpha * temperature * temperature)
    hard_loss = nn.CrossEntropyLoss()(student_output, labels) * (1 - alpha)
    return soft_loss + hard_loss

# 初始化模型
bytecode_feature_size = 80
hidden_size = 64
num_classes = 2
teacher_model = TeacherModel(bytecode_feature_size, hidden_size, num_classes)
student_model = StudentModel(bytecode_feature_size, hidden_size, num_classes)

# 定义优化器
teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=0.001)
student_optimizer = optim.Adam(student_model.parameters(), lr=0.001)

# 模拟训练数据
batch_size = 32
src_texts = [" ".join([str(x) for x in torch.randn(10)]) for _ in range(batch_size)]  # 模拟源代码文本
bytecode_features = torch.randn(batch_size, bytecode_feature_size)
labels = torch.randint(0, num_classes, (batch_size,))

# 训练教师模型
teacher_model.train()
teacher_optimizer.zero_grad()
teacher_output = teacher_model(src_texts, bytecode_features)
teacher_loss = nn.CrossEntropyLoss()(teacher_output, labels)
teacher_loss.backward()
teacher_optimizer.step()

# 训练学生模型
student_model.train()
student_optimizer.zero_grad()
student_output = student_model(bytecode_features)
with torch.no_grad():
    teacher_output = teacher_model(src_texts, bytecode_features)
student_loss = distillation_loss(student_output, teacher_output, labels)
student_loss.backward()
student_optimizer.step()

print(f"Teacher Loss: {teacher_loss.item()}")
print(f"Student Loss: {student_loss.item()}")

# 模拟测试数据
test_batch_size = 16
test_src_texts = [" ".join([str(x) for x in torch.randn(10)]) for _ in range(test_batch_size)]
test_bytecode_features = torch.randn(test_batch_size, bytecode_feature_size)
test_labels = torch.randint(0, num_classes, (test_batch_size,))

# 测试教师模型
teacher_model.eval()
with torch.no_grad():
    teacher_test_output = teacher_model(test_src_texts, test_bytecode_features)
    teacher_preds = torch.argmax(teacher_test_output, dim=1)
    teacher_accuracy = (teacher_preds == test_labels).float().mean()

# 测试学生模型
student_model.eval()
with torch.no_grad():
    student_test_output = student_model(test_bytecode_features)
    student_preds = torch.argmax(student_test_output, dim=1)
    student_accuracy = (student_preds == test_labels).float().mean()

print(f"Teacher Model Test Accuracy: {teacher_accuracy.item()}")
print(f"Student Model Test Accuracy: {student_accuracy.item()}")'''

import networkx as nx
edg = [[2, 4], [0, 4], [7, 8], [5, 8], [9, 12], [13, 15], [16, 47], [48, 53], [55, 60], [61, 64], [66, 71], [72, 75], [77, 82], [83, 86], [88, 91], [206, 209], [116, 118], [119, 122], [123, 126], [93, 95], [96, 99], [100, 103]]
'''G = nx.Graph()
G.add_edges_from(edg)
vulnerable= [7,16,66,83,209]
vuln = set()
for ss in nx.connected_components(G):
    for s in ss:
        print(s)
        if s in vulnerable :
            vuln = vuln.union(ss)
            print(vuln)
            break
edges = []
for e in edg:
    if e[0] in vuln :
        edges.append(e)
print(edges)'''

'''import dgl
gc = dgl.graph(edg)
gc = dgl.add_self_loop(gc)
print(gc)'''


import dgl
import torch
import networkx as nx
import numpy as np

# 生成多个示例图
def create_graph():
    G = nx.erdos_renyi_graph(10, 0.3)  # 10 个节点，0.3 概率连边
    g = dgl.from_networkx(G)  # 转换为 DGLGraph
    g.ndata["feat"] = torch.rand((g.num_nodes(), 10))  # 生成 10 维特征
    return g

graphs = [create_graph() for _ in range(5)]  # 生成 5 个图
print(graphs)

import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_dim, out_dim):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, hidden_dim, "mean")
        self.conv2 = SAGEConv(hidden_dim, out_dim, "mean")

    def forward(self, g, x):
        x = F.relu(self.conv1(g, x))
        x = self.conv2(g, x)
        return x.mean(dim=0)  # 池化得到整个图的向量表示

# 初始化模型
model = GraphSAGE(in_feats=10, hidden_dim=100, out_dim=256)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练
for epoch in range(100):
    total_loss = 0
    for g in graphs:
        x = g.ndata["feat"]
        optimizer.zero_grad()
        graph_emb = model(g, x)  # 获取图的嵌入
        loss = graph_emb.norm()  # 这里用简单的 L2 约束（实际任务可使用对比学习等）
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

graph_embeddings = []
for g in graphs:
    with torch.no_grad():
        graph_embeddings.append(model(g, g.ndata["feat"]).numpy())




import torch
import torch.nn as nn
import torch.nn.functional as F

class Graph2VecModel(nn.Module):
    def __init__(self, embedding_dim=256, hidden_dim=100, dropout=0.5):
        super(Graph2VecModel, self).__init__()
        self.embedding_layer = nn.Linear(embedding_dim, embedding_dim)
        self.graph_weight = nn.Linear(embedding_dim, 1)
        self.hidden_layer = nn.Linear(embedding_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 2)  # 改为 2 维
        self.dropout = nn.Dropout(dropout)

    def forward(self, input2):
        graph2vec = F.relu(self.embedding_layer(input2))  # (None, 256)
        graphweight = torch.sigmoid(self.graph_weight(graph2vec))  # (None, 1)
        newgraphvec = graph2vec * graphweight  # (None, 256)
        x_final = F.relu(self.hidden_layer(newgraphvec))  # (None, 100)
        x_final = self.dropout(x_final)  # (None, 100)
        output = self.output_layer(x_final)  # (None, 2)
        prediction = F.softmax(output, dim=1)  # 进行 softmax 归一化
        return prediction

# 示例用法
model = Graph2VecModel(embedding_dim=256, hidden_dim=100, dropout=0.5)
input2 = torch.tensor(graph_embeddings)  # Batch size = 10, feature size = 256
output = model(input2)
print(output.shape)  # 预期输出: (10, 2)
print(output)  # 观察 softmax 归一化后的概率分布

