import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

# 对比学习损失函数
def contrastive_loss(x_student, x_teacher, temperature=0.1):
    cos_sim = F.cosine_similarity(x_student, x_teacher, dim=-1)
    # 计算对比损失
    loss = -torch.log(torch.exp(cos_sim / temperature) / torch.sum(torch.exp(cos_sim / temperature)))
    return loss.mean()
class GATModel(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=4)
        self.conv2 = GATConv(8 * 4, out_channels)
    def forward(self, x, edge_index):
        attn1 = self.conv1(x, edge_index, return_attention_weights=True)[1][1]  # 获取注意力权重
        x = F.relu(self.conv1(x, edge_index))
        attn2 = self.conv2(x, edge_index, return_attention_weights=True)[1][1]
        x = self.conv2(x, edge_index)
        return x, attn1, attn2
def structure_loss(adj_student, adj_teacher):
    return F.mse_loss(adj_student, adj_teacher)
def compute_distill_loss(student_output, teacher_output, student_attn, teacher_attn, 
                         student_adj, teacher_adj, alpha=1.0, beta=1.0, gamma=1.0):
    L_feature = F.mse_loss(student_output, teacher_output)
    L_structure = structure_loss(student_adj, teacher_adj)
    L_attention = F.mse_loss(student_attn[0], teacher_attn[0]) + F.mse_loss(student_attn[1], teacher_attn[1])
    
    # 总损失
    L_total = alpha * L_feature + beta * L_structure + gamma * L_attention
    return L_total
# 模拟数据
num_nodes = 100
in_channels = 16
out_channels = 8
edge_index = torch.randint(0, num_nodes, (2, 300))  # DFG 结构
x_teacher = torch.randn(num_nodes, in_channels)
x_student = torch.randn(num_nodes, in_channels)

# 初始化模型
teacher_model = GATModel(in_channels, out_channels)
student_model = GATModel(in_channels, out_channels)

# 计算前向传播
teacher_output, teacher_attn1, teacher_attn2 = teacher_model(x_teacher, edge_index)
student_output, student_attn1, student_attn2 = student_model(x_student, edge_index)
teacher_adj = torch.zeros(num_nodes, num_nodes)
student_adj = torch.zeros(num_nodes, num_nodes)
teacher_adj[edge_index[0], edge_index[1]] = 1
student_adj[edge_index[0], edge_index[1]] = 1
loss = compute_distill_loss(student_output, teacher_output, 
                            (student_attn1, student_attn2), 
                            (teacher_attn1, teacher_attn2), 
                            student_adj, teacher_adj, 
                            alpha=1.0, beta=0.5, gamma=0.5)
print(f"Distillation Loss: {loss.item()}")
