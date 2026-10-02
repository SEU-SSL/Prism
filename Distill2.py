'''import torch
import torch.nn as nn
import torch.optim as optim
# 超参数设置
bytecode_feature_size = 128  # 假设 Opcodes 特征维度为 5
hidden_size = 64
num_classes = 2
learning_rate = 0.001
epochs = 20

# 教师模型
class TeacherModel(nn.Module):
    def __init__(self, bytecode_feature_size, hidden_size, num_classes):
        super(TeacherModel, self).__init__()
        self.bytecode_fc = nn.Linear(bytecode_feature_size, hidden_size)
        self.src_fc = nn.Linear(bytecode_feature_size, hidden_size)
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, bytecode_features, src_features):
        # 确保输入张量为二维
        if bytecode_features.dim() == 1:
            bytecode_features = bytecode_features.unsqueeze(0)
        if src_features.dim() == 1:
            src_features = src_features.unsqueeze(0)

        bytecode_output = torch.relu(self.bytecode_fc(bytecode_features))
        src_output = torch.relu(self.src_fc(src_features))
        combined = torch.cat((bytecode_output, src_output), dim=1)
        output = self.fc(combined)
        return output


class StudentModel(nn.Module):
    def __init__(self, bytecode_feature_size, hidden_size, num_classes):
        super(StudentModel, self).__init__()
        self.bytecode_fc = nn.Linear(bytecode_feature_size, hidden_size)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, bytecode_features):
        # 确保输入张量为二维
        if bytecode_features.dim() == 1:
            bytecode_features = bytecode_features.unsqueeze(0)
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



def knowledge_model(emb_byte, emb_source, train_dataset, test_dataset):
    # 初始化模型
    teacher_model = TeacherModel(bytecode_feature_size, hidden_size, num_classes)
    student_model = StudentModel(bytecode_feature_size, hidden_size, num_classes)

    # 定义优化器
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=learning_rate)
    student_optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(epochs):
        teacher_model.train()
        student_model.train()
        running_teacher_loss = 0.0
        running_student_loss = 0.0
        for opcodes, ast_tokens, labels in train_dataset:
            try:
                # 将数据转换为张量
                bytecode_features = torch.tensor(emb_byte[opcodes[0]], dtype=torch.float32)
                src_features = torch.tensor(emb_source[ast_tokens[0]], dtype=torch.float32)
                labels = torch.tensor([labels], dtype=torch.long)  # 确保 labels 为二维

                # 教师模型训练
                teacher_optimizer.zero_grad()
                teacher_output = teacher_model(bytecode_features, src_features)
                teacher_loss = nn.CrossEntropyLoss()(teacher_output, labels)
                teacher_loss.backward()
                teacher_optimizer.step()
                running_teacher_loss += teacher_loss.item()

                # 学生模型训练
                student_optimizer.zero_grad()
                student_output = student_model(bytecode_features)
                with torch.no_grad():
                    teacher_output = teacher_model(bytecode_features, src_features)
                student_loss = distillation_loss(student_output, teacher_output, labels)
                student_loss.backward()
                student_optimizer.step()
                running_student_loss += student_loss.item()
            except Exception as e:
                print(f"Error during training at epoch {epoch}: {e}")

        print(f'Epoch {epoch + 1}, Teacher Loss: {running_teacher_loss / len(train_dataset)}, Student Loss: {running_student_loss / len(train_dataset)}')

    # 测试模型
    teacher_model.eval()
    student_model.eval()
    teacher_correct = 0
    student_correct = 0
    total = 0
    with torch.no_grad():
        for opcodes, ast_tokens, labels in test_dataset:
            try:
                # 确保 opcodes 和 ast_tokens 索引有效
                if opcodes[0] >= len(emb_byte) or ast_tokens[0] >= len(emb_source):
                    print(f"Invalid index: opcodes[0]={opcodes[0]}, ast_tokens[0]={ast_tokens[0]}")
                    continue

                bytecode_features = torch.tensor(emb_byte[opcodes[0]], dtype=torch.float32)
                src_features = torch.tensor(emb_source[ast_tokens[0]], dtype=torch.float32)
                labels = torch.tensor([labels], dtype=torch.long)  # 确保 labels 为二维

                teacher_output = teacher_model(bytecode_features, src_features)
                student_output = student_model(bytecode_features)

                _, teacher_predicted = torch.max(teacher_output.data, 1)
                _, student_predicted = torch.max(student_output.data, 1)

                total += labels.size(0)
                teacher_correct += (teacher_predicted == labels).sum().item()
                student_correct += (student_predicted == labels).sum().item()
            except Exception as e:
                print(f"Error during testing: {e}")

    print(f'Teacher Accuracy: {100 * teacher_correct / total}%')
    print(f'Student Accuracy: {100 * student_correct / total}%')'''


import torch
import torch.nn as nn
import torch.optim as optim
from gnnmodels.BatchProgramClassifier import BatchProgramClassifier
from torch.utils.data import DataLoader

device = torch.device('cuda:0')
# 超参数设置
bytecode_feature_size = 128  # 假设 Opcodes 特征维度为 5
hidden_size = 64
num_classes = 2
learning_rate = 0.001
epochs = 50
alpha = 0.5  # 硬标签损失和软标签损失的权重
temperature = 2  # 温度参数

class StModel(nn.Module):
    def __init__(self, bytecode_feature_size, hidden_size, num_classes, num_layers=1):
        super(StModel, self).__init__()
        self.bytecode_fc = nn.Linear(bytecode_feature_size, hidden_size)
        # 添加 BiLSTM 层
        self.bilstm = nn.LSTM(input_size=hidden_size,
                              hidden_size=hidden_size,
                              num_layers=num_layers,
                              batch_first=True,
                              bidirectional=True)
        # 由于 BiLSTM 是双向的，输出维度会翻倍，所以输入维度为 2 * hidden_size
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, bytecode_features):
        # 确保输入张量为二维
        if bytecode_features.dim() == 1:
            bytecode_features = bytecode_features.unsqueeze(0)
        bytecode_output = torch.relu(self.bytecode_fc(bytecode_features))
        # 为了适应 LSTM 的输入要求，增加一个序列长度维度
        bytecode_output = bytecode_output.unsqueeze(1)
        # 通过 BiLSTM 层
        lstm_output, _ = self.bilstm(bytecode_output)
        # 取最后一个时间步的输出
        lstm_output = lstm_output[:, -1, :]
        output = self.fc(lstm_output)
        return output


class TeModel(nn.Module):
    def __init__(self, emb_ast, tokens_size, device=None):
        super(TeModel, self).__init__()
        self.n_classes = num_classes
        self.device = device
        # self.embedding_matrix = embedding_matrix
        self.layers = nn.ModuleList()  # 通过网络层拼接的方式来完成模型构建
        self.emb_ast = emb_ast

        self.model = BatchProgramClassifier(embedding_dim=len(emb_ast[0]), hidden_dim=100, vocab_size=tokens_size, encode_dim=128,
                                             label_size=128, batch_size=8, use_gpu=True, pretrained_weight=emb_ast).to(device)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 2)

    def forward(self, ast):
        out = self.model(ast)
        out = self.fc1(out)
        out = self.fc2(out)
        return out


def distillation_loss(student_output, teacher_output, labels, alpha=alpha, temperature=temperature):
    soft_loss = nn.KLDivLoss(reduction='batchmean')(
        nn.functional.log_softmax(student_output / temperature, dim=1),
        nn.functional.softmax(teacher_output / temperature, dim=1)
    ) * (alpha * temperature * temperature)
    hard_loss = nn.CrossEntropyLoss()(student_output, labels) * (1 - alpha)
    return soft_loss + hard_loss


def knowledge_model(emb_byte, emb_source, train_dataset, test_dataset):
    # 初始化教师模型和学生模型
    teacher_model = TeModel(emb_source, len(emb_source), device=device).to(device)
    student_model = StModel(bytecode_feature_size, hidden_size, num_classes).to(device)

    # 定义优化器
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=learning_rate)
    student_optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)

    def collate(samples):
        op, nodes, labels = map(list, zip(*samples))
        label = torch.tensor(labels).to(device)
        return op, nodes, label

    data_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate, pin_memory=False)

    # 训练教师模型
    for epoch in range(epochs):
        teacher_model.train()
        running_teacher_loss = 0.0
        for iter, (opcodes, AstTokens, labels) in enumerate(data_loader):
            teacher_optimizer.zero_grad()
            teacher_output = teacher_model(AstTokens)
            teacher_loss = nn.CrossEntropyLoss()(teacher_output, labels)
            teacher_loss.backward()
            teacher_optimizer.step()
            running_teacher_loss += teacher_loss.item()

        print(f'Epoch {epoch + 1}, Teacher Loss: {running_teacher_loss / len(data_loader)}')

    # 训练学生模型进行知识蒸馏
    for epoch in range(epochs):
        student_model.train()
        running_student_loss = 0.0
        for iter, (opcodes, AstTokens, labels) in enumerate(data_loader):
            student_optimizer.zero_grad()
            bytecode_features = torch.tensor([emb_byte[op][0] for op in opcodes], dtype=torch.float32).to(device)
            labels = labels.to(device)

            # 教师模型输出
            with torch.no_grad():
                teacher_output = teacher_model(AstTokens)

            # 学生模型输出
            student_output = student_model(bytecode_features)

            # 计算蒸馏损失
            student_loss = distillation_loss(student_output, teacher_output, labels)
            student_loss.backward()
            student_optimizer.step()
            running_student_loss += student_loss.item()

        print(f'Epoch {epoch + 1}, Student Loss: {running_student_loss / len(data_loader)}')

    # 测试学生模型
    student_model.eval()
    student_correct = 0
    total = 0
    with torch.no_grad():
        for opcodes, _, labels in test_dataset:
            bytecode_features = torch.tensor(emb_byte[opcodes[0]], dtype=torch.float32).to(device)
            labels = torch.tensor([labels], dtype=torch.long).to(device)

            student_output = student_model(bytecode_features)

            _, student_predicted = torch.max(student_output.data, 1)

            total += labels.size(0)
            student_correct += (student_predicted == labels).sum().item()

    print(f'Student Accuracy: {100 * student_correct / total}%')
