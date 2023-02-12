import torch
from torchvision import models
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F

from opacus.validators import ModuleValidator
from opacus.layers import DPLSTM
import numpy as np

class pretrained_fc_layer_3(nn.Module):
    def __init__(self, input_dim, hidden_1_dim, hidden_2_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_1_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_1_dim, hidden_2_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_2_dim, output_dim)
        self.relu3 = nn.ReLU()
    
    def forward(self, data):
        out = self.fc1(data)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        return out

class CNN_layer_4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            # 二维卷积
            torch.nn.Conv2d(in_channels=3,# 输入图片的通道数
                            out_channels=16,# 卷积产生的通道数
                            kernel_size=3,# 卷积核尺寸
                            stride=2,# 步长,默认1
                            padding=1),# 补0数，默认0
            # 数据在进入ReLU前进行归一化处理，num_features=batch_size*num_features*height*width
            # 先分别计算每个通道的方差和标准差，然后通过公式更新矩阵中每个值，即每个像素。相关参数：调整方差，调整均值
            # 输出期望输出的(N,C,W,H)中的C (数量，通道，高度，宽度)
            # 实际上，该层输入输出的shape是相同的
            torch.nn.BatchNorm2d(16),
            # 设置该层的激活函数RELU()
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            # torch.nn.BatchNorm2d(16),
            torch.nn.Conv2d(16,32,3,2,1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.conv3 = torch.nn.Sequential(
            # torch.nn.BatchNorm2d(32),
            torch.nn.Conv2d(32,64,3,2,1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        self.conv4 = torch.nn.Sequential(
            # torch.nn.BatchNorm2d(64),
            torch.nn.Conv2d(64,64,2,2,0),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        # 全连接层参数设置
        self.mlp1 = torch.nn.Linear(2*2*64,100)# 为了输出y=xA^T+b,进行线性变换（输入样本大小，输出样本大小）
        self.mlp2 = torch.nn.Linear(100,10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.mlp1(x.view(x.size(0),-1))# 将多维度的tensor展平成一维
        x = self.mlp2(x)
        return x
        
def get_pretrained_resnet18(device, result_dim=1000, sample_dim=10):
    resnet_model = models.resnet18(num_classes=result_dim, pretrained=True)
    fc_model = pretrained_fc_layer_3(result_dim, 256, sample_dim)
    model = nn.Sequential(resnet_model, fc_model)
    model = model.to(device)

    return model

def get_resnet18(device, LR, num_classes):
    model = models.resnet18(num_classes=num_classes)
    errors = ModuleValidator.validate(model, strict=False)
    errors[-5:]

    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=LR)

    print("Finished Load Model!")
    return model, criterion, optimizer

def get_cnn(device, LR, num_classes):
    model = CNN_layer_4()
    errors = ModuleValidator.validate(model, strict=False)
    errors[-5:]
    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    return model, criterion, optimizer

def get_resnet50(device, LR, num_classes):
    model = models.resnet50(num_classes=num_classes)
    errors = ModuleValidator.validate(model, strict=False)
    errors[-5:]
    model = ModuleValidator.fix(model)
    ModuleValidator.validate(model, strict=False)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=LR)

    print("Finished Load Model!")
    return model, criterion, optimizer

class CharNNClassifier(nn.Module):
    def __init__(
        self,
        embedding_size,
        hidden_size,
        output_size,
        vocab_size,
        num_lstm_layers=1,
        bidirectional=False,
    ):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = DPLSTM(
            embedding_size,
            hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.out_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)  # -> [B, T, D]
        x, _ = self.lstm(x, hidden)  # -> [B, T, H]
        x = x[:, -1, :]  # -> [B, H]
        x = self.out_layer(x)  # -> [B, C]
        return x

def get_LSTM(device, LR, label_num,
            vocab_size, embedding_size, hidden_size, 
            n_lstm_layers, bidirectional_lstm):
    model = CharNNClassifier(
        embedding_size,
        hidden_size,
        label_num,
        vocab_size,
        n_lstm_layers,
        bidirectional_lstm,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    print("Finished Load Model!")
    return model, criterion, optimizer

def r_squared(predictions, labels):
    """Returns R^2 values for given predictions on labels.
    Args:
    predictions: Matrix where each row consists of predictions from one model.
    labels: Vector of labels.
    Returns:
    Vector of length len(predictions) containing an R^2 value for each model.
    """
    sum_squared_residuals = np.sum(np.square(predictions - labels), axis=1)
    total_sum_squares = np.sum(np.square(labels - np.mean(labels)))
    return 1 - np.divide(sum_squared_residuals, total_sum_squares)

def r_squared_from_models(models, features, labels):
    """Returns 0.25, 0.5, and 0.75 quantiles of R^2 values for given models.
    Args:
    models: Matrix where each row consists of a model.
    features: Matrix where each row consists of one data point.
    labels: Column vector of labels.
    """
    predictions = np.matmul(models, features.T)
    r2_vals = r_squared(predictions, labels)
    return np.quantile(
            r2_vals, 0.25, axis=0), np.quantile(
                r2_vals, 0.5, axis=0), np.quantile(
                    r2_vals, 0.75, axis=0)

def adassp(features, labels, epsilon, delta, rho=0.05):
    """Returns model computed using AdaSSP DP linear regression.
    Args:
    features: Matrix of feature vectors. Assumed to have intercept feature.
    labels: Vector of labels.
    epsilon: Computed model satisfies (epsilon, delta)-DP.
    delta: Computed model satisfies (epsilon, delta)-DP.
    rho: Failure probability. The default of 0.05 is the one used in
        https://arxiv.org/pdf/1803.02596.pdf.
    Returns:
    Vector of regression coefficients. AdaSSP is described in Algorithm 2 of
    https://arxiv.org/pdf/1803.02596.pdf.
    """
    _, d = features.shape
    # these bounds are data-dependent and not dp
    bound_x = np.amax(np.linalg.norm(features, axis=1))
    bound_y = np.amax(np.abs(labels))
    lambda_min = max(0,
                    np.amin(np.linalg.eigvals(np.matmul(features.T, features))))
    z = np.random.normal(size=1)
    sensitivity = np.sqrt(np.log(6 / delta)) / (epsilon / 3)
    private_lambda = max(
        0, lambda_min + sensitivity * (bound_x**2) * z -
        (bound_x**2) * np.log(6 / delta) / (epsilon / 3))
    final_lambda = max(
        0,
        np.sqrt(d * np.log(6 / delta) * np.log(2 * (d**2) / rho)) * (bound_x**2) /
        (epsilon / 3) - private_lambda)
    # generate symmetric noise_matrix where each upper entry is iid N(0,1)
    noise_matrix = np.random.normal(size=(d, d))
    noise_matrix = np.triu(noise_matrix)
    noise_matrix = noise_matrix + noise_matrix.T - np.diag(np.diag(noise_matrix))
    priv_xx = np.matmul(features.T,
                        features) + sensitivity * (bound_x**2) * noise_matrix
    priv_xy = np.dot(features.T, labels).flatten(
    ) + sensitivity * bound_x * bound_y * np.random.normal(size=d)
    model_adassp = np.matmul(
        np.linalg.pinv(priv_xx + final_lambda * np.eye(d)), priv_xy)
    return model_adassp

def run_adassp(features, labels, epsilon, delta, num_trials):
    """Returns 0.25, 0.5, and 0.75 quantiles from num_trials AdaSSP models.
    Args:
    features: Matrix of feature vectors. Assumed to have intercept feature.
    labels: Vector of labels.
    epsilon: Each DP model satisfies (epsilon, delta)-DP.
    delta: Each DP model satisfies (epsilon, delta)-DP.
    num_trials: Number of trials to run.
    """
    models = np.zeros((num_trials, len(features[0])))
    for trial in range(num_trials):
        models[trial, :] = adassp(features, labels, epsilon, delta)
    return r_squared_from_models(models, features, labels)

def get_DNN(device, LR, input_dim, hidden_1_dim, hidden_2_dim, output_dim):
    model = pretrained_fc_layer_3(
        input_dim,
        hidden_1_dim,
        hidden_2_dim,
        output_dim
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print("Finished Load Model!")
    return model, criterion, optimizer

class PBS_LSTM(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, opacus_flag):
        super(PBS_LSTM, self).__init__()
        
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        # Create Embeddings
        # Word to Vector
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # Define LSTM Model
        # nn.LSTM(input, hidden, num_hidden_layers, dropout, batch_first=True)
        if opacus_flag:
            self.lstm = DPLSTM(
                embedding_dim,
                hidden_dim,
                num_layers=n_layers,
                bidirectional=False,
                batch_first=True,
            )
        else: 
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, batch_first=True)
        # Dropout (Deactivate some neurons randomly)
        # self.dropout = nn.Dropout(drop_prob)
        # Define the Fully Connected Layers
        self.fc = nn.Linear(hidden_dim, output_size)
        # Activation Function
        # self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
        
    def forward(self, x, h0, c0):
        # batch_size = x.size(0) # Rows
        # x = x.long()
        # print("check x: {}".format(x.shape))
        embeds = self.embedding(x)
        # print("check embeds: {}".format(embeds.shape))
        lstm_out, hidden = self.lstm(embeds, (h0, c0))
        
        # print("check lstm_out: {}".format(lstm_out.shape))
        # print("check lstm_out: {}".format(lstm_out[:, -1: :].shape))
        out = self.fc(lstm_out[:, -1: :])
        # print("check out: {}".format(out.shape))
        out = self.relu(out).squeeze(1)  
        
        # print("check out: {}".format(out.shape))
        return out, hidden
    
class PBS_FF(nn.Module):
    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim1, hidden_dim2):
        super(PBS_FF, self).__init__()
        self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, output_size)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        embeds = self.embedding(x)
        # print("check embeds: {}; embeds.mean(1).squeeze(1): {}".format(embeds.shape, embeds.mean(1).shape))
        out = self.fc1(embeds.mean(1)) # embeds[:, -1: :].squeeze(1)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        out = self.relu3(out)
        return out

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """
        super(focal_loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes   # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            # print(" --- Focal_loss alpha = {}, 将对每一类权重进行精细化赋值 --- ".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha<1   #如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            # print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]

        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1) # log_softmax
        preds_softmax = torch.exp(preds_logsoft)    # softmax

        preds_softmax = preds_softmax.gather(1,labels.view(-1,1))   # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss

def get_PBS_LSTM(device, LR, vocab_size, label_distributions, embedding_dim, hidden_dim, n_layers, opacus_flag):
    num_classes = len(label_distributions)
    list_label_distribution = []
    for i in sorted(label_distributions): 
        print((i, label_distributions[i]), end =" ")
        list_label_distribution.append(1 - label_distributions[i])

    model = PBS_LSTM(vocab_size, num_classes, embedding_dim, hidden_dim, n_layers, opacus_flag).to(device)

    criterion = focal_loss(alpha=list_label_distribution, gamma=2, num_classes=num_classes) # nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print("Finished Load Model!")
    return model, criterion, optimizer

def get_PBS_FF(device, LR, vocab_size, label_distributions, embedding_dim, hidden_dim1, hidden_dim2):
    num_classes = len(label_distributions)
    list_label_distribution = []
    for i in sorted(label_distributions): 
        print((i, label_distributions[i]), end =" ")
        list_label_distribution.append(1 - label_distributions[i])
    model = PBS_FF(vocab_size, num_classes, embedding_dim, hidden_dim1, hidden_dim2).to(device)

    criterion = focal_loss(alpha=list_label_distribution, gamma=2, num_classes=num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    print("Finished Load Model!")
    return model, criterion, optimizer

def get_pretained_Bert(device):
    print("nothing happen!")
    return None

# if __name__ == "__main__":
#     device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
#     model = get_pretained_Bert(device)
#     texts = [
#         "Replace me by any text you'd like.",
#         "Beijing Shanghai Guangzhou Shenzhen."
#     ]
#     SEQUENCE_LEN = 10
    
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     encoded_input = tokenizer(texts, return_tensors='pt', padding=True).to(device)
#     output = model(**encoded_input)
#     print(output.last_hidden_state.shape)