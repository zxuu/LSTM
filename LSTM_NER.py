import torch
import json
from torch import nn
import torch.utils.data as Data

# 自实现lstm(单层)

texts, labels = [], []
unique_text, unique_label = [], []
for line in open('D:\project\\vscode\RNN_Proj\data\\train_data3.txt','r',encoding='utf-8'):
    json_data = json.loads(line)
    texts.append(json_data['text'])
    labels.append(json_data['labels'])
    unique_text.extend(json_data['text'])
    unique_label.extend(json_data['labels'])

unique_text = list(set(unique_text))    # vocab
unique_label = list(set(unique_label))    # unique_label
# print('len(unique_text),len(unique_label)', len(unique_text), len(unique_label))

vocab = unique_text
word2id = {v:k for k,v in enumerate(sorted(vocab))}
id2word = {k:v for k,v in enumerate(sorted(vocab))}
label2id = {v:k for k,v in enumerate(sorted(unique_label))}
id2label = {k:v for k,v in enumerate(sorted(unique_label))}
texts_id = [[word2id[v] for k,v in enumerate(text)] for text in texts]
labels_id = [[label2id[v] for k,v in enumerate(label)] for label in labels]
length_vocab = len(vocab)
length_labels = len(unique_label)
lenght_texts = len(texts)

in_dim = 512
hidden_dim = length_labels
epoch = 10
batch_size = 1

word_embedding = nn.Embedding(length_vocab, in_dim)
# torch.rand, torch.randn
# word_embedding = torch.rand(length_vocab, in_dim)   # 输入的词向量是正态分布
# one-hot编码输入
# word_embedding = torch.eye(length_vocab)

class MyDataSet(Data.Dataset):
    def __init__(self, texts, labels):
        super(MyDataSet, self).__init__()
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, index):
        return self.texts[index], self.labels[index]
loader = Data.DataLoader(MyDataSet(texts=texts_id, labels=labels_id), batch_size=1)

# RNNCell
class LSTM_Cell(nn.Module):
    def __init__(self, input_sz, hidden_sz):
        """
        in_dim: 输入字符的向量的维度
        hidden_dim: 隐层维度
        """
        super(LSTM_Cell, self).__init__()
        self.input_size=input_sz
        self.hidden_size=hidden_sz

        #input_gate
        self.U_i=nn.Parameter(torch.rand(input_sz,hidden_sz))
        self.V_i = nn.Parameter(torch.rand(hidden_sz,hidden_sz))
        self.b_i = nn.Parameter(torch.rand(1,hidden_sz))

        #forget_gate
        self.U_f = nn.Parameter(torch.rand(input_sz, hidden_sz))
        self.V_f = nn.Parameter(torch.rand(hidden_sz, hidden_sz))
        self.b_f = nn.Parameter(torch.rand(1,hidden_sz))
    
        #c_t
        self.U_c = nn.Parameter(torch.rand(input_sz, hidden_sz))
        self.V_c = nn.Parameter(torch.rand(hidden_sz, hidden_sz))
        self.b_c = nn.Parameter(torch.rand(1,hidden_sz))
    
        #output_gate
        self.U_o = nn.Parameter(torch.rand(input_sz, hidden_sz))
        self.V_o = nn.Parameter(torch.rand(hidden_sz, hidden_sz))
        self.b_o = nn.Parameter(torch.rand(1,hidden_sz))

    def forward(self, x_t, c_t, h_t):
        i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
        f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
        g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)
        o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
        c_t = f_t * c_t + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return c_t, h_t

class LSTM(torch.nn.Module):
    def __init__(self, input_size, hidden_size, batch_size):
        """
        input_size: 字符嵌入向量维度
        hidden_size: 隐层维度
        batch_size: 
        """
        super(LSTM, self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.rnncell1 = LSTM_Cell(input_sz=self.input_size, hidden_sz=self.hidden_size)

    def forward(self, x_t, c_t, h_t):
        c_t, h_t = self.rnncell1(x_t, c_t, h_t)
        return c_t, h_t

    def init_hidden(self):
        return torch.zeros(self.batch_size, self.hidden_size),torch.zeros(self.batch_size, self.hidden_size)
    

net = LSTM(input_size=in_dim, hidden_size=hidden_dim, batch_size=batch_size)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.1)

for i in range(epoch):
    net.train()
    right = []    # 记录全部句子的准确率
    loss_epoch = 0
    for texts_idd, labels_idd in loader:    # 一句话
        texts_idd = texts_idd
        labels_idd = labels_idd
        texts_embedding = word_embedding(torch.tensor(texts_idd))    # [seq_len, in_dim]
        # texts_embedding = Embedding(torch.tensor(texts_idd))
        # texts_embedding = zero_mat[torch.tensor(texts_idd)]
        labels_y = torch.tensor(labels_idd).view(-1, 1)    # [seq_len, 1]
        loss = 0
        c_t, h_t = net.init_hidden()
        is_right = [0]    # 记录一个句子的准确度
        for input, label in zip(texts_embedding, labels_y):  #一句话中的每个词 inputs：seg_len * batch_size * input_size；labels：[seg_len]
            # input.cuda()
            # hidden.cuda()
            # label.cuda()
            input = input.unsqueeze(0)
            optimizer.zero_grad()
            c_t, h_t = net.forward(x_t=input, c_t=c_t, h_t=h_t)
            loss = loss + criterion(h_t, label)  # 要把每个字母的loss累加    =([1,4], [1])
            _, idx = h_t.max(dim=1)
            # 记录预测是否正确
            # if label.item()!=26:
            is_right.extend([1 if label.item()==idx.item() else 0])
        sentence_acc = sum(is_right)/len(is_right)
        loss_epoch = loss_epoch + loss
        right.extend([sentence_acc])
        loss.backward()
        optimizer.step()
    print('all_sentence acc:%.4f, loss:%.4f' % (sum(right)/len(right), loss_epoch/len(right)))