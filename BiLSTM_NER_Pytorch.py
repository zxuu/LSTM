import torch
import json
from torch import nn
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings("ignore")

texts, labels = [], []
unique_text, unique_label = [], []
sum_example = 0
for line in open('./data/resume.json','r',encoding='utf-8'):
    json_data = json.loads(line)
    string = json_data['text']
    string_split = [string[_] for _ in range(len(string))]
    texts.append(string_split)
    labels.append(json_data['labels'])
    unique_text.extend(string_split)
    unique_label.extend(json_data['labels'])
    sum_example += 1
    if sum_example==1000:
        break
unique_text = list(set(unique_text))    # vocab
unique_label = list(set(unique_label))    # unique_label

vocab = unique_text
word2id = {v:k for k,v in enumerate(sorted(vocab))}
id2word = {k:v for k,v in enumerate(sorted(vocab))}
label2id = {v:k for k,v in enumerate(sorted(unique_label))}
id2label = {k:v for k,v in enumerate(sorted(unique_label))}
texts_id = [[word2id[v] for k,v in enumerate(text)] for text in texts]    # [每句话中的每个字在字典中的ID]
labels_id = [[label2id[v] for k,v in enumerate(label)] for label in labels]
length_vocab = len(vocab)
length_labels = len(unique_label)
lenght_texts = len(texts)
print(lenght_texts)

in_dim = 50    # 输入维度(字嵌入维度)
hidden_dim = 128    # 隐层维度
epoch = 50
batch_size = 1    # batch=1就不用考虑句子不等长了
num_layers = 3
bidirectional = 2    # 双向LSTM

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

class MyModel(torch.nn.Module):
    def __init__(self, in_dim, hidden_size, num_layers, length_labels):
        """
        input_size: 字符嵌入向量维度
        hidden_size: 隐层维度
        """
        super(MyModel, self).__init__()

        self.net = nn.LSTM(input_size=in_dim, hidden_size=hidden_size,batch_first=True, num_layers=num_layers, bidirectional=True)
        self.linear = nn.Linear(hidden_size*2, length_labels)

    def forward(self, input):
        out, _ = self.net(input)
        out = self.linear(out)
        return out

    
device = torch.device('cuda')
net = MyModel(in_dim=in_dim, hidden_size=hidden_dim, num_layers=num_layers, length_labels=length_labels).to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for i in range(epoch):
    print('epoch:', i)
    net.train()
    loss = 0
    y_true = []
    y_pred = []
    i = 0
    for texts_idd, labels_idd in loader:    # 一句话
        texts_idd = texts_idd
        labels_idd = labels_idd
        texts_embedding = word_embedding(torch.tensor(texts_idd))    # [seq_len, in_dim]
        # texts_embedding = Embedding(torch.tensor(texts_idd))
        # texts_embedding = zero_mat[torch.tensor(texts_idd)]
        labels_y = torch.tensor(labels_idd).view(-1).to(device)    # [seq_len]
        input = texts_embedding.to(device)    # [seq_len, in_dim]
        
        output = net(input)    
        
        l = criterion(output, labels_y)  #
        _, idx = output.max(dim=1)

        y_true.extend(labels_y.data.cpu())
        y_pred.extend(idx.data.cpu())

        loss += l
        i += 1
        if i%32 == 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = 0
    print('Accuracy:'+"%.4f" % accuracy_score(y_true,y_pred)+',Precision:'+"%.4f" % precision_score(y_true,y_pred,average='macro')+',Recall:'+"%.4f" % recall_score(y_true, y_pred, average='macro')+',f1 score:'+"%.4f" % f1_score(y_true, y_pred, average='macro'))
    