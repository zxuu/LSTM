# LSTM
实现LSTM内部结构
> **框架图**

![lstm框架图](./img/LSTM%E6%A1%86%E6%9E%B6.png)
![框架图2](./img/%E6%A1%86%E6%9E%B6%E5%9B%BE2.png)
>**多层结构**

![多层结构](./img/%E5%A4%9A%E5%B1%82.webp)
[3层RNN](https://github.com/zxuu/RNN/blob/main/3/rnn_cell/NER/RNN_Cell.py)
### 在用LSTM做NER任务中数据集长这样(LSTM_NER.py)
**940条数据，数据不算多**
```bash
{'text': ['俄', '罗', '斯', '天', '然', '气', '工', '业', '股', '份', '公', '司', '（', 'G', 'a', 'z', 'p', 'r', 'o', 'm', '，', '下', '称', '俄', '气', '）', '宣', '布', '于', '4', '月', '2', '7', '日', '停', '止', '对', '波', '兰', '和', '保', '加', '利', '亚', '的', '天', '然', '气', '供', '应', '。'], 'labels': ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'B-TIM', 'I-TIM', 'I-TIM', 'I-TIM', 'I-TIM', 'O', 'O', 'O', 'B-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'O', 'O', 'O', 'O', 'O', 'O']}
{'text': ['据', 'I', 'T', '之', '家', '消', '息', '，', '台', '湾', '经', '济', '日', '报', '报', '道', '，', '业', '界', '人', '士', '称', '，', '苹', '果', '携', '手', '电', '子', '纸', '（', 'e', 'P', 'a', 'p', 'e', 'r', '）', '龙', '头', '企', '业', '元', '太', '开', '发', '新', '款', 'i', 'P', 'h', 'o', 'n', 'e', '。'], 'labels': ['B-TIM', 'I-TIM', 'B-COU', 'I-COU', 'I-ORG', 'B-LOC', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'B-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-ORG', 'B-COU', 'I-COU', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']}
{'text': ['5', '月', '8', '日', '，', '俄', '罗', '斯', '总', '统', '普', '京', '发', '表', '致', '辞', '，', '向', '白', '俄', '罗', '斯', '、', '亚', '美', '尼', '亚', '、', '摩', '尔', '多', '瓦', '、', '哈', '萨', '克', '斯', '坦', '、', '吉', '尔', '吉', '斯', '斯', '坦', '、', '塔', '吉', '克', '斯', '坦', '、', '土', '库', '曼', '斯', '坦', '、', '乌', '兹', '别', '克', '斯', '坦', '等', '国', '领', '导', '人', '致', '贺', '电', '，', '并', '向', '上', '述', '多', '国', '民', '众', '以', '及', '格', '鲁', '吉', '亚', '和', '乌', '克', '兰', '民', '众', '表', '示', '祝', '贺', '。'], 'labels': ['B-TIM', 'I-TIM', 'I-TIM', 'I-TIM', 'O', 'B-COU', 'I-COU', 'I-COU', 'O', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'O', 'O', 'O', 'O', 'O', 'O', 'O']}
```
```bash
自实现1层的LSTM就比3层的RNN高了大约三个百分点
all_sentence acc:0.5989, loss:116.9872
all_sentence acc:0.5999, loss:116.8669
all_sentence acc:0.5999, loss:116.8870
all_sentence acc:0.6016, loss:116.7693

3层LSTM的精度LSTM_NER.py, 每层之间加了个线性层
all_sentence acc:0.6191, loss:129.7561
```

```bash
torch.nn.LSTM效果
epoch: 0
all_sentence acc:0.6178, loss:125.8694
epoch: 1
all_sentence acc:0.6192, loss:124.7826
......
epoch: 7
all_sentence acc:0.6192, loss:121.9417
epoch: 8
all_sentence acc:0.6192, loss:121.9417
epoch: 9
all_sentence acc:0.6192, loss:121.9417
```

```bash
双向torch.nn.LSTM效果。没有提升
epoch: 0
all_sentence acc:0.4625, loss:150.3097
epoch: 1
all_sentence acc:0.6192, loss:150.4354
epoch: 2
all_sentence acc:0.6192, loss:150.4009
epoch: 3
all_sentence acc:0.6192, loss:150.3566
epoch: 4
all_sentence acc:0.6192, loss:150.3579
epoch: 5
all_sentence acc:0.6192, loss:150.1663
epoch: 6
all_sentence acc:0.6192, loss:149.6976
epoch: 7
all_sentence acc:0.6192, loss:149.4714
epoch: 8
all_sentence acc:0.6192, loss:148.9310
epoch: 9
all_sentence acc:0.6192, loss:148.8260
```

----
**自实现双向的LSTM下次继续更新**