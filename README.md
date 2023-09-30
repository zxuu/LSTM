# LSTM
实现LSTM内部结构
> **框架图**

![lstm框架图](./img/LSTM%E6%A1%86%E6%9E%B6.png)
![框架图2](./img/%E6%A1%86%E6%9E%B6%E5%9B%BE2.png)
>**多层结构**

![多层结构](./img/%E5%A4%9A%E5%B1%82.webp)
[3层RNN](https://github.com/zxuu/RNN/blob/main/3/rnn_cell/NER/RNN_Cell.py)
### 在用LSTM做NER任务中数据集长这样(LSTM_NER.py)
**resume.json数据集**
```bash
{"text": "吴重阳，中国国籍，大学本科，教授级高工，享受国务院特殊津贴，历任邮电部侯马电缆厂仪表试制组长、光缆分厂副厂长、研究所副所长，获得过山西省科技先进工作者、邮电部成绩优异高级工程师等多种荣誉称号。", "labels": ["B-NAME", "I-NAME", "I-NAME", "O", "B-CONT", "I-CONT", "I-CONT", "I-CONT", "O", "B-EDU", "I-EDU", "I-EDU", "I-EDU", "O", "B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "I-ORG", "B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "O", "B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "O", "B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O"]}
{"text": "历任公司副总经理、总工程师，", "labels": ["O", "O", "B-ORG", "I-ORG", "B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "O", "B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "O"]}
{"text": "2009年5月至今，受聘为公司首席资深技术顾问；", "labels": ["O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "O", "B-ORG", "I-ORG", "B-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "I-TITLE", "O"]}
```
**train_data3.txt**
```bash
{'text': ['俄', '罗', '斯', '天', '然', '气', '工', '业', '股', '份', '公', '司', '（', 'G', 'a', 'z', 'p', 'r', 'o', 'm', '，', '下', '称', '俄', '气', '）', '宣', '布', '于', '4', '月', '2', '7', '日', '停', '止', '对', '波', '兰', '和', '保', '加', '利', '亚', '的', '天', '然', '气', '供', '应', '。'], 'labels': ['B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'B-TIM', 'I-TIM', 'I-TIM', 'I-TIM', 'I-TIM', 'O', 'O', 'O', 'B-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'O', 'O', 'O', 'O', 'O', 'O']}
{'text': ['据', 'I', 'T', '之', '家', '消', '息', '，', '台', '湾', '经', '济', '日', '报', '报', '道', '，', '业', '界', '人', '士', '称', '，', '苹', '果', '携', '手', '电', '子', '纸', '（', 'e', 'P', 'a', 'p', 'e', 'r', '）', '龙', '头', '企', '业', '元', '太', '开', '发', '新', '款', 'i', 'P', 'h', 'o', 'n', 'e', '。'], 'labels': ['B-TIM', 'I-TIM', 'B-COU', 'I-COU', 'I-ORG', 'B-LOC', 'O', 'O', 'B-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'B-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-ORG', 'B-COU', 'I-COU', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']}
{'text': ['5', '月', '8', '日', '，', '俄', '罗', '斯', '总', '统', '普', '京', '发', '表', '致', '辞', '，', '向', '白', '俄', '罗', '斯', '、', '亚', '美', '尼', '亚', '、', '摩', '尔', '多', '瓦', '、', '哈', '萨', '克', '斯', '坦', '、', '吉', '尔', '吉', '斯', '斯', '坦', '、', '塔', '吉', '克', '斯', '坦', '、', '土', '库', '曼', '斯', '坦', '、', '乌', '兹', '别', '克', '斯', '坦', '等', '国', '领', '导', '人', '致', '贺', '电', '，', '并', '向', '上', '述', '多', '国', '民', '众', '以', '及', '格', '鲁', '吉', '亚', '和', '乌', '克', '兰', '民', '众', '表', '示', '祝', '贺', '。'], 'labels': ['B-TIM', 'I-TIM', 'I-TIM', 'I-TIM', 'O', 'B-COU', 'I-COU', 'I-COU', 'O', 'O', 'B-PER', 'I-PER', 'O', 'O', 'O', 'O', 'O', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-COU', 'I-COU', 'I-COU', 'I-COU', 'O', 'B-COU', 'I-COU', 'I-COU', 'O', 'O', 'O', 'O', 'O', 'O', 'O']}
```
```bash
BiLSTM_NER_Pytorch.py
.......
Accuracy:0.9990,Precision:0.9995,Recall:0.9974,f1 score:0.9985
epoch: 39
Accuracy:0.9990,Precision:0.9995,Recall:0.9974,f1 score:0.9985
epoch: 40
Accuracy:0.9990,Precision:0.9995,Recall:0.9977,f1 score:0.9986
epoch: 41
Accuracy:0.9990,Precision:0.9995,Recall:0.9976,f1 score:0.9986
epoch: 42
Accuracy:0.9990,Precision:0.9995,Recall:0.9976,f1 score:0.9986
```

----
**自实现双向的LSTM下次继续更新**