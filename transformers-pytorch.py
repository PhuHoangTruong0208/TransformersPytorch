import torch
from torch import nn
import torch.nn.functional as F
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# chuyển các lô dữ liệu văn bản thành tokens
class Tokenizer(nn.Module):
    def __init__(self, x_batch, y_batch, hidden_dim, max_sequence_length=100, pad="<pad>", end="<end>", start="<start>", limit_total=10):
        super().__init__()
        self.x_batch = x_batch[:limit_total]
        self.y_batch = y_batch[:limit_total]
        self.y_compute_loss = y_batch[:limit_total]
        self.pad = pad
        self.end = end
        self.start = start
        self.max_sequence_length = max_sequence_length
        self.vocab = [c for c in list("".join(self.x_batch + self.y_batch))] + [pad, end, start]
        self.text_to_number = {v:k for k,v in enumerate(self.vocab)}
        self.number_to_text = {k:v for k,v in enumerate(self.vocab)}
        self.embedding = nn.Embedding(len(self.vocab), hidden_dim)
        self.__run__()
    
    def __run__(self):
        self.normalize_sentences()
        self.tokenizer()
        self.add_token()
        self.padding()
        self.normalize_2times()
        self.x_batch = torch.tensor(self.x_batch)
        self.y_batch = torch.tensor(self.y_batch)
        self.y_compute_loss = self.y_batch
        self.x_batch = self.embedding(self.x_batch)
        self.y_batch = self.embedding(self.y_batch)
    
    def normalize_sentences(self):
        for i in range(len(self.x_batch)):
            if len(list(self.x_batch[i])) > self.max_sequence_length:
                self.x_batch[i] = "".join(list(self.x_batch[i])[:self.max_sequence_length-3])
            if len(list(self.y_batch[i])) > self.max_sequence_length:
                self.y_batch[i] = "".join(list(self.y_batch[i])[:self.max_sequence_length-3])
    
    def tokenizer(self):
        x_batch_tokenize, y_batch_tokenize = [], []
        for i in range(len(self.x_batch)):
            x_batch_tokenize.append([self.text_to_number[token] for token in list(self.x_batch[i])])
            y_batch_tokenize.append([self.text_to_number[token] for token in list(self.y_batch[i])])
        self.x_batch = x_batch_tokenize
        self.y_batch = y_batch_tokenize
    
    def add_token(self):
        for i in range(len(self.x_batch)):
            self.x_batch[i] = self.x_batch[i] + [self.text_to_number[self.end]]
            self.y_batch[i] = [self.text_to_number[self.start]] + self.y_batch[i] + [self.text_to_number[self.end]]
    
    def padding(self):
        for i in range(len(self.x_batch)):
            paddding_tensor_x = []
            for _ in range(len(self.x_batch[i]), self.max_sequence_length):
                paddding_tensor_x.append(self.text_to_number[self.pad])
            self.x_batch[i] = self.x_batch[i] + paddding_tensor_x
            paddding_tensor_y = []
            for _ in range(len(self.y_batch[i]), self.max_sequence_length):
                paddding_tensor_y.append(self.text_to_number[self.pad])
            self.y_batch[i] = self.y_batch[i] + paddding_tensor_y
    
    def normalize_2times(self):
        for i in range(len(self.x_batch)):
            if len(self.x_batch[i]) > self.max_sequence_length:
                self.x_batch[i] = self.x_batch[i][:self.max_sequence_length]
            if len(self.y_batch[i]) > self.max_sequence_length:
                self.y_batch[i] = self.y_batch[i][:self.max_sequence_length]

# chuyển các vector tensor đầu ra thành danh sách từ
class TensorToWord(nn.Module):
    def __init__(self, tensor_input, tokenizer_class):
        super().__init__()
        self.tokenizer = tokenizer_class
        self.output = tensor_input
        self.choose_words()
        self.get_words()
    
    def choose_words(self):
        self.output = torch.argmax(self.output, dim=-1)
        
    def get_words(self):
        sentences = []
        for get_in_tensor1d in self.output:
            word_batch_last = []
            for word_number_type in get_in_tensor1d:
                word = self.tokenizer.number_to_text[int(word_number_type)]
                word_batch_last.append(word)
                if word == self.tokenizer.end:
                    break
            sentences.append("".join(word_batch_last))
        self.output = sentences

# mô hình transformers học sinh văn bản
class TransformerGEN(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=2, num_decoder_layers=2, batch_first=True):
        super().__init__()
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, batch_first=batch_first)
        self.linear_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, y):
        x = self.transformer(x, y).to(device)
        x = self.linear_out(x).to(device)
        return x

# x = ["xin chào", 'hello', "bạn tên là gì vậy?", "tôi tên human, rất vui được gặp", "tôi sống ở tokyo",
#      "cảm ơn, bạn hoạt động ổn không?", "à không, tôi chỉ hỏi thế thôi", "tất nhiên, nhưng có vài vấn đề với dự án coding",
#      "có rất nhiều lỗi :@", "vậy thôi tôi đi fix bug đây, tạm biệt, tối nay gặp"]

# y = ["chào ạ", "hi", "tôi tên là transformer còn bạn?", "rất vui được gặp, bạn sống ở đâu?", "chà bạn sống ở thành phố đẹp đấy",
#      "tôi ổn, có vấn đề gì sao?", "à vậy à, hôm nay mọi việc ổn chứ?", "ồ!? vấn đề gì thế?", "haha nghe có vẻ hơi mệt đấy",
#      "tạm biệt, hẹn gặp tối nay!"]

x, y = [], []
with open(file="data.txt", mode="r", encoding="utf-8") as file:
    data = file.read().splitlines()
for i in range(len(data)):
    i = i+1
    if i % 2 == 0:
        y.append(data[i-1])
    else:
        x.append(data[i-1])

tokenizer = Tokenizer(x, y, hidden_dim=512, limit_total=500, max_sequence_length=100)
transformers = TransformerGEN(vocab_size=len(tokenizer.vocab))

criterian = nn.CrossEntropyLoss(ignore_index=tokenizer.text_to_number[tokenizer.pad]).to(device)
optimizer = torch.optim.Adam(params=transformers.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    transformers.train()
    optimizer.zero_grad()
    predict = transformers(tokenizer.x_batch, tokenizer.y_batch)
    loss = criterian(predict.view(-1, len(tokenizer.vocab)).to(device), tokenizer.y_compute_loss.view(-1).to(device)).to(device)
    loss.backward(retain_graph=True)
    optimizer.step()
    print(f"Epoch {epoch} Loss {loss.item()}")

# words_batch = TensorToWord(transformers(tokenizer.x_batch, tokenizer.y_batch), tokenizer)
# print("lô đầu vào thực tế:")
# print(x)
# print()

# print("lô đầu ra thực tế:")
# print(y)
# print()

# print("lô dự đoán của mô hình:")
# print(words_batch.output)
# print()

while True:
    inp = input("bạn : ")
    tokenizer_chat = Tokenizer([inp], [tokenizer.start], hidden_dim=512)
    words_batch = TensorToWord(transformers(tokenizer_chat.x_batch, tokenizer_chat.y_batch), tokenizer)
    print("Bot: ", words_batch.output[0])