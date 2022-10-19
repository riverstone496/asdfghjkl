import torch
class Encoder(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_size, 1000)
        self.fc2 = torch.nn.Linear(1000, 500)
        self.fc3 = torch.nn.Linear(500, 250)
        self.fc4 = torch.nn.Linear(250, 30)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class Decoder(torch.nn.Module):
    def __init__(self, output_size):
        super().__init__()
        self.fc1 = torch.nn.Linear(30, 250)
        self.fc2 = torch.nn.Linear(250, 500)
        self.fc3 = torch.nn.Linear(500, 1000)
        self.fc4 = torch.nn.Linear(1000, output_size)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))  # -1～1に変換
        return x

class AutoEncoder(torch.nn.Module):
    def __init__(self, img_size):
        super().__init__()
        self.enc = Encoder(img_size*img_size)
        self.dec = Decoder(img_size*img_size)
        self.img_size=img_size
    def forward(self, x):
        x = x.view([-1, self.img_size*self.img_size])
        x = self.enc(x)  # エンコード
        x = self.dec(x)  # デコード
        return x