from torch.nn.modules.container import ModuleList
from dataset import Dataset
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter
from torch.cuda.amp import autocast, GradScaler

BATCH_SIZE=512

class Memory(nn.Module):
    def __init__(self, batch_size=BATCH_SIZE, size=256, dim=32):
        super().__init__()
        self.batch_size = batch_size
        self.size = size
        self.dim = dim

        self.device = torch.device('cuda')
        self.reset()
    
    def reset(self):
        self.memory = torch.Tensor().new_full((self.batch_size, self.size, self.dim), 1e-6).to(self.device)
    
    def read(self, weight):
        weight = weight.unsqueeze(1)
        readed = torch.matmul(weight, self.memory).squeeze(1)
        return readed
        #TODO: confirm this
    
    def write(self, weight, erase, add):
        erased = self.memory * (1-weight.unsqueeze(2)*erase.unsqueeze(1))
        self.memory = erased + (weight.unsqueeze(2)*add.unsqueeze(1))

class Bias(nn.Module):
    def __init__(self, bias):
        super().__init__()
        self.bias = bias
    
    def forward(self, x):
        return self.bias + x

class MemoryHead(nn.Module):
    def __init__(self, mode, memory:Memory, controller_dim=128):
        super().__init__()

        self.mode = mode
        self.controller_dim=controller_dim
        self.memory = memory

        self.net_k = nn.Sequential(
            nn.Linear(controller_dim, self.memory.dim),
            nn.Tanh(),
        )
        self.net_beta = nn.Sequential(
            nn.Linear(controller_dim, 1),
            nn.Softplus(),
        )
        self.net_gate = nn.Sequential(
            nn.Linear(controller_dim, 1),
            nn.Sigmoid(),
        )
        self.net_shift = nn.Sequential(
            nn.Linear(controller_dim, 3),
            nn.Softmax(dim=1),
        )
        self.net_gamma = nn.Sequential(
            nn.Linear(controller_dim, 1),
            nn.Softplus(),
            Bias(1),
        )
        self.net_erase = nn.Sequential(
            nn.Linear(controller_dim, self.memory.dim),
            nn.Sigmoid(),
        )
        self.net_add = nn.Sequential(
            nn.Linear(controller_dim, self.memory.dim),
            nn.Sigmoid(),
        )

        self.device = torch.device('cuda')
        self.reset()

    def reset(self):
        self.prev_weight = torch.Tensor(np.zeros((self.memory.batch_size, self.memory.size), dtype=np.float32)).to(self.device)
    
    def content_addressing(self, key, beta):
        #key(N, M)
        #beta(N, 1)
        #memory(N, B, M)
        #sim(N, B)
        sim = F.cosine_similarity(self.memory.memory, key.unsqueeze(1), dim=2)
        sim *= beta
        weight = F.softmax(sim, dim=1)
        return weight
    
    def conv_shift(self, w, s):
        '''Returns the convolved weights
        Args:
            w (tensor): weights (batch_size, N)
            s (tensor): shift weights (batch_size, 2 * max_shift + 1)
        Returns:
            (tensor): convolved weights (batch_size, N)
        '''
        batch_size = w.size(0)
        max_shift = int((s.size(1) - 1) / 2)
        
        unrolled = torch.cat([w[:, -max_shift:], w, w[:, :max_shift]], 1)
        return F.conv1d(unrolled.unsqueeze(1), s.unsqueeze(1))[range(batch_size), range(batch_size)]

    def sharpen(self, w, gamma):
        w = w.pow(gamma)
        return torch.div(w, w.sum(1).view(-1, 1) + 1e-16)
    
    def forward(self, contoller_output):
        #print(contoller_output.shape, contoller_output.device)
        k = self.net_k(contoller_output)
        beta = self.net_beta(contoller_output)
        gate = self.net_gate(contoller_output)
        shift = self.net_shift(contoller_output)
        gamma = self.net_gamma(contoller_output)

        weight_c = self.content_addressing(k, beta)
        weight_g = gate * weight_c + (1-gate) * self.prev_weight
        weight_s = self.conv_shift(weight_g, shift)
        weight = self.sharpen(weight_s, gamma)
        self.prev_weight = weight

        if self.mode == 'r':
            readed = self.memory.read(weight)
            return readed
        elif self.mode == 'w':
            erase = self.net_erase(contoller_output)
            add = self.net_add(contoller_output)
            self.memory.write(weight, erase, add)
        else:
            raise Exception()

class Controller__(nn.Module):
    def __init__(self, input_dim=8, hidden_size=128, dropout=0.0, num_layers=3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, dropout=dropout, num_layers=num_layers)
        self.h_n = 0
        self.c_n = 0
    
    def reset(self):
        self.past_hidden = None
    
    def forward(self, x):
        #output(1, N, D*H_out) h_n(D*numlayers, N, H_out) c_n(D*numlayers, N, H_cell)
        x = x.unsqueeze(0)
        print(x.shape, x.device)
        if self.past_hidden is None:
            output, (h_n, c_n) = self.lstm(x)
        else:
            output, (h_n, c_n) = self.lstm(x, self.past_hidden)
        self.past_hidden = (h_n, c_n)
        
        assert len(output.shape) == 3
        assert output.shape[0] == 1

        output = output.squeeze(0)
        return output
class Controller(nn.Module):
    def __init__(self, input_dim=8, hidden_size=128, dropout=0.0, num_layers=3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_size = hidden_size

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
    
    def reset(self):
        self.past_hidden = None
    
    def forward(self, x):
        #output(1, N, D*H_out) h_n(D*numlayers, N, H_out) c_n(D*numlayers, N, H_cell)
        #print(x.shape)
        return self.net(x)

class Machine(nn.Module):
    def __init__(self, batch_size=BATCH_SIZE):
        super().__init__()
        
        self.num_heads = 2
        self.input_dim = 8
        self.output_dim = 8

        self.contoller = Controller(input_dim=self.input_dim)
        
        self.memory = Memory(batch_size=batch_size)

        self.memory_heads = ModuleList()
        for _ in range(self.num_heads):
            self.memory_heads.append(MemoryHead('r', self.memory))
            self.memory_heads.append(MemoryHead('w', self.memory))
        
        self.output = nn.Sequential(
            nn.Linear(self.num_heads*self.memory.dim, self.output_dim),
            nn.Sigmoid(),
        )
    
    def reset(self):
        self.memory.reset()
        self.contoller.reset()
        for head in self.memory_heads:
            head.reset()

    def forward(self, operation):
        controller_output = self.contoller(operation)
        reads = []
        for head in self.memory_heads:
            output = head(controller_output)
            if head.mode == 'r':
                reads.append(output)
        reads = torch.cat(reads, dim = 1)
        output = self.output(reads)
        return output

def main():
    device = torch.device('cuda')
    model = Machine()
    model = model.to(device)
    model.train()
    model.reset()

    criterion = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=0.0001)
    
    data = Dataset(batch_size=BATCH_SIZE)

    step = 0
    while True:
        step += 1

        model.reset()

        opt.zero_grad()
        batch_input, batch_output = data.batch()
        batch_input = batch_input.to(device)
        batch_output = batch_output.to(device)
        
        outputs = []
        for i in range(batch_input.shape[0]):
            input = batch_input[i]
            model_output = model(input)
            outputs.append(model_output)
        outputs = torch.stack(outputs, dim=0)

        #print(outputs.shape, batch_output.shape)
        loss = criterion(outputs, batch_output)
        loss.backward()
        print(step, loss.item())

        opt.step()

        if step % 100:
            torch.save({
                'model':model.state_dict(),
                'opt':opt.state_dict(),
                'step':step,
            }, 'model.pth')

if __name__ == '__main__':
    main()