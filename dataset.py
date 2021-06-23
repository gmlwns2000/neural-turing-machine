import numpy as np
from numpy.core.numeric import zeros_like
import torch

class CopyGenerator:
    def __init__(self, batch_size=64):
        self.len = 16
        self.dim = 8
        self.batch_size = batch_size

    def gen(self):
        return np.random.randint(0, 2, size=(self.len, self.batch_size, self.dim-2))
    
    def batch(self):
        data = self.gen()
        input = np.zeros((self.len*2 + 2, self.batch_size, self.dim))
        input[0,:,0] = 1
        input[1:self.len+1, :, 2:] = data
        input[self.len+1, :, 1] = 1
        output = np.zeros((self.len*2 + 2, self.batch_size, self.dim))
        output[self.len+2:, :, 2:] = data
        return torch.Tensor(input), torch.Tensor(output)

class Dataset:
    def __init__(self, task = 'copy', batch_size=64):
        self.task = task
        self.gen_copy = CopyGenerator(batch_size=batch_size)
    
    def batch(self):
        if self.task == 'copy':
            return self.gen_copy.batch()
        raise Exception()