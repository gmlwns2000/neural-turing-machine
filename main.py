import torch
from torch import nn, optim
import torch.nn.functional as F

class Memory(nn.Module):
    def __init__(self, memory_units, memory_unit_size):
        super(Memory, self).__init__()
        
        # N = No. of memory units (rows)
        self.n = memory_units
        # M = Size of each memory unit (cols)
        self.m = memory_unit_size
        
        # Define the memory matrix of shape (batch_size, N, M)
        self.memory = torch.zeros([1, self.n, self.m])
        
        # Layer to learn initial values for memory reset
        # self.memory_bias_fc = nn.Linear(1, self.n * self.m)
        
        # Reset/Initialize
        self.reset()
        
    def read(self, weights):
        '''Returns a read vector using the attention weights
        Args:
            weights (tensor): attention weights (batch_size, N)
        Returns:
            (tensor): read vector (batch_size, M)
        '''
        read_vec = torch.matmul(weights.unsqueeze(1), self.memory).squeeze(1)
        return read_vec
    
    def write(self, weights, erase_vec, add_vec):
        '''Erases and Writes a new memory matrix
        Args:
            weights (tensor): attention weights (batch_size, N)
            erase_vec (tensor): erase vector (batch_size, M)
            add_vec (tensor): add vector (batch_size, M)
        '''
        # Erase
        memory_erased = self.memory * (1 - weights.unsqueeze(2) * erase_vec.unsqueeze(1))
        # Add
        self.memory = memory_erased + (weights.unsqueeze(2) * add_vec.unsqueeze(1))
        
    def content_addressing(self, key, beta):
        '''Performs content addressing and returns the content_weights
        Args:
            key (tensor): key vector (batch_size, M)
            beta (tensor): key strength scalar (batch_size, 1)
        Returns:
            (tensor): content weights (batch_size, N)
        '''
        # Compare key with every location in memory using cosine similarity
        similarity_scores = F.cosine_similarity(key.unsqueeze(1), self.memory, dim=2)
        # Apply softmax over the product of beta and similarity scores
        content_weights = F.softmax(beta * similarity_scores, dim=1)
        
        return content_weights

    def reset(self, batch_size=1):
        '''Reset/initialize the memory'''
        # Parametric Initialization
        # in_data = torch.tensor([[0.]]) # dummy input
        # Generate initial memory values
        # memory_bias = torch.sigmoid(self.memory_bias_fc(in_data))
        # Push it to memory matrix
        # self.memory = memory_bias.view(self.n, self.m).repeat(batch_size, 1, 1)
        
        # Uniform Initialization of 1e-6
        self.memory = torch.Tensor().new_full((1, self.n, self.m), 1e-6)

class Head(nn.Module):
    def __init__(self, mode, ctrl_dim, memory_unit_size):
        super(Head, self).__init__()
        # Valid modes are 'r' and 'w' for reading and writing respectively
        self.mode = mode
        # Size of each memory vector (key size)
        self.m = memory_unit_size
        
        self.max_shift = 1
        
        # Linear Layers for converting controller output to addressing parameters
        self.key_fc = nn.Linear(ctrl_dim, self.m)
        self.key_strength_fc = nn.Linear(ctrl_dim, 1)
        self.interpolation_gate_fc = nn.Linear(ctrl_dim, 1)
        self.shift_weighting_fc = nn.Linear(ctrl_dim, 3)
        self.sharpen_factor_fc = nn.Linear(ctrl_dim, 1)
        self.erase_weight_fc = nn.Linear(ctrl_dim, self.m)
        self.add_data_fc = nn.Linear(ctrl_dim, self.m)
        
        # Reset
        self.reset()
        
    def forward(self, ctrl_state, prev_weights, memory):
        '''Extracts the parameters and returns the attention weights
        Args:
            ctrl_state (tensor): output vector from the controller (batch_size, ctrl_dim)
            prev_weights (tensor): previous attention weights (batch_size, N)
            memory (nn.Module): memory module
        '''
        
        # Extract the parameters from controller state
        key = torch.tanh(self.key_fc(ctrl_state))
        beta = F.softplus(self.key_strength_fc(ctrl_state))
        gate = torch.sigmoid(self.interpolation_gate_fc(ctrl_state))
        shift = F.softmax(self.shift_weighting_fc(ctrl_state), dim=1)
        gamma = 1 + F.softplus(self.sharpen_factor_fc(ctrl_state))
        erase = torch.sigmoid(self.erase_weight_fc(ctrl_state))
        add = torch.tanh(self.add_data_fc(ctrl_state))
        
        # ==== Addressing ====
        # Content-based addressing
        content_weights = memory.content_addressing(key, beta)
        
        # Location-based addressing 
        # Interpolation
        gated_weights = self._gated_interpolation(content_weights, prev_weights, gate)
        # Convolution
        shifted_weights = self._conv_shift(gated_weights, shift)
        # Sharpening
        weights = self._sharpen(shifted_weights, gamma)
        
        # ==== Read / Write Operation ====
        # Read
        if self.mode == 'r':
            read_vec = memory.read(weights)
        # Write
        elif self.mode == 'w':
            memory.write(weights, erase, add)
            read_vec = None
        else:
            raise ValueError("mode must be read ('r') or write ('w')")
        
        return weights, read_vec
    
    def _gated_interpolation(self, w, prev_w, g):
        '''Returns the interpolated weights between current and previous step's weights
        Args:
            w (tensor): weights (batch_size, N)
            prev_w (tensor): weights of previous timestep (batch_size, N)
            g (tensor): a scalar interpolation gate (batch_size, 1)
        Returns:
            (tensor): content weights (batch_size, N)
        '''
        return (g * w) + ((1 - g) * prev_w)
    
    def _conv_shift(self, w, s):
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
    
    def _sharpen(self, w, gamma):
        '''Returns the sharpened weights
        Args:
            w (tensor): weights (batch_size, N)
            gamma (tensor): gamma value for sharpening (batch_size, 1)
        Returns:
            (tensor): sharpened weights (batch_size, N)
        '''
        w = w.pow(gamma)
        return torch.div(w, w.sum(1).view(-1, 1) + 1e-16)
        
    def reset(self):
        '''Reset/initialize the head parameters'''
        # Weights
        nn.init.xavier_uniform_(self.key_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.key_strength_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.interpolation_gate_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.shift_weighting_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.sharpen_factor_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.add_data_fc.weight, gain=1.4)
        nn.init.xavier_uniform_(self.erase_weight_fc.weight, gain=1.4)
        
        # Biases
        nn.init.normal_(self.key_fc.bias, std=0.01)
        nn.init.normal_(self.key_strength_fc.bias, std=0.01)
        nn.init.normal_(self.interpolation_gate_fc.bias, std=0.01)
        nn.init.normal_(self.shift_weighting_fc.bias, std=0.01)
        nn.init.normal_(self.sharpen_factor_fc.bias, std=0.01)
        nn.init.normal_(self.add_data_fc.bias, std=0.01)
        nn.init.normal_(self.erase_weight_fc.bias, std=0.01)
        

class Controller(nn.Module):
    
    def __init__(self, input_dim, ctrl_dim, output_dim, read_data_size):
        super(Controller, self).__init__()
        
        self.input_size = input_dim
        self.ctrl_dim = ctrl_dim
        self.output_size = output_dim
        self.read_data_size = read_data_size
        
        # Controller neural network
        self.controller_net = nn.LSTMCell(input_dim, ctrl_dim)
        # Output neural network
        self.out_net = nn.Linear(read_data_size, output_dim)
        # Initialize the weights of output net
        nn.init.kaiming_uniform_(self.out_net.weight)
        
        # Learnable initial hidden and cell states
        self.h_state = torch.zeros([1, ctrl_dim])
        self.c_state = torch.zeros([1, ctrl_dim])
        # Layers to learn init values for controller hidden and cell states
        self.h_bias_fc = nn.Linear(1, ctrl_dim)
        self.c_bias_fc = nn.Linear(1, ctrl_dim)
        # Reset
        self.reset()
        
    def forward(self, x, prev_reads):
        '''Returns the hidden and cell states'''
        x = torch.cat([x, *prev_reads], dim=1)
        # Get hidden and cell states
        self.h_state, self.c_state = self.controller_net(x, (self.h_state, self.c_state))
        
        return self.h_state, self.c_state
    
    def output(self, reads):
        '''Returns the external output from the read vectors'''
        out_state = torch.cat([self.h_state, *reads], dim=1)
        # Compute output
        output = torch.sigmoid(self.out_net(out_state))
        
        return output
        
    def reset(self, batch_size=1):
        '''Reset/initialize the controller states'''
        # Dummy input
        in_data = torch.tensor([[0.]])
        # Hidden state
        h_bias = self.h_bias_fc(in_data)
        self.h_state = h_bias.repeat(batch_size, 1)
        # Cell state
        c_bias = self.c_bias_fc(in_data)
        self.c_state = c_bias.repeat(batch_size, 1)

class NTM(nn.Module):
    
    def __init__(self,
                 input_dim,
                 output_dim,
                 ctrl_dim,
                 memory_units,
                 memory_unit_size,
                 num_heads):
        super(NTM, self).__init__()
        
        # Create controller
        self.ctrl_dim = ctrl_dim
        self.controller = Controller(input_dim + num_heads * memory_unit_size,
                                     ctrl_dim, 
                                     output_dim, 
                                     ctrl_dim + num_heads * memory_unit_size)
        
        # Create memory
        self.memory = Memory(memory_units, memory_unit_size)
        self.memory_unit_size = memory_unit_size # M
        self.memory_units = memory_units # N
        
        # Create Heads
        self.num_heads = num_heads
        self.heads = nn.ModuleList([])
        for head in range(num_heads):
            self.heads += [
                Head('r', ctrl_dim, memory_unit_size),
                Head('w', ctrl_dim, memory_unit_size)
            ]
        
        # Init previous head weights and read vectors
        self.prev_head_weights = []
        self.prev_reads = []
        # Layers to initialize the weights and read vectors
        self.head_weights_fc = nn.Linear(1, self.memory_units)
        self.reads_fc = nn.Linear(1, self.memory_unit_size)
        
        self.reset()
        
        
    def forward(self, x):
        '''Returns the output of the Neural Turing Machine'''
        # Get controller states
        ctrl_hidden, ctrl_cell = self.controller(x, self.prev_reads)
        
        # Read, and Write
        reads = []
        head_weights = []
        
        for head, prev_head_weights in zip(self.heads, self.prev_head_weights):
            # Read
            if head.mode == 'r':
                weights, read_vec = head(ctrl_cell, prev_head_weights, self.memory)
                reads.append(read_vec)
            # Write    
            elif head.mode == 'w':
                weights, _ = head(ctrl_cell, prev_head_weights, self.memory)
            
            head_weights.append(weights)
        
        # Compute output
        output = self.controller.output(reads)
        
        self.prev_head_weights = head_weights
        self.prev_reads = reads
            
        return output
    
    
    def reset(self, batch_size=1):
        '''Reset/initialize NTM parameters'''
        # Reset memory and controller
        self.memory.reset(batch_size)
        self.controller.reset(batch_size)
        
        # Initialize previous head weights (attention vectors)
        self.prev_head_weights = []
        for i in range(len(self.heads)):
            # prev_weight = torch.zeros([batch_size, self.memory.n])
            prev_weight = F.softmax(self.head_weights_fc(torch.Tensor([[0.]])), dim=1)
            self.prev_head_weights.append(prev_weight)
        
        # Initialize previous read vectors
        self.prev_reads = []
        for i in range(self.num_heads):
            # prev_read = torch.zeros([batch_size, self.memory.m])
            # nn.init.kaiming_uniform_(prev_read)
            prev_read = self.reads_fc(torch.Tensor([[0.]]))
            self.prev_reads.append(prev_read)