import torch
import torch.nn as nn
import math

class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize weights and biases
        # [W_ii|W_if|W_ig|W_io]
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        # [W_hi|W_hf|W_hg|W_ho]
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.zeros(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.zeros(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, input, state):
        # Input shape: (batch, input_size)
        # h, c shape: (batch, hidden_size)
        hx, cx = state

        gates = (torch.mm(input, self.weight_ih.t()) + self.bias_ih +
                torch.mm(hx, self.weight_hh.t()) + self.bias_hh)

        # Get all gates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        # Apply activations
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        # Compute new cell state and hidden state
        cy = (forgetgate * cx) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)

        return hy, cy

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # Create a list of LSTM cells
        self.cells = nn.ModuleList([
            LSTMCell(input_size if layer == 0 else hidden_size, hidden_size)
            for layer in range(num_layers)
        ])

    def forward(self, input, initial_states=None):
        # Input shape: (batch, seq_len, input_size) if batch_first=True
        #             (seq_len, batch, input_size) if batch_first=False

        if self.batch_first:
            batch_size, seq_len, _ = input.size()
            input = input.transpose(0, 1)  # Convert to (seq_len, batch, input_size)
        else:
            seq_len, batch_size, _ = input.size()

        if initial_states is None:
            h_states = [torch.zeros(batch_size, self.hidden_size,
                                device=input.device) for _ in range(self.num_layers)]
            c_states = [torch.zeros(batch_size, self.hidden_size,
                                device=input.device) for _ in range(self.num_layers)]
        else:
            h_states, c_states = initial_states

        output = []

        # Process each time step
        for t in range(seq_len):
            inner_input = input[t]

            # Process each layer
            for layer in range(self.num_layers):
                h_states[layer], c_states[layer] = self.cells[layer](
                    inner_input, (h_states[layer], c_states[layer])
                )
                inner_input = h_states[layer]

            output.append(h_states[-1])

        # Stack output sequence
        output = torch.stack(output)

        if self.batch_first:
            output = output.transpose(0, 1)  # Convert back to batch_first format

        return output, (h_states, c_states)

# Example usage:
if __name__ == "__main__":
    # Create a sample input
    batch_size = 32
    seq_length = 10
    input_size = 50
    hidden_size = 100
    num_layers = 2

    # Initialize the LSTM
    lstm = LSTM(input_size, hidden_size, num_layers, batch_first=True)

    # Create random input
    x = torch.randn(batch_size, seq_length, input_size)

    # Forward pass
    output, (h_n, c_n) = lstm(x)

    print(f"Output shape: {output.shape}")  # Should be (batch_size, seq_length, hidden_size)
    print(f"Hidden state shape: {len(h_n)}, {h_n[0].shape}")  # Should be num_layers, (batch_size, hidden_size)
    print(f"Cell state shape: {len(c_n)}, {c_n[0].shape}")    # Should be num_layers, (batch_size, hidden_size)
