# coding: utf-8

import torch
from torch.autograd import Variable
from torch import nn
from onmt.modules import GlobalAttention


class BatchNormConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding, activation=None):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_dim, out_dim,
                                kernel_size=kernel_size,
                                stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_dim, momentum=0.99, eps=1e-4)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)


class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes=[256, 128]):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList([nn.Linear(in_size, out_size)
                                     for (in_size, out_size) in zip(in_sizes, sizes)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        for linear in self.layers:
            inputs = self.dropout(self.relu(linear(inputs)))
        return inputs


class CBHG(nn.Module):
    """CBHG module: a recurrent neural network composed of:
        - 1-d convolution banks
        - Highway networks + residual connections
        - Bidirectional gated recurrent units
    """

    def __init__(self, in_dim, K=16, projections=[128, 128]):
        super(CBHG, self).__init__()
        self.in_dim = in_dim
        self.relu = nn.ReLU()
        self.conv1d_banks = nn.ModuleList(
            [BatchNormConv1d(in_dim, in_dim, kernel_size=k, stride=1,
                             padding=k // 2, activation=self.relu)
             for k in range(1, K + 1)])
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        in_sizes = [in_dim] + projections[:-1]
        activations = [self.relu] * len(projections)
        activations[-1] = None
        self.conv1d_projections = nn.ModuleList(
            [BatchNormConv1d(in_size, out_size, kernel_size=3, stride=1,
                             padding=1, activation=ac)
             for (in_size, out_size, ac) in zip(
                 in_sizes, projections, activations)])

        self.pre_highway = nn.Linear(projections[-1], in_dim)
        self.highways = nn.ModuleList(
            [Highway(in_dim, in_dim) for _ in range(4)])

        self.gru = nn.GRU(
            in_dim, in_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, inputs, input_lengths=None):
        x = inputs

        # Needed to perform conv1d on time-axis
        if x.size(-1) == self.in_dim:
            x = x.transpose(1, 2)

        T = x.size(-1)

        # [B, in_dim, T_in]
        for conv1d in self.conv1d_banks:
            x = conv1d(x)[:, :, :T]
        x = self.max_pool1d(x)[:, :, :T]

        for conv1d in self.conv1d_projections:
            x = conv1d(x)

        # [B, T_in, in_dim]
        # Back to the original shape
        x = x.transpose(1, 2)

        if x.size(-1) != self.in_dim:
            x = self.pre_highway(x)

        # Residual connection
        x += inputs
        for highway in self.highways:
            x = highway(x)

        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True)

        # [B, T_in, in_dim*2]
        outputs, state = self.gru(x)
        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True)

        return outputs


class Encoder(nn.Module):
    def __init__(self, in_dim):
        super(Encoder, self).__init__()
        self.prenet = Prenet(in_dim, sizes=[256, 128])
        self.cbhg = CBHG(128, K=16, projections=[128, 128])

    def forward(self, inputs, input_lengths=None):
        inputs = self.prenet(inputs)
        return self.cbhg(inputs, input_lengths)


class Decoder(nn.Module):
    def __init__(self, in_dim, r):
        super(Decoder, self).__init__()
        self.in_dim = in_dim
        self.r = r
        self.prenet = Prenet(in_dim * r, sizes=[256, 128])
        # [prenet_out + attention context]
        self.attention_rnn = nn.GRUCell(256 + 128, 256)
        self.attn = GlobalAttention(256, attn_type="mlp")
        self.decoder_rnns = nn.ModuleList(
            [nn.GRUCell(256, 256) for _ in range(2)])
        self.proj_to_mel = nn.Linear(256, in_dim * r)
        self.max_decoder_steps = 300

    def forward(self, encoder_outputs, inputs=None, mask=None):
        B = encoder_outputs.size(0)

        # Run greedy decoding if inputs is None
        greedy = inputs is None

        if inputs is not None:
            # Grouping multiple frames if necessary
            if inputs.size(-1) == self.in_dim:
                inputs = inputs.view(B, inputs.size(1) // self.r, -1)
            assert inputs.size(-1) == self.in_dim * self.r
            T_decoder = inputs.size(1)
        if mask is not None:
            self.attn.mask = mask

        # go frames
        initial_input = Variable(
            encoder_outputs.data.new(B, self.in_dim * self.r).zero_())

        # Init decoder states
        attention_rnn_hidden = Variable(
            encoder_outputs.data.new(B, 256).zero_())
        decoder_rnn_hiddens = [Variable(
            encoder_outputs.data.new(B, 256).zero_()) for _ in range(len(self.decoder_rnns))]
        prev_attention_output = Variable(
            encoder_outputs.data.new(B, 256).zero_())

        # Time first (T_decoder, B, in_dim)
        if inputs is not None:
            inputs = inputs.transpose(0, 1)

        outputs = []
        alignments = []

        t = 0
        current_input = initial_input
        while True:
            if t > 0:
                current_input = outputs[-1] if greedy else inputs[t - 1]
            # Prenet
            current_input = self.prenet(current_input)

            # Input feed
            attention_rnn_input = torch.cat(
                (current_input, prev_attention_output), -1)

            # Attention RNN
            attention_rnn_hidden = self.attention_rnn(
                attention_rnn_input, attention_rnn_hidden)

            # Attention mechanism
            attn_output, alignment = self.attn(
                attention_rnn_hidden, encoder_outputs)

            # Keep this for next state
            prev_attention_output = attn_output

            # This is handled in GlobalAttention, I believe
            # decoder_input = torch.cat((attn_output, attention_rnn_hidden), -1)
            decoder_input = attn_output

            # Pass through the decoder RNNs
            for i in range(len(self.decoder_rnns)):
                decoder_rnn_hiddens[i] = self.decoder_rnns[i](
                    decoder_input, decoder_rnn_hiddens[i])
                # Residual connectinon
                decoder_input = decoder_rnn_hiddens[i] + decoder_input

            # Last decoder hidden state is the output
            output = decoder_rnn_hiddens[-1]
            output = self.proj_to_mel(output)

            outputs += [output]
            alignments += [alignment]

            t += 1

            if greedy:
                if t > 1 and is_end_of_frames(output):
                    break
                elif t > self.max_decoder_steps:
                    print("Warning! doesn't seems to be converged")
                    break
            else:
                if t >= T_decoder:
                    break

        assert greedy or len(outputs) == T_decoder

        # Back to batch first
        alignments = torch.stack(alignments).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()

        return outputs, alignments


def is_end_of_frames(output, eps=1e-5):
    return (output.data <= eps).all()


def get_mask_from_lengths(inputs, input_lengths):
    mask = inputs.data.new(inputs.size(0), inputs.size(1)).byte().zero_()
    for idx, l in enumerate(input_lengths):
        mask[idx][:l] = 1
    mask = ~mask
    mask = mask.unsqueeze(0)
    return mask


class Tacotron(nn.Module):
    def __init__(self, n_vocab, in_dim=256, mel_dim=80, linear_dim=1025, r=5):
        super(Tacotron, self).__init__()
        self.mel_dim = mel_dim
        self.embedding = nn.Embedding(n_vocab, in_dim, padding_idx=None)
        # Trying smaller std
        self.embedding.weight.data.normal_(0, 0.5)
        self.encoder = Encoder(in_dim)
        self.decoder = Decoder(mel_dim, r)

        self.postnet = CBHG(mel_dim, K=8, projections=[256, mel_dim])
        self.last_linear = nn.Linear(mel_dim * 2, linear_dim)

    def forward(self, inputs, targets=None, input_lengths=None):
        B = inputs.size(0)

        # Used for attention mask
        if input_lengths is None:
            mask = None
        else:
            mask = get_mask_from_lengths(inputs, input_lengths)

        inputs = self.embedding(inputs)
        encoder_outputs = self.encoder(inputs, input_lengths)
        mel_outputs, alignments = self.decoder(encoder_outputs, targets, mask=mask)

        # Post net processing below

        # Reshape
        # (B, T, mel_dim)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments
