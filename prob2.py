import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# DEVICE CONFIGURATION
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOAD DATASET
# We wrap each name with special tokens so the model knows where a name starts and ends
START_TOKEN = "<"
END_TOKEN = ">"

with open("TrainingNames.txt", "r", encoding="utf-8") as f:
    raw_lines = f.readlines()

names = []
idx_line = 0
while idx_line < len(raw_lines):
    names.append(raw_lines[idx_line].strip().lower())
    idx_line += 1

names = [START_TOKEN + nm + END_TOKEN for nm in names]

# Build a character-level vocabulary from all unique characters across all names
chars = sorted(list(set("".join(names))))

char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for ch, i in char_to_ix.items()}

vocab_size = len(chars)

# DATA PREPARATION
def name_to_tensor(name):
    # Convert each character in the name to its corresponding index
    return torch.tensor([char_to_ix[ch] for ch in name], dtype=torch.long)

encoded_names = []
k = 0
while k < len(names):
    encoded_names.append(name_to_tensor(names[k]))
    k += 1

# TASK 1
# MODEL 1: VANILLA RNN
class VanillaRNN(nn.Module):

    def __init__(self, vocab_size, hidden_dim):

        super().__init__()

        self.hidden_size = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.rnn = nn.RNN(
            hidden_dim,
            hidden_dim,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, vocab_size)


    def forward(self, x, hidden):

        x = self.embedding(x)
        output, hidden = self.rnn(x, hidden)
        # Project RNN outputs to vocab size so we get a score for each possible next character
        output = self.fc(output)
        return output, hidden

    def init_hidden(self):
        # Start with an all-zero hidden state at the beginning of each name
        return torch.zeros(1,1,self.hidden_size).to(device)

# MODEL 2: BLSTM
class BLSTMModel(nn.Module):

    def __init__(self, vocab_size, hidden_dim):

        super().__init__()

        self.hidden_size = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # bidirectional=True makes it BLSTM
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        # output becomes 2 * hidden_size because bidirectional
        self.fc = nn.Linear(
            hidden_dim * 2,
            vocab_size
        )

    def forward(self, x, hidden):

        x = self.embedding(x)

        output, hidden = self.lstm(x, hidden)

        forward_output = output[:, :, :self.hidden_size]

        if self.training:
            # During training, the full sequence is available so we can use both directions
            output = self.fc(output)

        else:
            # At generation time we only have past context, so we fake the backward direction
            # by mirroring the forward output — not ideal but keeps the fc layer happy
            combined = torch.cat(
                (forward_output, forward_output),
                dim=2
            )

            output = self.fc(combined)
        return output, hidden


    def init_hidden(self):
        # LSTM needs both a hidden state and a cell state, one pair per direction
        h0 = torch.zeros(2,1,self.hidden_size).to(device)
        c0 = torch.zeros(2,1,self.hidden_size).to(device)
        return (h0, c0)

# MODEL 3: RNN WITH ATTENTION
class AttentionRNN(nn.Module):

    def __init__(self, vocab_size, hidden_dim):

        super().__init__()

        self.hidden_size = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        self.rnn = nn.RNN(
            hidden_dim,
            hidden_dim,
            batch_first=True
        )

        self.attn = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        self.attn_combine = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        self.fc = nn.Linear(
            hidden_dim,
            vocab_size
        )


    def forward(self, x, hidden):

        x = self.embedding(x)

        output, hidden = self.rnn(x, hidden)
        # Compute self-attention: each position attends to every other position in the sequence
        attn_scores = torch.bmm(output,output.transpose(1, 2))
        attn_weights = torch.softmax(attn_scores,dim=2)
        # Weighted sum of all hidden states gives us a context vector for each position
        context = torch.bmm(attn_weights,output)
        # Merge the RNN output with the attention context before predicting the next character
        combined = torch.cat((output, context),dim=2)
        combined = self.attn_combine(combined)
        output = self.fc(combined)

        return output, hidden


    def init_hidden(self):

        return torch.zeros(
            1,
            1,
            self.hidden_size
        ).to(device)

# TRAIN FUNCTION

def train(model, epochs=10, lr=0.003):

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.CrossEntropyLoss()

    ep = 0
    while ep < epochs:

        total_loss = 0

        idx = 0
        while idx < len(encoded_names):

            name = encoded_names[idx].to(device)

            # Input is every character except the last; target is every character except the first
            # i.e. given "<ali", predict "ali>"
            input_seq = name[:-1].unsqueeze(0)
            target_seq = name[1:].unsqueeze(0)

            hidden = model.init_hidden()

            optimizer.zero_grad()

            output, hidden = model(
                input_seq,
                hidden
            )

            loss = criterion(
                output.squeeze(0),
                target_seq.squeeze(0)
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()
            idx += 1

        print("Epoch:", ep + 1,
              "Loss:", total_loss)
        ep += 1

# NAME GENERATION FUNCTION
def generate_name(model, max_length=20):

    model.eval()

    with torch.no_grad():

        hidden = model.init_hidden()

        # Kick off generation by feeding the start token
        input_char = torch.tensor(
            [[char_to_ix[START_TOKEN]]]
        ).to(device)

        generated = ""

        step = 0
        while step < max_length:

            output, hidden = model(input_char,hidden)
            logits = output[:, -1, :]

            # Lower temperature = more conservative/predictable names; higher = more creative/chaotic
            temperature = 0.6

            probs = torch.softmax(
                logits.squeeze() / temperature,
                dim=0
            )
            idx = torch.multinomial(probs,1).item()
            char = ix_to_char[idx]
            if char == END_TOKEN:
                break
            generated += char
            input_char = torch.tensor([[idx]]).to(device)
            step += 1

        return generated

# EVALUATION METRICS(TASK 2)
def evaluate_model(model, num_samples=200):

    generated = []

    i = 0
    while i < num_samples:
        generated.append(generate_name(model))
        i += 1

    generated_set = set(generated)

    training_set = set(
        name.replace(START_TOKEN, "")
        .replace(END_TOKEN, "")
        for name in names
    )

    # Novelty: fraction of generated names that don't exist in the training data
    novelty = sum(
        name not in training_set
        for name in generated
    ) / num_samples

    # Diversity: fraction of unique names among all generated names
    diversity = len(generated_set) / num_samples

    return novelty, diversity, generated[:10]

# PARAMETER COUNT FUNCTION
def count_parameters(model):

    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

# MAIN EXECUTION
hidden_dim = 128
learning_rate = 0.003
num_epochs = 15

print("\nTraining Vanilla RNN\n")

rnn_model = VanillaRNN(
    vocab_size,
    hidden_dim
)

train(
    rnn_model,
    num_epochs,
    learning_rate
)

print(
    "Trainable parameters:",
    count_parameters(rnn_model)
)

novelty, diversity, samples = evaluate_model(rnn_model)

print("Novelty:", novelty)
print("Diversity:", diversity)
print("Samples:", samples)

print("\nTraining BLSTM Model\n")

blstm_model = BLSTMModel(
    vocab_size,
    hidden_dim
)

train(
    blstm_model,
    num_epochs,
    learning_rate
)

print(
    "Trainable parameters:",
    count_parameters(blstm_model)
)

novelty, diversity, samples = evaluate_model(blstm_model)

print("Novelty:", novelty)
print("Diversity:", diversity)
print("Samples:", samples)

print("\nTraining Attention RNN Model\n")

attention_model = AttentionRNN(
    vocab_size,
    hidden_dim
)

train(
    attention_model,
    num_epochs,
    learning_rate
)

print(
    "Trainable parameters:",
    count_parameters(attention_model)
)

novelty, diversity, samples = evaluate_model(attention_model)

print("Novelty:", novelty)
print("Diversity:", diversity)
print("Samples:", samples)