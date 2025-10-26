import torch

f = open("input.txt", "r", encoding="utf-8")
text = f.read()
text_length = len(text)

# 텍스트에 있는 모든 고유 문자를 찾아 정렬합니다.
chars = sorted(list(set(text)))
vocab_size = len(chars)

# 문자 <-> 정수 매핑을 만듭니다.
stoi = {ch: i for i, ch in enumerate(chars)}  # string to integer
itos = {i: ch for i, ch in enumerate(chars)}  # integer to string


# Encoder: 문자를 정수로 변환 (stoi 사용)
def encode(s):
    output = []
    for character in s:
        output.append(stoi[character])
    return output

# Decoder: 정수를 문자로 변환 (itos 사용)
def decode(l):
    output = []
    for code in l:
        output.append(itos[code])
    return "".join(output)


data = torch.tensor(encode(text), dtype=torch.long)

training_data = data[:text_length]
validation_data = data[text_length:]

sequence_length = 8
x = training_data[:sequence_length]
y = training_data[1: sequence_length + 1]

# batch
batch_size = 4
sequence_length = 8


# Autoregressive model
def get_batch(is_training):
    data = training_data if is_training else validation_data
    start_positions = torch.randint(len(data) - sequence_length, (batch_size,))
    # data point t-1
    context_list = torch.stack([data[i: i + sequence_length] for i in start_positions])
    # data point at t
    target_list = torch.stack(
        [data[i + 1: i + sequence_length + 1] for i in start_positions]
    )
    return context_list, target_list


# Example
contexts, targets = get_batch(True)

for start_position in range(batch_size):
    for current_end in range(sequence_length):
        context = contexts[start_position, : current_end + 1]
        target = targets[start_position, current_end]
        print(
            f"when current_end: {current_end} , start_position: {start_position} input is {context.tolist()} the target: {target}"
        )

# Bi-gram language model
import torch.nn as nn
from torch.nn import functional as F

vocab_size = len(set(text))


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding_table(idx)  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


m = BigramLanguageModel(vocab_size)
logits, loss = m.forward(contexts, targets)
print(logits)
# print(loss)
