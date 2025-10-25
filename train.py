import torch

f = open("input.txt", "r", encoding="utf-8")
text = f.read()
text_length = len(text)


# Encoder
def encode(s):
    output = []
    for character in s:
        output.append(ord(character))
    return output


# Decoder
def decode(l):
    output = []
    for code in l:
        output.append(chr(code))
    return "".join(output)


data = torch.tensor(encode(text), dtype=torch.int32)

training_data = data[:text_length]
validation_data = data[text_length:]

context_window = 8
x = training_data[:context_window + 1]