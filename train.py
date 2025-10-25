import torch

f = open("input.txt", "r", encoding="utf-8")
text = f.read()


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


print(len(text))
