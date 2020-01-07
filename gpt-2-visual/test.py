import torch
import torch.nn.functional as F
import numpy as np
from sources.modeling_gpt2 import GPT2Model,GPT2LMHeadModel
from sources.tokenization_gpt2 import GPT2Tokenizer
import warnings
warnings.filterwarnings("ignore")


tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True)

generated = tokenizer.encode('A unicorn is a fascinating')
context, past = torch.tensor([generated]), None

for _ in range(1):
    logits, past, attention = model(context, past=past)

    context = torch.multinomial(F.softmax(logits[:, -1]), 1)

    generated.append(context.item())

sequence = tokenizer.decode(generated)

print(sequence)