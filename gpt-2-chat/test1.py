from transformers.modeling_gpt2 import GPT2LMHeadModel
from transformers.tokenization_gpt2 import GPT2Tokenizer
import torch
import torch.nn.functional as F
from datasets_scripts import personachat

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True)

generated = tokenizer.encode('I love my dog')
context, past = torch.tensor([generated]), None

for _ in range(5):
    logits, past, attention = model(context, past=past)

    context = torch.multinomial(F.softmax(logits[:, -1]), 1)

    generated.append(context.item())

sequence = tokenizer.decode(generated)

print(sequence)