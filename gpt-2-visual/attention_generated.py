"""this model will pay more attention on the generated word

"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np

class attention_analyse():
    text = None
    attn = None
    decode = False
    head_avg = True
    show_layer = 11

    def __init__(self, generated):
        self.text = generated

    def show(self, attention, generated, **kwargs):
        # update
        self.text = generated
        self.attn = attention

        if 'decode' in kwargs:
            self.decode = kwargs['decode']
        if 'head_avg' in kwargs:
            self.head_avg = kwargs['head_avg']
        if 'layer' in kwargs:
            self.show_layer = kwargs['layer']

        # if decode, show the words; otherwise show the number of the words
        # why: because in some situations, a word will be decoded in more than one part. And we dont know that.
        # eg. understood <--> under stood
        if self.decode:
            from sources.tokenization_gpt2 import GPT2Tokenizer
            # from transformers import GPT2Tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            text = []
            for token in self.text:
                text.append(tokenizer.decode(token))
            self.text = text

        # pic part
        # subplot
        if self.head_avg:
            # 2 sub-plot: 1.big show 12 heads, 2.small show the avg of 12 heads
            rect1 = [0.08, 0.05, 228/325, 0.95]  # [left, bottom, wide, high]
            rect2 = [0.08+228/325, 0.05, 19/325, 0.95]
            main_ax = plt.axes(rect1)
            sub_ax = plt.axes(rect2,sharey=main_ax)
            plts = [main_ax, sub_ax]
        else:
            # 1 plot: 12 heads
            plts = [plt.axes()]

        # Layers and Heads
        for (layer, a) in enumerate(self.attn):  # 12 Layers
            if layer != self.show_layer:
                continue
            # used to calculate the head_avg
            head_sum = [0]* (len(self.text)-1)
            # Head
            plts[0].text(6.5, -1.2, 'Heads in Layer %d'%layer, ha='center', va='center')
            for (head, b) in enumerate(a[0]):
                # show the heads in x-ax
                # plts[0].text(head + 0.5, -0.7, 'head_%d'%head, ha='center', va='center', rotation=45)
                plts[0].text(head + 0.5, -0.5, '%d' % head, ha='center', va='center')
                # Token
                for (token_num, c) in enumerate(b[-1]):
                    head_sum[token_num] += c
                    y_pos = len(self.text) - 2 - token_num
                    plts[0].add_patch(
                    patches.Rectangle(
                        (head, y_pos),  # (x,y)
                        1,  # width
                        1,  # height
                        color='darkorange',
                        alpha=abs(c)
                    ))
                    # show the tokens in y-ax
                    if head == 0:
                        plts[0].text(-0.5, y_pos +0.5, self.text[token_num], ha='right', va='center')

            plts[0].plot([12], [len(self.text) - 1])
            plts[0].set_aspect(1)
            # plts[0].set_yticklabels(self.text[::-1])
            # plts[0].spines['top'].set_visible(False)
            # plts[0].spines['right'].set_visible(False)
            # plts[0].spines['bottom'].set_visible(False)
            # plts[0].spines['left'].set_visible(False)
            plts[0].axis('off')

            if self.head_avg:
                for token_num in range(len(self.text)-1):
                    y_pos = len(self.text) - 2 - token_num
                    plts[1].add_patch(
                    patches.Rectangle(
                        (0, y_pos),  # (x,y)
                        1,  # width
                        1,  # height
                        color='darkorange',
                        alpha=abs(head_sum[token_num]/12)
                    ))
                plts[1].plot([1],[len(self.text)-1])
                plts[1].set_aspect(1)
                plts[1].axis('off')
                # plts[1].text(0.5, -0.7, 'head_avg', ha='center', va='center', rotation=45)
                plts[1].text(0.5, -0.5, 'avg', ha='center', va='center')
                # generated word
                plts[1].text(1.5, (len(self.text)-1)/2, self.text[-1], ha='left', va='center')
            else:
                plts[0].text(12.5, (len(self.text) - 1) / 2, self.text[-1], ha='left', va='center')

            plt.show()



# -- debug ---------------
if __name__ == "__main__":

    from sources.modeling_gpt2 import GPT2Model, GPT2LMHeadModel
    from sources.tokenization_gpt2 import GPT2Tokenizer
    import torch

    input_text = 'I hate the movie.'

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True)
    generated = tokenizer.encode(input_text)
    context, past = torch.tensor([generated]), None

    analyzer = attention_analyse(generated)

    for _ in range(1):
        logits, past, attention = model(context, past=past)

        # the last dimension(token) of logits is the possibility of each word in vocab
        # multinomial-function chooses the highest possible word
        context = torch.multinomial(torch.nn.functional.softmax(logits[:, -1]), 1)

        generated.append(context.item())
        generated_token = tokenizer.decode(context.item())

        analyzer.show(attention, generated, decode=True, head_avg=True)

    sequence = tokenizer.decode(generated)

    stop=''