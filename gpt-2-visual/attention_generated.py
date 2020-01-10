"""this model will pay more attention on the generated word

"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np

class attention_analyse():
    """

    """
    text = None
    attn = None

    def __init__(self, generated):
        self.text = generated

    def show(self, attention, generated, head_avg = True):
        # update
        self.text = generated
        self.attn = attention
        # pic
        if head_avg:
            # grid = plt.GridSpec(1,12, wspace=0.5, hspace=0.5)
            # main_ax = plt.subplot(grid[0,0:10])
            # sub_ax = plt.subplot(grid[0,11],sharey=main_ax)
            rect1 = [0.1, 0.05, 0.9/7*6, 0.95]  # [left, bottom, wide, high]
            rect2 = [0.88, 0.05, 0.9/14, 0.95]
            main_ax = plt.axes(rect1)
            sub_ax = plt.axes(rect2,sharey=main_ax)
            plts = [main_ax, sub_ax]
        else:
            plts = [plt.axes()]

        for (layer, a) in enumerate(self.attn):  # 12 Layers
            if layer != 11: #先做最后一个layer
                continue
            head_sum = [0]* (len(self.text)-1)

            for (head, b) in enumerate(a[0]):  # 12 Heads  /*a[0] -> the first sentence, normally there is only one sentence
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

            plts[0].plot([12], [len(self.text) - 1])
            plts[0].set_yticklabels(self.text[::-1])
            plts[0].set_aspect(1)
            plts[0].spines['top'].set_visible(False)
            plts[0].spines['right'].set_visible(False)
            plts[0].spines['bottom'].set_visible(False)
            plts[0].spines['left'].set_visible(False)
            # plts[0].xticks([range(12)])
            plts[0].set_xlabel('head')
            x_ticks=[]
            for i in range(12):
                x_ticks.append('head_%d'%i)
            my_x_ticks = np.arange(0, 12, 1)
            plt.xticks(my_x_ticks, axes=plts[0])
            # plts[0].set_xticklabels(x_ticks)

            if head_avg:
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
                plts[1].set_yticklabels(self.text[: :-1])
                plts[1].set_aspect(1)
                plts[1].axis('off')



            plt.show()
            break



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

for _ in range(3):
    logits, past, attention = model(context, past=past)

    # the last dimension(token) of logits is the possibility of each word in vocab
    # multinomial-function chooses the highest possible word
    context = torch.multinomial(torch.nn.functional.softmax(logits[:, -1]), 1)

    generated.append(context.item())
    generated_token = tokenizer.decode(context.item())

    analyzer.show(attention, generated)


    break


sequence = tokenizer.decode(generated)

stop=''