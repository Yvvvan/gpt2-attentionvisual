"""this model will pay more attention on the generated word

"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# import math
# import numpy as np

class attention_analyse():
    text = None
    attn = None
    decode = False
    avg = True
    show_layer = 11
    layer_avg = False

    def __init__(self, generated):
        self.text = generated

    def setup_subplot(self, subplot):
        if subplot:
            # 2 sub-plot: 1.big show 12 heads, 2.small show the avg of 12 heads
            rect1 = [0.1, 0.05, 228/325, 0.95]  # [left, bottom, wide, high]
            rect2 = [0.08+228/325, 0.05, 19/325, 0.95]
            main_ax = plt.axes(rect1)
            sub_ax = plt.axes(rect2,sharey=main_ax)
            plts = [main_ax, sub_ax]
        else:
            # 1 plot: 12 heads
            plts = [plt.axes()]
        return plts

    def draw(self,plts, a):
        # used to calculate the head_avg
        head_sum = [0] * (len(self.text) - 1)
        # Draw each head as a column
        for (head, b) in enumerate(a):
            # Show the head number on x-axis
            # plts[0].text(head + 0.5, -0.7, 'head_%d'%head, ha='center', va='center', rotation=45)
            plts[0].text(head + 0.5, -0.5, '%d' % head, ha='center', va='center')
            # Draw each token as a square
            for (token_num, c) in enumerate(b[-1]):
                # Record each token to calculate the head_avg
                head_sum[token_num] += c
                #layer_sum[head][token_num] += c
                # Draw the square
                y_pos = len(self.text) - 2 - token_num
                plts[0].add_patch(
                    patches.Rectangle(
                        (head, y_pos),  # (x,y)
                        1,  # width
                        1,  # height
                        color='darkorange',
                        alpha=abs(c)
                    ))
                # Show the token-word in y-ax
                if head == 0:
                    plts[0].text(-0.5, y_pos + 0.5, self.text[token_num], ha='right', va='center')
        # Adjust the size of the pic
        plts[0].plot([12], [len(self.text) - 1])
        # Adjust the square
        plts[0].set_aspect(1)
        # Dont show the original axis
        plts[0].axis('off')

        # If to show the head_avg
        if self.avg:
            # the similar code as above
            for token_num in range(len(self.text) - 1):
                y_pos = len(self.text) - 2 - token_num
                plts[1].add_patch(
                    patches.Rectangle(
                        (0, y_pos),  # (x,y)
                        1,  # width
                        1,  # height
                        color='darkorange',
                        alpha=abs(head_sum[token_num] / 12)
                    ))
            plts[1].plot([1], [len(self.text) - 1])
            plts[1].set_aspect(1)
            plts[1].axis('off')

            # Show 'avg' on the x-axis
            # plts[1].text(0.5, -0.7, 'head_avg', ha='center', va='center', rotation=45)
            plts[1].text(0.5, -0.5, 'avg', ha='center', va='center')

            # Print the generated word at the right side of the Subplot
            plts[1].text(1.5, (len(self.text) - 1) / 2, self.text[-1], ha='left', va='center')
        else:
            plts[0].text(12.5, (len(self.text) - 1) / 2, self.text[-1], ha='left', va='center')
        return

    def show_attention(self, attention, generated, **kwargs):
        # update
        self.text = generated
        self.attn = attention

        if 'decode' in kwargs:
            self.decode = kwargs['decode']
        if 'avg' in kwargs:
            self.avg = kwargs['avg']
        if 'layer' in kwargs:
            self.show_layer = kwargs['layer']
        if 'layer_avg' in kwargs:
            self.layer_avg = kwargs['layer_avg']

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
        # set subplot
        plts = self.setup_subplot(self.avg)

        # used to calculate the layer_avg
        layer_sum = []

        for (layer, a) in enumerate(self.attn):  # 12 Layers
            # used to calculate the layer_avg
            if self.layer_avg:
                layer_sum.append([[0] * (len(self.text) - 1)])
                for (head, b) in enumerate(a[0]):
                    for (token_num, c) in enumerate(b[-1]):
                        layer_sum[layer][0][token_num] += c/12
            #  Draw only the certain layer
            if layer == self.show_layer:
                # Title
                plts[0].text(6.5, -1.2, 'Heads in Layer %d'%layer, ha='center', va='center')
                # Draw
                self.draw(plts,a[0])
        plt.show()
        plt.close()

        #if draw all layer_avg
        if self.layer_avg:
            plts2 = self.setup_subplot(self.avg)
            self.draw(plts2, layer_sum)
            plts2[0].text(6.5, -1.2, 'Layer Avg' , ha='center', va='center')
            plt.show()
            plt.close()




# -- debug ---------------
if __name__ == "__main__":

    from sources.modeling_gpt2 import GPT2Model, GPT2LMHeadModel
    from sources.tokenization_gpt2 import GPT2Tokenizer
    import torch

    input_text = "I don't like the movie."

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

        analyzer.show_attention(attention, generated, decode=True, avg=True, layer=0, layer_avg=True)

    sequence = tokenizer.decode(generated)

    stop=''