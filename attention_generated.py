"""this model will pay more attention on the generated word

"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sources.tokenization_gpt2 import GPT2Tokenizer
# import math
# import numpy as np


class AttentionAnalyser():
    def __init__(self):
        # parameters
        self.decode = False
        self.avg = True # the subplot in the right
        self.show_layer = 11 # should be 0-11; but if not, only layer_avg_view will be shown
        self.layer_avg = False # new pic to show the layers situation
        self.show_generation = False
        # save_data
        self.generated_dict = {}

    @ staticmethod
    def _setup_subplot(subplot):
        """
        this method is used to setup the plot
        :param subplot: Boolean. it will decide whether to add a subplot that shows the avg of 12 heads/layers.
        :return: List. [p1, p2] pr [p1]. the 12 heads or layers are located in p1. p2 is for the avg.
        """
        if subplot:
            # 2 sub-plot: 1.big show 12 heads, 2.small show the avg of 12 heads
            rect1 = [0.11, 0.05, 228/325, 0.95]  # [left, bottom, wide, high]
            rect2 = [0.09+228/325, 0.05, 19/325, 0.95]
            main_ax = plt.axes(rect1)
            sub_ax = plt.axes(rect2,sharey=main_ax)
            plts = [main_ax, sub_ax]
        else:
            # 1 plot: 12 heads
            plts = [plt.axes()]
        return plts

    def _draw(self, plts, a, t):
        """
        this function is used to draw the attention (whatever attn from gpt-2 or own-built attention),
        which is in form  12_heads(or layers) * b * n_tocken.
        if the input attention is directly from gpt-2: b = n_tocken,
        if the input attention is own-built: b = 1
        we only consider the attention of generated word, which means the attention of last word: b -> -1
        :param plts: list. [p1,p2] with avg of 12 heads/layers, or [p1] without avg
        :param a: ndarray / list / tensor. in size 12_heads(or layers) * b * n_tocken
        :param t: String. text to show. eg."token_0 token_1 token_2 token_3 token_4 generated_word"
        :return:
        """
        # used to calculate the avg, will be drawn in the plts[1]
        avg_sum = [0] * (len(t) - 1)
        # Draw each head/layer as a column
        for (loop, b) in enumerate(a):
            # Show the head/layer number on x-axis
            # plts[0].text(loop + 0.5, -0.7, '%d'%loop, ha='center', va='center', rotation=45)
            plts[0].text(loop + 0.5, -0.5, '%d' % loop, ha='center', va='center')
            # Draw each token as a square
            for (token_num, c) in enumerate(b[-1]):
                # Record each token to calculate the avg
                avg_sum[token_num] += c
                # Draw the square
                y_pos = len(t) - 2 - token_num
                plts[0].add_patch(
                    patches.Rectangle(
                        (loop, y_pos),  # (x,y)
                        1,  # width
                        1,  # height
                        color='darkorange',
                        alpha=abs(c)
                    ))
                # Show the token-word in y-ax
                if loop == 0:
                    plts[0].text(-0.5, y_pos + 0.5, t[token_num], ha='right', va='center')
        # Adjust the size of the pic
        plts[0].plot([12], [len(t) - 1])
        # Adjust the square
        plts[0].set_aspect(1)
        # Dont show the original axis
        plts[0].axis('off')

        # If to show the avg
        if self.avg:
            # the similar code as above
            for token_num in range(len(t) - 1):
                y_pos = len(t) - 2 - token_num
                plts[1].add_patch(
                    patches.Rectangle(
                        (0, y_pos),  # (x,y)
                        1,  # width
                        1,  # height
                        color='darkorange',
                        alpha=abs(avg_sum[token_num] / 12)
                    ))
            plts[1].plot([1], [len(t) - 1])
            plts[1].set_aspect(1)
            plts[1].axis('off')

            # Show 'avg' on the x-axis
            # plts[1].text(0.5, -0.7, 'avg', ha='center', va='center', rotation=45)
            plts[1].text(0.5, -0.5, 'avg', ha='center', va='center')

            # Print the generated word at the right side of the Subplot
            plts[1].text(1.5, (len(t) - 1) / 2, t[-1], ha='left', va='center')
        else:
            plts[0].text(12.5, (len(t) - 1) / 2, t[-1], ha='left', va='center')
        return

    def update(self, **kwargs):
        """
        to update the parameters
        :param kwargs: dictionary. parameters to update
        :return:
        """
        if 'decode' in kwargs:
            self.decode = kwargs['decode']
        if 'show_avgplot' in kwargs:
            self.avg = kwargs['show_avgplot']
        if 'layer' in kwargs:
            self.show_layer = kwargs['layer']
        if 'show_layeravg' in kwargs:
            self.layer_avg = kwargs['show_layeravg']

    def show_attention(self, attention, generated, **kwargs):
        """

        :param attention:
        :param generated:
        :param kwargs:
        :return:
        """
        # update
        self.update(**kwargs)

        # if decode, show the words; otherwise show the number of the words
        # why: because in some situations, a word will be decoded in more than one part. And we dont know how.
        # eg. understood <--> under stood
        if self.decode:
            # from transformers import GPT2Tokenizer
            decoder = GPT2Tokenizer.from_pretrained('gpt2')
            text = []
            for token in generated:
                text.append(decoder.decode(token))
            generated = text

        # pic part
        # set subplot
        plts = self._setup_subplot(self.avg)

        # used to calculate the layer_avg
        layer_sum = []

        for (layer, a) in enumerate(attention):  # 12 Layers
            # used to calculate the layer_avg
            if self.layer_avg:
                layer_sum.append([[0] * (len(generated) - 1)])
                for (head, b) in enumerate(a[0]):
                    for (token_num, c) in enumerate(b[-1]):
                        layer_sum[layer][0][token_num] += c / 12
            #  Draw only the certain layer
            if layer == self.show_layer:
                # Title
                plts[0].text(6.5, -1.2, 'Heads in Layer %d' % layer, ha='center', va='center')
                # Draw
                self._draw(plts, a[0], generated)
                plt.show()
                plt.close()

        # if draw all layer_avg
        if self.layer_avg:
            plts2 = self._setup_subplot(self.avg)
            self._draw(plts2, layer_sum, generated)
            plts2[0].text(6.5, -1.2, 'Layer Avg', ha='center', va='center')
            plt.show()
            plt.close()


    def save_attention(self, attention, generated, **kwargs):
        self.update(**kwargs)
        if self.decode:
            # from transformers import GPT2Tokenizer
            decoder = GPT2Tokenizer.from_pretrained('gpt2')
            text = []
            for token in generated:
                text.append(decoder.decode(token))
            generated = text
        layer_sum = []
        for (layer, a) in enumerate(attention):  # 12 Layers
            layer_sum.append([[0] * (len(generated) - 1)])
            for (head, b) in enumerate(a[0]):
                for (token_num, c) in enumerate(b[-1]):
                    layer_sum[layer][0][token_num] += c / 12
        if generated[-1] not in self.generated_dict:
            self.generated_dict[generated[-1]] = {'attn': [], 'layer_avg': [], 'input': []}
        self.generated_dict[generated[-1]]['attn'].append(attention)
        self.generated_dict[generated[-1]]['layer_avg'].append(layer_sum)
        self.generated_dict[generated[-1]]['input'].append(generated)

    def show_sorted(self, **kwargs):
        if 'search' in kwargs:
            word = kwargs['search']
            if word not in self.generated_dict:
                print("generated word not found.")
                if ' '+word in self.generated_dict:
                    print("try to add a space(' ') in the front of the word")
                return
        else:
            for word in self.generated_dict:
                break

        if 'show_generation' in kwargs:
            self.show_generation = kwargs['show_generation']

        min_len = len(self.generated_dict[word]['input'][0])  # include generated word
        for (loop, layer_avg) in enumerate(self.generated_dict[word]['layer_avg']):
            text_len = len(self.generated_dict[word]['input'][loop])
            if text_len < min_len:
                min_len = text_len
            # to draw all generated-situation of this word
            if self.show_generation:
                plts = self._setup_subplot(True)
                self._draw(plts, layer_avg, self.generated_dict[word]['input'][loop])  # include generated word
                plts[0].text(6.5, -1.2, 'Layer Avg', ha='center', va='center')
                plt.show()
                plt.close()
        pos = list(range(- 1 * (min_len - 1), 0))
        avg_attn = []
        for (loop, layer_avg) in enumerate(self.generated_dict[word]['layer_avg']):
            for (layer, layer_attn) in enumerate(layer_avg):
                if loop == 0: avg_attn.append([[0] * (min_len - 1)])
                #print(layer_attn[-1][-1 * min_len:])
                norm = self._norm(layer_attn[-1][(- 1 * (min_len - 1)):])
                #print(loop, layer, norm, sum(norm))
                for (token, token_attn) in enumerate(norm):
                    avg_attn[layer][0][token] += norm[token] / len(self.generated_dict[word]['layer_avg'])
        plts = self._setup_subplot(True)
        self._draw(plts, avg_attn, pos+[word])
        plts[0].text(6.5, -1.2, "Avg Attention of '%s' in each Layer to different Position" % word, ha='center', va='center')
        plt.show()
        plt.close()

    @staticmethod
    def _norm(list):
        return [float(i)/sum(list) for i in list]


# -- debug ---------------
if __name__ == "__main__":
    from sources.modeling_gpt2 import GPT2Model, GPT2LMHeadModel
    from sources.tokenization_gpt2 import GPT2Tokenizer
    import torch

    input_text = "I don't like the movie."
    input_texts = [
                   # "I don't like the movie.",
                   # "I like the movie.",
                   "The movie is terrible.",
                   "The movie is wonderful.",
                   "I love my dog.",
                   "I love my cat. It is so lovely and beautiful. "
                   "I'd love to play with it everyday."
                ]
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True)

    analyzer = AttentionAnalyser()

    for input_text in input_texts:
        generated = tokenizer.encode(input_text)
        context, past = torch.tensor([generated]), None
        for _ in range(5):
            logits, past, attention = model(context, past=past)

            # the last dimension(token) of logits is the possibility of each word in vocab
            # multinomial-function chooses the highest possible word
            context = torch.multinomial(torch.nn.functional.softmax(logits[:, -1]), 1)

            generated.append(context.item())

            #analyzer.show_attention(attention, generated, decode=True, show_avgplot=True, layer=12, show_layeravg=True)
            analyzer.save_attention(attention, generated, decode = True)

        sequence = tokenizer.decode(generated)

    analyzer.show_sorted( show_generation = True)
    stop = ''

