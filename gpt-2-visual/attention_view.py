"""Module to display the attention of each layer

"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#color = ['r','orange','gold','yellowgreen','g','c','deepskyblue','b','darkviolet','violet','pink','slategray']
color_cycle = plt.rcParams['axes.prop_cycle']
color = color_cycle.by_key()['color']

def model_view(attention):
    """
    this function is used to draw the attentions from 12 Layers, 12 Heads,
    which means it contains 144 subplot.
    :param attention: the attention witch is returned from model(input) and with parameter: output_attentions=True
    :return:
    line: Layer
    row: Head
    in each subplot (for example: the first one means Layer 0, Head 0):
    ----------------------------------------------------
    | Left Axis        --attention-->       Right Axis |
    |--------------------------------------------------|
    | tocken[0] I                          I tocken[0] |
    |                                                  |
    | tocken[1] love                    love tocken[1] |
    |                                                  |
    | tocken[2] my                        my tocken[2] |
    |                                                  |
    |    ...                                  ...      |
    ----------------------------------------------------
    """
    subplot_num = 1
    #background_color:black, pixel: (pic_size*100) x (pic_size*100) (-> pic_size: 5000*5000 = about 1 MB)
    pic_size = 50
    plt.figure(facecolor='black',figsize=(pic_size, pic_size))
    for (layer, a) in enumerate(attention): # 12 Layers
        for (head, b) in enumerate(a[0]): # 12 Heads
            plt.subplot(12, 12, subplot_num)
            token_len = b.size()[0] # sentence length
            for left in range(token_len):
                for right in range(token_len):
                    if right > left: # attention: the current word considers only the previous words
                        continue
                    else:
                        y0 = token_len - 1 - left
                        y1 = token_len - 1 - right
                        plt.axis('off')
                        plt.plot([0, 1], [y0, y1], color = color[layer-len(color)], linewidth = pic_size/5, alpha = b[left][right].item())
            subplot_num += 1
    plt.show()


def head_view(attention, **kwargs):
    """
    the function is to show the attention from each head of one certain layer or all layers
    :param attention: attetntion
    :param kwargs: layer = which layer to show
    :return:
    line: Layer
    row: token
    in each subplot (for example: the first one means Layer 0, and the Model has read the first token):
    ------------------------------------------------------
    | token          |  heads *12                        |
    |----------------------------------------------------|
    | tocken[0] I    |  |  |  |  |  |  |  |  |  |  |  |  |
    |                 -----------------------------------|
    | tocken[1] love |  |  |  |  |  |  |  |  |  |  |  |  |
    |                 -----------------------------------|
    | tocken[2] my   |  |  |  |  |  |  |  |  |  |  |  |  |
    |                 -----------------------------------|
    |    ...                                  ...        |
    ------------------------------------------------------

    """

    if 'layer' in kwargs:
        layer_num = kwargs['layer']
        if layer_num > 12:
            print('layer must from 0 to 11')
            return
    else:
        layer_num = 12

    subplot_num = 1
    token_len = attention[0][0].size()[1] # sentence_length

    pic_size = 50
    # change the pic size according to how many layers will be shown
    if layer_num != 12:
        plt.figure(figsize=(pic_size, pic_size/6))  # facecolor='black',
    else:
        plt.figure(figsize=(pic_size, pic_size))

    for (layer, a) in enumerate(attention):  # 12 Layers
        if layer_num != 12 and layer != layer_num:
            continue
        for left in range(token_len):

            #set the subplot
            if layer_num != 12:
                ax = plt.subplot(1, token_len, subplot_num)
            elif layer_num == 12:
                ax = plt.subplot(12, token_len, subplot_num)
            ax.axis('off')

            for right in range(token_len):
                if right > left:  # attention: the current word considers only the previous words
                    continue
                else:
                    x = (token_len - 1 - right) * 2
                    for i in range(12):
                        ax.add_patch(
                            patches.Rectangle(
                                (i, x),  # (x,y)
                                1,  # width
                                2,  # height
                                color=color[i-len(color)],
                                alpha=a[0][i][left][right].item()
                            )
                        )
            ax.plot([0], [0])
            subplot_num += 1
    plt.show()


## in this function, the detail of q,k,v will be shown
def neuron_view():
    return

# --- debug ---------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    from sources.modeling_gpt2 import GPT2Model, GPT2LMHeadModel
    from sources.tokenization_gpt2 import GPT2Tokenizer

    text = 'I hate the movie last night. It is really bad.'
    text = 'I hate the movie'

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', output_attentions=True)
    generated = tokenizer.encode(text)
    context, past = torch.tensor([generated]), None
    logits, past, attention = model(context, past=past)
    model_view(attention)
    head_view(attention) # ,layer =
    head_view(attention, layer=0)  # ,layer =
    head_view(attention, layer=5)
    head_view(attention, layer=11)  # ,layer =