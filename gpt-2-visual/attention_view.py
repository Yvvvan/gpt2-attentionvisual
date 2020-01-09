"""Module to display the attention of each layer

"""

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math
import numpy as np

#color = ['r','orange','gold','yellowgreen','g','c','deepskyblue','b','darkviolet','violet','pink','slategray']
color_cycle = plt.rcParams['axes.prop_cycle']
color = color_cycle.by_key()['color']

def model_view(attention):
    """
    this function is used to draw the attentions from 12 Layers, 12 Heads,
    which means it contains 144 subplot.
    :param attention: tuple(attn_0, ... attn_11) , attn_n: n_sentence(1) * 12Layers * sentence_len *  sentence_len
                      the attention witch is returned from model(input) and with parameter: output_attentions=True
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
        for (head, b) in enumerate(a[0]): # 12 Heads  /*a[0] -> the first sentence, normally there is only one sentence
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
    plt.savefig('fig/modelview')
    plt.show()
    plt.close()

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
    plt.savefig('fig/headview_layer%d'%layer_num)
    plt.show()
    plt.close()

def neuron_view(qkv, attention, layer, head):
    """
    in this function, the detail of q,k,v will be shown
    :param qkv: [(q'0, k'0, v'0), ... , (q'11, k'11, v'11)]  (list of 12(Layers)tuple)
    :param attention:
    :param layer: which layer 0-11
    :param head: which head 0-11
    :return:
    """
    color_neuron = ['darkorange', 'dodgerblue']

    line_plot = 4 # draw 4 subplots in a line [q, k, q*k, qk]
    # q'/v': n_sentence(1) * 12Head * sentence_len * 64
    # k': n_sentence(1) * 12Head * 64 * sentence_len
    # to get a q/k/v from certain head in certain layer:  [layer][q_or_k_or_v][n_sentence(1)][head]
    q = qkv[layer][0][0][head]
    k = qkv[layer][1][0][head]
    v = qkv[layer][2][0][head]
    k_t = torch.transpose(k, 0, 1)
    attn = attention[layer][0][head]
    # now we get a q/k/v from certain head in certain layer,
    # q/v in form: sentence_len*64; k in form: 64*sentence_len
    # attn in form: sentence_len*sentence_len
    sentence_len = q.size()[0]

    for left in range(sentence_len):
        subplot_num = 1
        plt.figure(figsize=(50, 5))
        plt.suptitle('neuron view of the query and key value in Layer %d Head %d'%(layer,head), fontsize=20)
        q_norm = np.interp(q[left].detach().numpy(), (q[left].min(), q[left].max()), (-1, +1))
        scores = []
        for right in range(sentence_len):
            if right > left:
                break
            k_norm = np.interp(k_t[right].detach().numpy(), (k_t[right].min(), k_t[right].max()), (-1, +1))
            value = torch.mul(q[left],k_t[right])
            value_norm = np.interp(value.detach().numpy(), (value.min(), value.max()), (-1, +1))
            scores.append(sum(value.detach().numpy())/ math.sqrt(v.size(-1)))
            for line_subplot in range(line_plot - 1):
                current_plot = subplot_num + line_subplot
                ax = plt.subplot(sentence_len, line_plot, current_plot)
                ax.axis('off')
                if right==0:
                    if line_subplot == 0:
                        plt.title('Query q')
                    if line_subplot == 1:
                        plt.title('Key k')
                    if line_subplot == 2:
                        plt.title('q x k (elementwise)')
                    if line_subplot == 3:
                        plt.title('qk',x=0.05 )
                if left == right and line_subplot == 0: # draw q
                    for i in range(q.size()[1]): # 64
                        if q[left][i].item() >= 0:
                            c = color_neuron[0]  # +: orange
                        else: c = color_neuron[1]  # -: blue
                        ax.add_patch(
                            patches.Rectangle(
                                (i/2, 0),  # (x,y)
                                0.5,  # width
                                1,  # height
                                color=c,
                                alpha=abs(q_norm[i])
                            )
                        )
                    ax.plot([0], [0])
                elif line_subplot == 1: # draw k_t
                    for i in range(k_t.size()[1]): # 64
                        if k_t[right][i].item() >= 0:
                            c = color_neuron[0]  # +: orange
                        else: c = color_neuron[1]  # -: blue
                        ax.add_patch(
                            patches.Rectangle(
                                (i/2, 0),  # (x,y)
                                0.5,  # width
                                1,  # height
                                color=c,
                                alpha=abs(k_norm[i])
                            )
                        )
                    ax.plot([0], [0])
                elif line_subplot == 2: # draw q*k(elementwise)
                    for i in range(q.size()[1]): # 64
                        #value = q[left][i].item() * k_t[right][i].item()
                        if value[i] >= 0:
                            c = color_neuron[0]  # +: orange
                        else: c = color_neuron[1]  # -: blue
                        ax.add_patch(
                            patches.Rectangle(
                                (i/2, 0),  # (x,y)
                                0.5,  # width
                                1,  # height
                                color=c,
                                alpha=abs(value_norm[i])
                            )
                        )
                    ax.plot([0], [0])
            subplot_num += line_plot
        #elif line_subplot == 3: # draw q*k
        scores = np.array(scores)
        if scores.max() < 0:
            scores = np.interp(scores, (scores.min(), scores.max()), (scores.min(), abs(scores.max())))
        if scores.min() < -1 or scores.max() > 1:
            max = abs(scores.min())
            if max < scores.max():
                max = scores.max()
            for i in range(len(scores)):
                scores[i] /= max
        for right in range(left+1):
            score = scores[right]
            if score >= 0:
                c = color_neuron[0]  # +: orange
            else:
                c = color_neuron[1]  # -: blue
            ax = plt.subplot(sentence_len, line_plot, (right + 1) * 4)
            ax.axis('off')
            if right == 0:
                plt.title('qk', x=0.1)
            for i in range(8):  # 64/4
                if i == 0:
                    ax.add_patch(
                        patches.Rectangle(
                            (0, 0),  # (x,y)
                            4,  # width
                            1,  # height
                            color=c,
                            alpha=abs(score)
                        )
                    )
                else:
                    ax.add_patch(
                        patches.Rectangle(
                            (i*4, 0),  # (x,y)
                            4,  # width
                            1,  # height
                            color='r',
                            alpha=0
                        )
                    )
            ax.plot([0], [0])
        plt.savefig('fig/neuronview_layer%d_head%d_token%d' % (layer,head,left))
        plt.show()
        plt.close()
    # w = torch.matmul(q, k) / math.sqrt(v.size(-1))
    # for i in range(w.size()[0]):
    #     for j in range(w.size()[0]):
    #         if j>i:
    #             w[i][j]= -float('inf')
    # w = torch.nn.Softmax(dim=-1)(w)
    # print(w)
    # print(attn)
    return

def min_max_scale(t):
    min = torch.min(t)
    max = torch.max(t)
    scaled = []
    for i in range(t.size()[0]):
        scaled.append((t[i]-min)/(max-min))
    return torch.from_numpy(scaled)

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
    logits, past, attention, qkv = model(context, past=past)
    # model_view(attention)
    # head_view(attention) # ,layer =
    # head_view(attention, layer=0)  # ,layer =
    # head_view(attention, layer=5)
    # head_view(attention, layer=11)  # ,layer =
    neuron_view(qkv, attention, 0, 0)