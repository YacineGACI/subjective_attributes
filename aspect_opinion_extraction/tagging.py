import torch
import torch.nn as nn
from transformers import BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = torch.load("aspect_opinion_extraction/as_op_tagging_bert_attn_epoch20000.pt")
logsoftmax = nn.LogSoftmax(dim=-1)

def tokenize_sentence(sentence):
    return tokenizer.encode(sentence)

def index2tag(index):
    possible_tags = ['O', 'B-AS', 'I-AS', 'B-OP', 'I-OP']
    return possible_tags[index]


def softmax2tag(softmax):
    output = [] 
    for batch in range(softmax.size(0)):
        this_sequence = []
        for token_index in range(1, softmax.size(1) - 1): # To remove the special tokens inserted by BERT
            log_probabilities = softmax[batch][token_index]
            top_n, top_i = log_probabilities.data.topk(1)
            category_i = top_i[0].item()
            this_sequence.append(index2tag(category_i))
        output.append(this_sequence)
    return output



def adjust_tags(sentence, tags):
    # This is to remove tags associated with truncated words (those that beging with ##)
    # They will have the same tag as the first chunk of the word
    # This method works on a list of tags (not a list of lists)
    input = tokenizer.tokenize(sentence)
    new_tags = []
    for i, token in enumerate(input):
        if token[:2] != "##":
            new_tags.append(tags[i])
    return new_tags


def adjust_sentence(tokenized_sentence):
    new_sentence = []
    for token in tokenized_sentence:
        if "##" not in token:
            new_sentence.append(token)
        else:
            new_sentence[-1] += token[2:]
    return new_sentence


def sample(sentence):
    input = torch.tensor(tokenize_sentence(sentence)).unsqueeze(0)
    output = model(input)[0]
    output = logsoftmax(output)
    tags = softmax2tag(output)
    return adjust_tags(sentence, tags[0])