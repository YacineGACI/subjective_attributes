import torch
from transformers import BertTokenizer
import numpy as np

from aspect_opinion_extraction.tagging import *


def update_attentions(attentions, index):
    "Usable in presence of truncated words: Sums comluns and averages rows"
    x_sum = np.sum(attentions[:, index - 1: index + 1], axis=1)
    attentions[:, index - 1] = x_sum
    attentions = np.delete(attentions, index, 1)
    x_mean = np.mean(attentions[index - 1: index + 1], axis=0)
    attentions[index - 1] = x_mean
    attentions = np.delete(attentions, index, 0)
    return attentions



def process_tagging_output(tokens, tags, attention_scores):
    "Concatenates the tags and attention scores of truncated words"
    attention_scores = attention_scores.detach().numpy()
    attention_index = 0
    new_sentence= []
    new_tags = []
    for i, token in enumerate(tokens):
        if token[:2] != "##":
            if tags[i][0] == "I":
                # If the tag is I-AS or I-OP, don't add this tag to the new list of tags but append its token to the previous token
                new_sentence[-1] += " " + tokens[i]
                attention_scores = update_attentions(attention_scores, attention_index)
                attention_index -= 1
            else:
                new_sentence.append(token)
                new_tags.append(tags[i] if tags[i] == "O" else tags[i][2:]) # Remove the B-, only keep AS or OP
        else:
            new_sentence[-1] += token[2:]
            attention_scores = update_attentions(attention_scores, attention_index)
            attention_index -= 1
        attention_index += 1

    return new_sentence, new_tags, attention_scores



def choose_head(tensor, layer, head):
    "Tensor is of shape (layers, 1, 12, seq_length, seq_length)"
    return tensor[layer][0, head, :, :]



def split_into_aspects_opinions(tags):
    "Output the set of aspects and the set of opinions"
    aspects = []
    opinions = []
    for i, tag in enumerate(tags):
        if tag == 'AS':
            aspects.append(i)
        if tag == 'OP':
            opinions.append(i)
    return aspects, opinions


def select_matching_aspect_from_opinion(aspect_set, opinion_attention_scores):
    "Select the most likely aspect"
    return np.argmax(opinion_attention_scores[aspect_set]) if len(aspect_set) != 0  else None


def pair_aspects_opinions(aspects, opinions, attentions):
    "Output the (AS, OP) index pairs"
    pairs = []
    for o in opinions:
        best_as = select_matching_aspect_from_opinion(aspects, attentions[o])
        if best_as is not None:
            pairs.append((aspects[best_as], o))
    return pairs



def pairing(sentence, layer, head):
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    output, attentions = model(input_ids)
    output = logsoftmax(output)
    tags = ['O'] + softmax2tag(output)[0] + ['O']
    truncated_tokens = ['CLS'] + tokenizer.tokenize(sentence) + ['SEP'] # To keep track of which tags and attentions belong to the same word
    attention_scores = choose_head(attentions, layer, head)
    tokens, tags, attention_scores = process_tagging_output(truncated_tokens, tags, attention_scores)
    aspects, opinions = split_into_aspects_opinions(tags)
    pairs = pair_aspects_opinions(aspects, opinions, attention_scores)
    human_readable_pairs = []
    for pair in pairs:
        human_readable_pairs.append((tokens[pair[0]], tokens[pair[1]]))
    return human_readable_pairs




model.eval() # To disbale the dropout units