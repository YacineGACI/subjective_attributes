import torch
import torch.nn as nn
import math

from models import BERT_2Layer, BERT_2Layer_Pooling


def read_data(filename):
    phrases = []
    labels = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            s1, s2, sim = l.split(',')
            phrases.append((s1.strip(' '), s2.strip(' ')))
            labels.append(float(sim.strip('\n ')))
    return phrases, labels




def test_pooling(input_1, input_2, attention_mask_1, attention_mask_2, target):
    model.eval()
    output = model(input_1, input_2, attention_mask_1, attention_mask_2)
    loss = criterion(output, target.unsqueeze(-1))
    return loss.item()


def test(input, attention_mask, seg, target):
    model.eval()
    output = model(input, attention_mask, seg)
    loss = criterion(output, target.unsqueeze(-1))
    return loss.item()



if __name__ == "__main__":
    phrases, labels = read_data("data/sim_dataset_51_30000.csv")
    num_test_data = len(phrases)
    print("Dataset read")

    model = BERT_2Layer()
    model.load_state_dict(torch.load("models/bert-2layers-Adam_800_32_5e-05_0.0004_0.2_0.7.pt"))
    model.eval()
    print("Model loaded")

    criterion = nn.BCELoss(reduction='sum')
    tokenizer = model.tokenizer

    x_test = []
    attention_mask_test = []
    seg_test = []

    for i in range(len(phrases)):
        tmp = tokenizer.encode_plus(phrases[i][0], phrases[i][1], add_special_tokens=True, pad_to_max_length=True, max_length=20)
        x_test.append(tmp['input_ids'])
        attention_mask_test.append(tmp['attention_mask'])
        seg_test.append(tmp['token_type_ids'])


    x_test = torch.tensor(x_test)
    attention_mask_test = torch.tensor(attention_mask_test)
    seg_test = torch.tensor(seg_test)
    y_test = torch.tensor(labels)
    print("Dataset processing done")

    minibatch_size = 100
    total_loss = 0

    for i in range(math.ceil(num_test_data / minibatch_size)):
        boundary = i * minibatch_size
        print("Evaluation at {} %".format(round(100 * boundary / num_test_data, 2)))
        loss = test(x_test[boundary: boundary + minibatch_size], attention_mask_test[boundary: boundary + minibatch_size], seg_test[boundary: boundary + minibatch_size], y_test[boundary: boundary + minibatch_size])
        total_loss += loss

    average_loss = total_loss / num_test_data
    print()
    print("Average Test Loss = ", average_loss)