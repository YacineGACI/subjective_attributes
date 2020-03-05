import torch 
import torch.nn as nn
import random

from models import BERT_2Layer

def read_training_data(filename):
    phrases = []
    labels = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            s1, s2, sim = l.split(',')
            phrases.append((s1.strip(' '), s2.strip(' ')))
            labels.append(float(sim.strip('\n ')))
    return phrases, labels



def train(input, attention_mask, seg, target):
    "input_x --> [batch_size, seq_length]  |  target --> [batch_size]"
    model.train()
    model.zero_grad()
    output = model(input, attention_mask, seg)
    loss = criterion(output, target.unsqueeze(-1))

    loss.backward()
    optimizer.step()

    return loss.item()



def test(input, attention_mask, seg, target):
    model.eval()
    output = model(input, attention_mask, seg)
    loss = criterion(output, target.unsqueeze(-1))
    return loss.item()




if __name__ == "__main__":

    phrases, labels = read_training_data("data/sim_dataset_no_expansion_51_10000.csv")
    assert len(phrases) == len(labels)
    num_training_example = len(phrases)
    print('Training dataset read')

    learning_rate = 0.00005
    n_epochs = 800
    minibatch_size = 32
    weight_decay = 0.0004
    dropout = 0.2
    train_test_split = 0.7
    print_every = 10

    model = BERT_2Layer(dropout=dropout)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print("BERT model loaded successfully")

    tokenizer = model.tokenizer

    x_train = []
    attention_mask_train = []
    seg_train = []


    for i in range(len(phrases)):
        tmp = tokenizer.encode_plus(phrases[i][0], phrases[i][1], add_special_tokens=True, pad_to_max_length=True, max_length=20)
        x_train.append(tmp['input_ids'])
        attention_mask_train.append(tmp['attention_mask'])
        seg_train.append(tmp['token_type_ids'])


    x_train = torch.tensor(x_train)
    attention_mask_train = torch.tensor(attention_mask_train)
    seg_train = torch.tensor(seg_train)
    y_train = torch.tensor(labels)

    x_test = x_train[int(num_training_example * train_test_split):]
    attention_mask_test = attention_mask_train[int(num_training_example * train_test_split):]
    seg_test = seg_train[int(num_training_example * train_test_split):]
    y_test = y_train[int(num_training_example * train_test_split):]

    x_train = x_train[:int(num_training_example * train_test_split)]
    attention_mask_train = attention_mask_train[:int(num_training_example * train_test_split)]
    seg_train = seg_train[:int(num_training_example * train_test_split)]
    y_train = y_train[:int(num_training_example * train_test_split)]
    print("Dataset processing done")



    train_loss = 0
    test_loss = 0

    model.train()

    for epoch in range(n_epochs):
        boundary = random.randint(0, int(num_training_example * train_test_split) - minibatch_size - 1)
        input = x_train[boundary: boundary + minibatch_size]
        attention_mask = attention_mask_train[boundary: boundary + minibatch_size]
        seg = seg_train[boundary: boundary + minibatch_size]
        target = y_train[boundary: boundary + minibatch_size]

        train_loss += train(input, attention_mask, seg, target)



        boundary = random.randint(0, int(num_training_example * (1 - train_test_split)) - minibatch_size - 1)
        input = x_test[boundary: boundary + minibatch_size]
        attention_mask = attention_mask_test[boundary: boundary + minibatch_size]
        seg = seg_test[boundary: boundary + minibatch_size]
        target = y_test[boundary: boundary + minibatch_size]

        test_loss += test(input, attention_mask, seg, target)

        if (epoch + 1) % print_every == 0:
            print("Training {}% --> Training Loss = {}".format(round(((epoch + 1) / n_epochs) * 100, 2), train_loss/print_every))
            print("Training {}% --> Evaluation Loss = {}".format(round(((epoch + 1) / n_epochs) * 100, 2), test_loss/print_every))
            print()
            train_loss = 0
            test_loss = 0

    print("Training Complete")
    torch.save(model.state_dict(), "models/bert-2layers-Adam_{}_{}_{}_{}_{}_{}.pt".format(n_epochs, minibatch_size, learning_rate, weight_decay, dropout, train_test_split))
    print("Model saved")