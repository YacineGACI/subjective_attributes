import torch 
import torch.nn as nn
import random

from similarity.conceptual_sim.models import GPT2_2Layer

def read_training_data(filename):
    phrases = []
    labels = []
    with open(filename, 'r') as f:
        for l in f.readlines():
            s1, s2, sim = l.split(',')
            phrases.append((s1.strip(' '), s2.strip(' ')))
            labels.append(float(sim.strip('\n ')))
    return phrases, labels



def train(input_1, input_2, attention_mask_1, attention_mask_2, target):
    "input_x --> [batch_size, seq_length]  |  target --> [batch_size]"
    model.train()
    model.zero_grad()
    output = model(input_1, input_2, attention_mask_1, attention_mask_2)
    loss = criterion(output, target.unsqueeze(-1))

    loss.backward()
    optimizer.step()

    return loss.item()



def test(input_1, input_2, attention_mask_1, attention_mask_2, target):
    model.eval()
    output = model(input_1, input_2, attention_mask_1, attention_mask_2)
    loss = criterion(output, target.unsqueeze(-1))
    return loss.item()






if __name__ == "__main__":

    phrases, labels = read_training_data("similarity/conceptual_sim/data/sim_dataset_no_expansion_51_10000.csv")
    assert len(phrases) == len(labels)
    num_training_example = len(phrases)
    print('Training dataset read')

    learning_rate = 0.00005
    n_epochs = 800
    minibatch_size = 32
    weight_decay = 0
    dropout = 0.2
    train_test_split = 0.7
    print_every = 10

    model = GPT2_2Layer(dropout=dropout)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    print("GPT2 model loaded successfully")

    tokenizer = model.tokenizer

    x_train_1 = []
    x_train_2 = []
    attention_mask_train_1 = []
    attention_mask_train_2 = []

    for i in range(len(phrases)):
        sentence_1 = "[CLS] " + phrases[i][0] + " [SEP] " + phrases[i][1] + " [SEP]"
        tmp_1 = tokenizer.encode_plus(sentence_1, add_special_tokens=True, pad_to_max_length=True, max_length=20, return_token_type_ids=False)
        x_train_1.append(tmp_1['input_ids'])
        attention_mask_train_1.append(tmp_1['attention_mask'])

        sentence_2 = "[CLS] " + phrases[i][1] + " [SEP] " + phrases[i][0] + " [SEP]"
        tmp_2 = tokenizer.encode_plus(sentence_2, add_special_tokens=True, pad_to_max_length=True, max_length=20, return_token_type_ids=False)
        x_train_2.append(tmp_2['input_ids'])
        attention_mask_train_2.append(tmp_2['attention_mask'])


    x_train_1 = torch.tensor(x_train_1)
    x_train_2 = torch.tensor(x_train_2)
    attention_mask_train_1 = torch.tensor(attention_mask_train_1)
    attention_mask_train_2 = torch.tensor(attention_mask_train_2)
    y_train = torch.tensor(labels)

    x_test_1 = x_train_1[int(num_training_example * train_test_split):]
    x_test_2 = x_train_2[int(num_training_example * train_test_split):]
    attention_mask_test_1 = attention_mask_train_1[int(num_training_example * train_test_split):]
    attention_mask_test_2 = attention_mask_train_2[int(num_training_example * train_test_split):]
    y_test = y_train[int(num_training_example * train_test_split):]

    x_train_1 = x_train_1[:int(num_training_example * train_test_split)]
    x_train_2 = x_train_2[:int(num_training_example * train_test_split)]
    attention_mask_train_1= attention_mask_train_1[:int(num_training_example * train_test_split)]
    attention_mask_train_2 = attention_mask_train_2[:int(num_training_example * train_test_split)]
    y_train = y_train[:int(num_training_example * train_test_split)]
    print("Dataset processing done")

    train_loss = 0
    test_loss = 0

    model.train()

    for epoch in range(n_epochs):
        boundary = random.randint(0, int(num_training_example * train_test_split) - minibatch_size - 1)
        input_1 = x_train_1[boundary: boundary + minibatch_size]
        input_2 = x_train_2[boundary: boundary + minibatch_size]
        attention_mask_1 = attention_mask_train_1[boundary: boundary + minibatch_size]
        attention_mask_2 = attention_mask_train_2[boundary: boundary + minibatch_size]
        target = y_train[boundary: boundary + minibatch_size]

        train_loss += train(input_1, input_2, attention_mask_1, attention_mask_2, target)



        boundary = random.randint(0, int(num_training_example * (1 - train_test_split)) - minibatch_size - 1)
        input_1 = x_test_1[boundary: boundary + minibatch_size]
        input_2 = x_test_2[boundary: boundary + minibatch_size]
        attention_mask_1 = attention_mask_test_1[boundary: boundary + minibatch_size]
        attention_mask_2 = attention_mask_test_2[boundary: boundary + minibatch_size]
        target = y_test[boundary: boundary + minibatch_size]

        test_loss += test(input_1, input_2, attention_mask_1, attention_mask_2, target)

        if (epoch + 1) % print_every == 0:
            print("Training {}% --> Training Loss = {}".format(round(((epoch + 1) / n_epochs) * 100, 2), train_loss/print_every))
            print("Training {}% --> Evaluation Loss = {}".format(round(((epoch + 1) / n_epochs) * 100, 2), test_loss/print_every))
            print()
            train_loss = 0
            test_loss = 0

    print("Training Complete")
    torch.save(model.state_dict(), "similarity/conceptual_sim/models/gpt2-2layers-Adam_{}_{}_{}_{}_{}_{}.pt".format(n_epochs, minibatch_size, learning_rate, weight_decay, dropout, train_test_split))
    print("Model saved")
