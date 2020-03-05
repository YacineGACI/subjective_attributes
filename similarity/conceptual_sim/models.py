import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from transformers import GPT2Model, GPT2Tokenizer



class BERT_2Layer(nn.Module):
    def __init__(self, bert_hidden_size=768, classifier_hidden_size=512, dropout=0.1):
        super(BERT_2Layer, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.linear_1 = nn.Linear(bert_hidden_size, classifier_hidden_size)
        self.linear_2 = nn.Linear(classifier_hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)



    def forward(self, input, attn_mask=None, seg=None):
        last_hidden_state = self.model(input, attention_mask=attn_mask, token_type_ids=seg)[0] # [batch_size, seq_length, hidden_size]
        classification_vector = last_hidden_state[:, 0, :] # [batch_size, hidden_size]
        return self.sigmoid(self.linear_2(self.relu(self.dropout(self.linear_1(classification_vector)))))







class BERT_2Layer_Pooling(nn.Module):
    def __init__(self, bert_hidden_size=768, classifier_hidden_size=512, dropout=0.1):
        super(BERT_2Layer_Pooling, self).__init__()
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.linear_1 = nn.Linear(4 * bert_hidden_size, classifier_hidden_size)
        self.linear_2 = nn.Linear(classifier_hidden_size, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

    

    def forward(self, input_1, input_2, attn_mask_1=None, attn_mask_2=None):
        bert_output_1 = self.model(input_1, attention_mask=attn_mask_1)[0] # [batch_size, seq_length, hidden_dim]
        bert_output_2 = self.model(input_2, attention_mask=attn_mask_2)[0] # [batch_size, seq_length, hidden_dim]

        mean_pooled_vector_1 = bert_output_1.mean(1) # [batch_size, hidden_dim]
        mean_pooled_vector_2 = bert_output_2.mean(1) # [batch_size, hidden_dim]

        absolute_diff = torch.abs(mean_pooled_vector_1 - mean_pooled_vector_2)
        hadamard_product = mean_pooled_vector_1 * mean_pooled_vector_2
        classification_vector = torch.cat((mean_pooled_vector_1, mean_pooled_vector_2, absolute_diff, hadamard_product), dim=1)

        return self.sigmoid(self.linear_2(self.relu(self.dropout(self.linear_1(classification_vector)))))







class GPT2_2Layer(nn.Module):
    def __init__(self, gpt_final_size=768, hidden_size=256, dropout=0.1):
        super(GPT2_2Layer, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.model = GPT2Model.from_pretrained('gpt2')
        special_tokens = {'cls_token': '[CLS]', 'sep_token': '[SEP]', 'pad_token': '[PAD]'}
        num_added_toks = self.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.linear1 = nn.Linear(gpt_final_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)


    def forward(self, input1, input2, attention_mask_1=None, attention_mask_2=None):
        last_hidden_states_1 = self.model(input1, attention_mask=attention_mask_1)[0] # [batch_size, seq_length, hidden_size]
        last_hidden_states_2 = self.model(input2, attention_mask=attention_mask_2)[0] # [batch_size, seq_length, hidden_size]
        classification_vector_1 = last_hidden_states_1[:, 0, :] # Of shape [batch_size, hidden_size]
        classification_vector_2 = last_hidden_states_2[:, 0, :] # Of shape [batch_size, hidden_size]
        classification_vector = classification_vector_1.add(classification_vector_2)
        output = self.sigmoid(self.linear2(self.relu(self.dropout(self.linear1(classification_vector)))))
        return output

