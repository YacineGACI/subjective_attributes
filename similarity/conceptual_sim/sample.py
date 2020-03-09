import torch 

from models import BERT_2Layer


def compute_conceptual_similarity(s1, s2):
    tmp = tokenizer.encode_plus(s1, s2, add_special_tokens=True, pad_to_max_length=True, max_length=20)
    input_ids = torch.tensor(tmp['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(tmp['attention_mask']).unsqueeze(0)
    seg = torch.tensor(tmp['token_type_ids']).unsqueeze(0)

    output = model(input_ids, attention_mask, seg)
    return output.item()
    


model = BERT_2Layer()
model.load_state_dict(torch.load("models/bert-2layers-expanded-Adam_800_32_5e-05_0.0004_0.2_0.7.pt"))
model.eval()

tokenizer = model.tokenizer


if __name__ == "__main__":
    s1 = "soup not yummy"
    s2 = "pasta really really bad"

    print(compute_conceptual_similarity(s1, s2))
