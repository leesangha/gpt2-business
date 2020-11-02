from transformers import AutoModelWithLMHead, AutoTokenizer, top_k_top_p_filtering
import torch

from torch.nn import functional as F
tokenizer = AutoTokenizer.from_pretrained("laxya007/gpt2_business")
model = AutoModelWithLMHead.from_pretrained("laxya007/gpt2_business", return_dict=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

sentence = 'I am'
input_ids = tokenizer.encode(sentence, return_tensors='pt')
input_ids = input_ids.to(device)

min_length = len(input_ids.tolist()[0])
length=int(30)
length += min_length

sample_outputs = model.generate(input_ids, pad_token_id=50256,
                                        do_sample=True,
                                        max_length=length,
                                        min_length=length,
                                        top_k=40,
                                        num_return_sequences=5)

result = dict()
for idx, sample_output in enumerate(sample_outputs):
    result[idx] = tokenizer.decode(sample_output.tolist()[min_length:], skip_special_tokens=True)
    print(sentence + ' ' + result[idx] + '--------' + str(idx) + '\n')