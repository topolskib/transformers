import pandas as pd

from transformers import BartForConditionalGeneration, TapexTokenizer


tokenizer = TapexTokenizer.from_pretrained("facebook/bart-large", add_prefix_space=True)
model = BartForConditionalGeneration.from_pretrained("nielsr/tapex-large-finetuned-wtq")

data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
table = pd.DataFrame.from_dict(data)
query = "how many movies does Brad Pitt have?"

# 1-1
encoding = tokenizer(table, queries=query, return_tensors="pt")
print(tokenizer.decode(encoding.input_ids.squeeze()))

# forward pass
outputs = model.generate(**encoding)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# 1-many
queries = ["how many movies does Brad Pitt have?", "how many movies does George Clooney have?"]
encoding = tokenizer(table, queries=queries, padding=True, return_tensors="pt")

for k, v in encoding.items():
    print(k, v.shape)

# many-many
data = {"Actors": ["Niels Rogge", "Leonardo Di Caprio", "Veerle Declercq"], "Number of movies": ["100", "10", "69"]}
table_2 = pd.DataFrame.from_dict(data)
tables = [table, table_2]
queries = ["how many movies does Brad Pitt have?", "how many movies does Niels Rogge have?"]
encoding = tokenizer(tables, queries=queries, padding=True, return_tensors="pt")

for k, v in encoding.items():
    print(k, v.shape)

# batched inference
outputs = model.generate(**encoding)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# many-1
encoding = tokenizer(tables, queries=query, padding=True, return_tensors="pt")

for k, v in encoding.items():
    print(k, v.shape)
