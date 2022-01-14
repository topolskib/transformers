import pandas as pd

from transformers import BartForConditionalGeneration, TapexTokenizer


tokenizer = TapexTokenizer.from_pretrained("facebook/bart-large", add_prefix_space=True)
model = BartForConditionalGeneration.from_pretrained("nielsr/tapex-large-finetuned-wtq")

data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
table = pd.DataFrame.from_dict(data)
query = "how many movies does Brad Pitt have?"

encoding = tokenizer(table=table, queries=query, return_tensors="pt")
del encoding["token_type_ids"]
print(tokenizer.decode(encoding.input_ids.squeeze()))

# forward pass
outputs = model.generate(**encoding)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
