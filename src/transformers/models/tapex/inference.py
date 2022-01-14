from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd

tokenizer = TapexTokenizer.from_pretrained("facebook/bart-large", add_prefix_space=True)
model = BartForConditionalGeneration.from_pretrained("nielsr/tapex-large-finetuned-wtq")

data = {'Actors': ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], 'Number of movies': ["87", "53", "69"]}
table = pd.DataFrame.from_dict(data)
query = "what's his name?"

encoding = tokenizer(table=table, queries=query, return_tensors="pt")

print(encoding.keys())

print(tokenizer.decode(encoding.input_ids.squeeze()))