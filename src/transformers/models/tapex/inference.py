import pandas as pd

from transformers import BartForConditionalGeneration, TapexTokenizer


tokenizer = TapexTokenizer.from_pretrained("facebook/bart-large")

question = "Greece held its last Summer Olympics in 2004"
table_dict = {
    "header": ["Year", "City", "Country", "Nations"],
    "rows": [
        [1896, "Athens", "Greece", 14],
        [1900, "Paris", "France", 24],
        [1904, "St. Louis", "USA", 12],
        [2004, "Athens", "Greece", 201],
        [2008, "Beijing", "China", 204],
        [2012, "London", "UK", 204],
    ],
}

table = pd.DataFrame.from_dict(table_dict["rows"])
table.columns = table_dict["header"]

encoding = tokenizer(table, question, return_tensors="pt")

print(encoding.input_ids)

print(tokenizer.decode(encoding.input_ids.squeeze()))

### integration tests

model = BartForConditionalGeneration.from_pretrained("nielsr/tapex-large-finetuned-wtq")

data = {"Actors": ["Brad Pitt", "Leonardo Di Caprio", "George Clooney"], "Number of movies": ["87", "53", "69"]}
table = pd.DataFrame.from_dict(data)
query = "how many movies does Brad Pitt have?"

# 1-1
encoding = tokenizer(table, query, return_tensors="pt")
print(tokenizer.decode(encoding.input_ids.squeeze()))

# forward pass
outputs = model.generate(**encoding)

print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# 1-many
queries = ["how many movies does Brad Pitt have?", "how many movies does George Clooney have?"]
encoding = tokenizer(table, queries, padding=True, return_tensors="pt")

for k, v in encoding.items():
    print(k, v.shape)

# many-many
data = {"Actors": ["Niels Rogge", "Leonardo Di Caprio", "Veerle Declercq"], "Number of movies": ["100", "10", "69"]}
table_2 = pd.DataFrame.from_dict(data)
tables = [table, table_2]
queries = ["how many movies does Brad Pitt have?", "how many movies does Niels Rogge have?"]
encoding = tokenizer(tables, queries, padding=True, return_tensors="pt")

for k, v in encoding.items():
    print(k, v.shape)

# batched inference
outputs = model.generate(**encoding)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))

# many-1
encoding = tokenizer(tables, query, padding=True, return_tensors="pt")

for k, v in encoding.items():
    print(k, v.shape)
