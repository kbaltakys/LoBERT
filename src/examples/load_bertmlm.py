from transformers import (
    BertForMaskedLM,
    BertConfig,
    DataCollatorForLanguageModeling,
    BertTokenizer,
    )

config = BertConfig()
model = BertForMaskedLM(config)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

tokenizer.pad()