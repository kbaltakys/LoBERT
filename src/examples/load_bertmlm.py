from transformers import (
    BertForMaskedLM,
    BertConfig,
    DataCollatorForLanguageModeling,
    )

config = BertConfig()
model = BertForMaskedLM(config)
