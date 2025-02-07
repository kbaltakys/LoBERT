from datasets import Dataset
from transformers import TrainingArguments, Trainer
from theorder import (
    LoBertModel,
    LoBertConfig,
    LoBertForMaskedLM,
    LoBertTokenizer,
    DataCollatorForMessageModeling
    )


path_data_sample = "/workspaces/2025 LoBERT/data/LOBSTER_SampleFile_AAPL_2012-06-21_10/ArrowDataset"
ds = Dataset.load_from_disk(path_data_sample)
tokenized_dataset = ds.train_test_split(test_size=0.1)


config_lobert_tiny = {
    "hidden_size": 128,
    "intermediate_size": 512,
    "layer_norm_eps": 1e-12,
    "max_position_embeddings": 512,
    "model_type": "lobert",
    "num_attention_heads": 2,
    "num_hidden_layers": 2,
    "vocab_size": 3202,
}
config = LoBertConfig(**config_lobert_tiny)

lobert = LoBertForMaskedLM(config)
tokenizer = LoBertTokenizer()# .from_pretrained("/workspaces/2025 LoBERT/src/tokenizer/lobert")

data_collator = DataCollatorForMessageModeling(tokenizer=tokenizer, mlm=True)


args = TrainingArguments(
    output_dir='/workspaces/2025 LoBERT/models',
    num_train_epochs=10,
    include_num_input_tokens_seen=True,
    eval_strategy='steps',
    logging_steps=20,
    report_to=['tensorboard', 'wandb'],
    warmup_ratio=0.1,
    fp16=True,
    fp16_full_eval=True,
    )


trainer = Trainer(
    model=lobert,
    processing_class=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['test']
)


trainer.train()