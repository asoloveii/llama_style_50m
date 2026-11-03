from datasets import load_from_disk 
from torch.utils.data import Dataset


def alpaca_collate_fn(batch, pad_token_id):
    input_ids = [torch.tensor(item["input_ids"], dtype=torch.long) for item in batch]
    labels = [torch.tensor(item["labels"], dtyle=torch.long) for item in batch]

    input_ids_padded = pad_sequence(inputs, batch_first=True, padding_value=PAD_TOKEN_ID)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

    return input_ids_padded, labels_padded


def alpaca_get_data(dataset_path, 
                    tokenizer,
                    context_size=1024,
                    test_size=0.5,
                    seed=42):
    ds = load_from_disk(dataset_path)

    split_ds = ds.train_test_split(test_size=test_size, seed=seed)
    train_ds = split_ds["train"]
    val_ds = split_ds["test"]

    def tokenize_func(examples):
        prompts = [
            f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
            f"### Instruction:\n{inst}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
            for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
        ]

        tokenized = tokenizer(prompts, truncation=True, max_length=context_size, padding=False)
        tokenized["labels"] = [ids.copy() for ids in tokenized["input_ids"]
        return tokenized

    train_tokenized = train_ds.map(tokenize_func, batched=True, remove_columns=train_ds.column_names)
    val_tokenized = val_ds.map(tokenize_func, batched=True, remove_columns=val_ds.column_names)
    
    return train_tokenized, val_tokenized

