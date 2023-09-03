# -*- coding: utf-8 -*-

import transformers

# 选择的model
model_checkpoint = "google/flan-t5-base"

from datasets import load_dataset

raw_datasets = load_dataset("csv", data_files="9.1_idf_csv_output.csv")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "Simulation: "
else:
    prefix = ""

max_input_length = 2048
max_target_length = 2048

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["Prompt"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    labels = tokenizer(text_target=examples["Idf"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
# 这里 将输出做了处理，然后也添加到了 input中

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

tokenized_datasets

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

batch_size = 16
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    f"{model_name}-finetuned-eplus",
    # evaluation_strategy = "epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    # per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=1,
    predict_with_generate=True,
    fp16=True,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
# 这个类提供了一种将输入序列（例如源文本）和目标序列（例如目标文本或生成的文本）组合成模型所需格式的方法。
# 它处理了一些与序列长度、填充和截断相关的细节，以确保数据对模型的输入和输出具有一致的形状。

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_datasets["train"],
    args=args,
    data_collator=data_collator
)

trainer.train()

# 准备测试文本
small_test_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(1))
testdata=small_test_dataset['Prompt']

text_idf = small_test_dataset["Idf"]
# text_idf

input_text=testdata
input_text

device = "cuda:0"
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

# 使用模型预测
outputs = model.generate(input_ids = inputs.input_ids,
                           attention_mask = inputs.attention_mask,
                           # generation_config = generation_config
                           )

# 使用模型解码输出
print(tokenizer.decode(outputs[0], skip_special_tokens=True))