from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, TrainerCallback, EvalPrediction
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import wandb
from datetime import datetime
import time
import numpy as np
import pandas as pd
from transformers.trainer_callback import TrainerControl, TrainerState

def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",  # 确保返回PyTorch张量
    )
    # 将padding部分的标签设置为-100
    labels = tokenized["input_ids"].clone()
    attention_mask = tokenized["attention_mask"]
    labels[attention_mask == 0] = -100  # 使用attention_mask将填充位置设为-100
    tokenized["labels"] = labels
    return tokenized

class MyTrainerCallback(TrainerCallback):
    log_cnt = 0

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.log_cnt += 1
        if self.log_cnt % 2 == 0:
            torch.cuda.empty_cache()

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        control.should_save = True
        return control

if __name__ == '__main__':
    model_name = "H:\\Weight\\Qwen2-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # 与训练参数bf16=True保持一致
        device_map="auto"
    )

    tm = time.strftime("%Y-%m-%d-%H-%M-%S")
    wandb.init(project="common_train", name=f"common_train_{tm}")

    training_args = TrainingArguments(
        output_dir="./output/qwen2_common_cpt",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=2,
        weight_decay=0.1,
        ddp_find_unused_parameters=False,
        warmup_steps=0,
        learning_rate=1e-4,
        evaluation_strategy="steps",
        eval_steps=100,
        save_steps=125,
        save_total_limit=2,
        report_to="wandb", # 使用wandb进行实验跟踪
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=20,
        log_level="info",
        logging_first_step=True,
    )



    train_dataset = load_dataset('json', data_files='./data/data.jsonl')["train"]
    eval_dataset = load_dataset('json', data_files='./data/eval.jsonl')["train"]

    tokenized_dataset = train_dataset.map(tokenize_function, batched=True)

    tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

    my_trainer_callback = MyTrainerCallback()
    # 训练模型
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_eval_dataset,
        callbacks=[my_trainer_callback],
    )

    trainer.train()
    trainer.evaluate()
    # loss_log = pd.DataFrame(trainer.state.log_history)
    # loss_log.to_csv(f"./logs/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")
