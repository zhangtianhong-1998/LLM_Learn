from typing import Callable, List, Optional, Tuple, Union
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, TrainerCallback
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, KwargsForCausalLM
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
from transformers.cache_utils import Cache, DynamicCache
from transformers.processing_utils import Unpack

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

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-8):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(d_model))  

    def forward(self, x):
        # 计算均方根 (RMS)
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        x = x / rms

        return self.scale * x

class Qwen2WithMTP(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.mtp_linear = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.norm = RMSNorm(config.hidden_size)
        
        self.mtp_transformer_block = Qwen2DecoderLayer(config, layer_idx=0)

        self._initialize()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        training = self.training

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        last_hidden_state = hidden_states[:, slice_indices, :]
        logits = self.lm_head(last_hidden_state)
        # logits = Batch * seq_len * vocab_size   labels = Batch * seq_len
        loss = None
        # 交叉熵损失
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)


        # 开始实现 MTP的预测
        mtp_loss = 0.0  # 确保 mtp_loss 有默认值，防止 NoneType 加法错误

        T = last_hidden_state.size(1)
        shift_len = 1
        if training and T >= 3:
            embed_t_i1 = self.model.embed_tokens(input_ids[:, shift_len:])
            embed_t_i1 = self.norm(embed_t_i1)
            # ???????????????????????
            # print(last_hidden_state.shape)
            ls_hidden_state = self.norm(last_hidden_state[:, :-shift_len, :])
            # print(ls_hidden_state.shape)


            concat_input = torch.cat([ls_hidden_state, embed_t_i1], dim=-1)

            projected_input = self.mtp_linear(concat_input)

            # 构造MTP需要的position_ids和attention_mask
            seq_length = T - shift_len
            if position_ids is not None:
                mtp_position_ids = position_ids[:, shift_len:]
            else:
                mtp_position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
                mtp_position_ids = mtp_position_ids.unsqueeze(0).expand(input_ids.size(0), -1)

            if attention_mask is not None:
                # print(attention_mask.shape)
                mtp_attention_mask = attention_mask[:, shift_len:].unsqueeze(1).unsqueeze(2).to(torch.bfloat16)
            else:
                mtp_attention_mask = None

            position_embeddings = self.model.rotary_emb(projected_input, mtp_position_ids)

            # 调用DecoderLayer时显式传递旋转编码
            mtp_hidden = self.mtp_transformer_block(
                hidden_states=projected_input,
                attention_mask=mtp_attention_mask,
                position_ids=mtp_position_ids,
                position_embeddings=position_embeddings,
                past_key_value=None,
                use_cache=False
            )

            mtp_logits = self.lm_head(mtp_hidden[0]).contiguous()
            labels_mtp = labels[:, shift_len:].contiguous() 

            # 计算MTP损失
            mtp_loss = self.loss_function(
                logits=mtp_logits,
                labels=labels_mtp,
                vocab_size=self.config.vocab_size,
                **kwargs
            )

        total_loss = loss + self.config.lambda_mtp * mtp_loss if loss is not None else self.config.lambda_mtp * mtp_loss

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _initialize(self):
        last_layer_idx = len(self.model.layers) - 1
        self.mtp_transformer_block.load_state_dict(
            self.model.layers[0].state_dict(), 
            strict=True
        )

        std = self.config.initializer_range
        if isinstance(self.mtp_linear.weight, nn.Linear):
            self.mtp_linear.weight.data.normal_(mean=0.0, std=std)
            if self.mtp_linear.bias is not None:
                self.mtp_linear.bias.data.zero_()

if __name__ == '__main__':
    model_name = "H:\\Weight\\Qwen2-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # 与训练参数bf16=True保持一致
        device_map="auto"
    )
    
    config = Qwen2Config.from_pretrained(model_name)

    config.lambda_mtp = 0.3

    new_model = Qwen2WithMTP(config)

    new_model.load_state_dict(model.state_dict(), strict=False)
    new_model._initialize()


    tm = time.strftime("%Y-%m-%d-%H-%M-%S")
    wandb.init(project="mtp_train", name=f"mtp_train{tm}")

    training_args = TrainingArguments(
        output_dir="./output/qwen2_mtp",
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
        save_steps=250,
        save_total_limit=2,
        report_to="wandb", # 使用wandb进行实验跟踪
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=10,
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
        model=new_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_eval_dataset,
        callbacks=[my_trainer_callback],
    )

    trainer.train()
    trainer.evaluate()
    # loss_log = pd.DataFrame(trainer.state.log_history)
    # loss_log.to_csv(f"./logs/pre_train_log_{time.strftime('%Y%m%d-%H%M')}.csv")
