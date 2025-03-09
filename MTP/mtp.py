
# 加载模型和分词器
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config, Qwen2ForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import Trainer, TrainingArguments
import wandb
from datetime import datetime
import time

tm = time.strftime("%Y-%m-%d-%H-%M-%S")
wandb.init(project="mtp_train", name=f"mtp_train_{tm}")

# 定义新模型类
class Qwen2WithMTP(Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.mtp_linear = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.mtp_transformer_block = Qwen2DecoderLayer(config, layer_idx=0)

    def forward(self, input_ids, labels=None, return_dict=None, **kwargs):
        training = self.training

        position_ids = kwargs.get('position_ids', None)
        attention_mask = kwargs.get('attention_mask', None)

        base_outputs = self.model(input_ids, output_hidden_states=True, output_attentions=True, **kwargs)
        last_hidden_state = base_outputs.last_hidden_state
        # Compute main model's output
        main_logits = self.lm_head(last_hidden_state)
        main_loss = None
        if labels is not None:
            shift_logits = main_logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            main_loss = nn.functional.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        mtp_loss = 0.0  # 确保 mtp_loss 有默认值，防止 NoneType 加法错误
        T = input_ids.size(1)
        
        if training and T >= 3:
            # 准备MTP输入
            embed_t_i1 = self.model.embed_tokens(input_ids[:, 1:T-1])
            concat_input = torch.cat([last_hidden_state[:, :T-2, :], embed_t_i1], dim=-1)
            projected_input = self.mtp_linear(concat_input)

            # 构造MTP需要的position_ids和attention_mask
            seq_length = T - 2
            if position_ids is not None:
                mtp_position_ids = position_ids[:, :seq_length]
            else:
                mtp_position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
                mtp_position_ids = mtp_position_ids.unsqueeze(0).expand(input_ids.size(0), -1)

            if attention_mask is not None:
                mtp_attention_mask = attention_mask[:, :seq_length]
            else:
                mtp_attention_mask = None

            # query_states = projected_input  # 使用投影后的结果作为query

            cos, sin = self.model.rotary_emb(projected_input, mtp_position_ids)
            
            position_embeddings = (cos, sin)

            # 调用DecoderLayer时显式传递旋转编码
            mtp_hidden = self.mtp_transformer_block(
                hidden_states=projected_input,
                attention_mask=mtp_attention_mask,
                position_ids=mtp_position_ids,
                position_embeddings=position_embeddings,
                past_key_value=None,
                use_cache=False
            )

            # print(mtp_hidden[0].shape)
            # MTP损失计算
            mtp_logits = self.lm_head(mtp_hidden[0])
            labels_mtp = input_ids[:, 2:T]
            
            # print(mtp_logits.shape, ' and ', labels_mtp.shape) torch.Size([2, 510, 151936])  and  torch.Size([2, 510])
            mtp_loss = nn.functional.cross_entropy(
                mtp_logits.contiguous().view(-1, mtp_logits.size(-1)),
                labels_mtp.contiguous().view(-1)
                )

        total_loss = main_loss + self.config.lambda_mtp * mtp_loss if main_loss is not None else self.config.lambda_mtp * mtp_loss
        # Return the appropriate output
        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=main_logits,
            past_key_values=base_outputs.past_key_values,
            hidden_states=base_outputs.hidden_states,
            attentions=base_outputs.attentions,
        )
# 配置模型

model_name = "H:\\Weight\\Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
config = Qwen2Config.from_pretrained(model_name)
# # print(config)
# # exit()
config.lambda_mtp = 0.3

new_model = Qwen2WithMTP(config)

new_model.load_state_dict(model.state_dict(), strict=False)

# 准备数据集（持续预训练示例）
dataset = load_dataset('json', data_files='./test_data.jsonl')["train"]

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./output/qwen2_mtp",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=10,
    weight_decay=0.1,
    warmup_steps=0,
    learning_rate=1e-5,
    report_to="wandb", # 使用wandb进行实验跟踪
    optim="adamw_torch",
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=20,
    log_level="info",
    logging_first_step=True,
)

# 训练模型
trainer = Trainer(
    model=new_model,
    # tokenizer=tokenizer,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()

# 保存模型
trainer.save_model("./output/weight")

# # 推理示例
trained_model = AutoModelForCausalLM.from_pretrained("./output/weight")
trained_model.eval()
prompt = "你是谁？"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids
output = trained_model.generate(input_ids, max_length=100)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)