import torch
from transformers import AutoTokenizer

# -------------------------
# 配置本地模型路径
# -------------------------
model_path = "/hpai/aios3.0/private/user/s2025244189/s1025244189/Model_Env/Qwen2.5-Omni-7B"

# -------------------------
# 选择设备
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# 加载本地 tokenizer
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, repo_type="model")

# -------------------------
# 占位模型类，仅用于前向测试
# -------------------------
class MinimalQwenModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_ids, **kwargs):
        # 返回占位输出: batch x seq_len x hidden_dim (4096 占位)
        return torch.zeros(input_ids.shape[0], input_ids.shape[1], 4096, device=input_ids.device)

# -------------------------
# 初始化占位模型
# -------------------------
model = MinimalQwenModel().to(device)
model.eval()

# -------------------------
# 测试文本
# -------------------------
text = "Hello, this is a test."
inputs = tokenizer(text, return_tensors="pt").to(device)

# -------------------------
# 前向测试
# -------------------------
with torch.no_grad():
    outputs = model(**inputs)

print("Forward pass successful, output shape:", outputs.shape)