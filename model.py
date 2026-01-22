import torch
import torch.nn.functional as F
import torch.nn as nn
import math
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# =====================================================
# 🔹  Embeding_layer
# =====================================================
max_len=128
vocab_len=97
class Embeding_layer(nn.Module):
    def __init__(self,
                 vocab_size=256,
                 d_model=128,
                 max_len=100,
                 n_out=128,):
        super().__init__()

        # Byte embedding layer (0–255)
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Positional embeddings
        #self.pos_embedding = nn.Embedding(max_len, d_model)
        # Final normalization
        
        self.projection = nn.Linear(in_features=d_model, out_features=n_out)
        self.norm = nn.LayerNorm(n_out)
        self.activation = nn.GELU()

    def forward(self, x):
        """
        x: (batch, seq_len) — byte indices [0–255]
        """
        #positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        x = self.embedding(x) #+ self.pos_embedding(positions)

        x = self.projection(x)
        x = self.norm(x)
        x = self.activation(x)

        return x  # (B, L, d_model)
# 🔹 Residual Depthwise-Separable Multi-Kernel Block
class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_sizes=[3],stride=1, dilations=[1], reduction=16, num_groups=32):
        super().__init__()

        #mid_ch = max(in_ch // 16, 8)  # reduce dimension before heavy convs
        self.branches = nn.ModuleList()

        for k in kernel_sizes:
            if k <= 5:
                for d in dilations:
                    branch = nn.Sequential(
                        # (B) Reduce channels first
                        #nn.Conv1d(in_ch, 1, kernel_size=1, bias=False),
                        
                        #nn.GroupNorm(num_groups=8, num_channels=mid_ch),
                        #nn.GELU(),
                        #nn.Dropout1d(0.25),

                        # (A) Depthwise conv
                        nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=(k*d-1) // 2, bias=False, stride=stride, dilation=d),


                        # Pointwise to expand to out_ch
                        #nn.Conv1d(mid_ch, out_ch, kernel_size=1, bias=False),
                        #nn.GroupNorm(num_groups=8, num_channels=out_ch),
                        #nn.GELU()
                    )
                    self.branches.append(branch)
        
        for k in kernel_sizes:
            if k > 5:
                branch = nn.Sequential(
                        nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=(k) // 2, bias=False, stride=stride, dilation=1),
                    )
                self.branches.append(branch)
        
        # Combine all kernel branches
        self.merge_conv = nn.Conv1d(out_ch * (len(self.branches)), out_ch, kernel_size=1, bias=False)
        #self.merge_bn = nn.BatchNorm1d(out_ch)
        #self.se = SEBlock(out_ch, reduction)
        self.shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride) if (in_ch != out_ch or stride!=1) else nn.Identity()
        self.group_norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_ch)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout1d(0.25)

    def forward(self, x):
        # Parallel multi-kernel branches
        out = [branch(x) for branch in self.branches]
        out = torch.cat(out, dim=1)

        out = self.merge_conv(out)
        
        #out = self.se(out)
        out += self.shortcut(x)
        out = self.group_norm(out)
        out = self.gelu(out)
        out = self.dropout(out)
        return F.relu(out)

    

class DualAttentionPooling(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()

        # Temporal attention (softmax over T)
        self.temporal_attn = nn.Linear(channels, 1)

        # Channel attention (SE-style)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        # x: (B, T, C)

        # ---- Temporal attention ----
        t_scores = self.temporal_attn(x)            # (B, T, 1)
        t_weights = torch.softmax(t_scores, dim=1)
        x = x * t_weights                           # (B, T, C)

        # ---- Channel attention ----
        c_context = x.mean(dim=1)                   # (B, C)
        c_weights = torch.sigmoid(
            self.fc2(F.gelu(self.fc1(c_context)))
        )                                           # (B, C)

        x = x * c_weights.unsqueeze(1)              # (B, T, C)

        # ---- Pool ----
        return x.sum(dim=1)                         # (B, C)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: [B, T, d_model]
        return x + self.pe[:, :max_len]
class TransformerBlock(nn.Module):
    def __init__(self, d_model=64, n_heads=4, ff_dim=256, num_layers=2, dropout=0.1):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x, mask=None):
        return self.encoder(x, src_key_padding_mask=mask)

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x):
        # x: (B, T, C)
        weights = torch.softmax(self.attn(x), dim=1)
        return (weights * x).sum(dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import json
class URLBinaryCNN_bestmodel(nn.Module):
    def __init__(self, maxlen=max_len, vocab_size = vocab_len,d_model=128, embed_dim=64):
        super().__init__()
        # Shared Layer (Global)
        self.shared_layer = nn.ModuleDict({
            "embeding":  Embeding_layer(vocab_size=vocab_size, max_len=maxlen, d_model=d_model, n_out=embed_dim),
            "conv": ResidualConvBlock(embed_dim, 64, kernel_sizes=[3,5,7], num_groups=8),
            #"conv2": ResidualConvBlock(64, 32, kernel_sizes=[3,5,7], num_groups=8),
            #"conv3": ResidualConvBlockDW(64, 32, kernel_sizes=[3,5,7]),
            #"conv4": ResidualConvBlockDW(128, 64, kernel_sizes=[3,5,7]),
            #"conv5": ResidualConvBlockDW(64, 32, kernel_sizes=[3,5,7]),

            #"proj": nn.Linear(64, 128),
            #"pos_enc": PositionalEncoding(d_model=64, max_len=maxlen),
            #"transformer": TransformerBlock(d_model=64, n_heads=4, ff_dim=256, num_layers=1),
            "bilstm": nn.LSTM(input_size=64, hidden_size=64,num_layers=1, batch_first=True, bidirectional=True),
            "layer_norm": nn.LayerNorm(64*2),
            "gelu1": nn.GELU(),
            #"postconv": nn.Conv1d(64, 32, kernel_size=3),
            #"maxpool": nn.MaxPool1d(2),
            #"avg_pool1": nn.AvgPool1d(kernel_size=2),
            "attentionpooling": DualAttentionPooling(128),
            "fc1": nn.Linear(128 , 64),
            "layer_normalization1": nn.LayerNorm(64),
            "gelu2": nn.GELU(),
            "dropout1": nn.Dropout(0.25),

            
            
            
        })

        # Personalization Layer (Local)
        self.personal_layer = nn.ModuleDict({
            
            "fc2": nn.Linear(64, 48),
            "gelu3": nn.GELU(),
            #"dropout2": nn.Dropout(0.25),
            "head": nn.Linear(48, 1)
        })

    def forward(self, x):
        # Shared layers
        x = self.shared_layer["embeding"](x)
        x = x.permute(0, 2, 1)

        x = self.shared_layer["conv"](x)
        #x = self.shared_layer["conv2"](x)
        #x = self.shared_layer["conv3"](x)
        #x = self.shared_layer["max_pool"](x)
        x = x.permute(0, 2, 1)
        
        #x = self.shared_layer["proj"](x)          # [B, T, 64]
        #x = self.shared_layer["pos_enc"](x)
        #x = self.shared_layer["transformer"](x)
        x, _ = self.shared_layer["bilstm"](x)
        x = self.shared_layer["layer_norm"](x)
        x = self.shared_layer["gelu1"](x)
        #x = x.permute(0, 2, 1)
        #x = self.shared_layer["postconv"](x)
        #x = self.shared_layer["maxpool"](x)
        #x = self.shared_layer["avg_pool1"](x)
        #x = x.permute(0, 2, 1)
        #x = x[:, -1, :]
        #x = x.flatten(1)
        x = self.shared_layer["attentionpooling"](x)
        x = self.shared_layer["dropout1"](self.shared_layer["gelu2"](self.shared_layer["layer_normalization1"](self.shared_layer["fc1"](x))))
        # Personalization head
        x = self.personal_layer["gelu3"](self.personal_layer["fc2"](x)) #self.personal_layer["dropout2"](
        x = self.personal_layer["head"](x)
        return x
