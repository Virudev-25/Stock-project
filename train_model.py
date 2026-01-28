import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# =========================
# Load Dataset
# =========================
df = pd.read_csv("noise_resilient_stock_data.csv")
df.sort_values("date", inplace=True)

features = [
    "open", "high", "low", "close", "volume",
    "ma5", "ma10", "rsi", "gold_price", "volatility"
]
sequence_length = 30

# =========================
# Dataset Class
# =========================
class StockDataset(Dataset):
    def __init__(self, df):
        self.X, self.y = [], []
        for i in range(len(df) - sequence_length):
            seq = df[features].iloc[i:i+sequence_length].values
            label = df["stock_trend"].iloc[i+sequence_length] == "UP"
            self.X.append(seq)
            self.y.append(int(label))

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = StockDataset(df)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# =========================
# MAMBA-STYLE BLOCK (Windows safe)
# =========================
class MambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.A = nn.Linear(d_model, d_model)
        self.B = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        h = torch.tanh(self.A(x))
        g = torch.sigmoid(self.B(x))
        return self.norm(h * g + x)

# =========================
# HYBRID MODEL
# =========================
class HybridTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.mamba = MambaBlock(d_model=len(features))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=len(features),
            nhead=2,
            dim_feedforward=64,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            self.encoder_layer,
            num_layers=1
        )

        self.fc = nn.Linear(len(features) * sequence_length, 2)

    def forward(self, x):
        x = self.mamba(x)
        x = self.transformer(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

# =========================
# Training
# =========================
model = HybridTransformer()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    total_loss = 0
    for X, y in loader:
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss/len(loader):.4f}")

# =========================
# Save Model
# =========================
torch.save(model.state_dict(), "hybrid_stock_model.pth")
print("✅ Model training completed and saved")
