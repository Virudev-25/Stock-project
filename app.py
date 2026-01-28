from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

app = Flask(__name__)
CORS(app)

# =========================
# Load Data
# =========================
df = pd.read_csv("noise_resilient_stock_data.csv")

features = [
    "open", "high", "low", "close", "volume",
    "ma5", "ma10", "rsi", "gold_price", "volatility"
]

SEQ_LEN = 30

# =========================
# MAMBA-STYLE BLOCK
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

        self.fc = nn.Linear(len(features) * SEQ_LEN, 2)

    def forward(self, x):
        x = self.mamba(x)
        x = self.transformer(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)

# =========================
# Load Model
# =========================
model = HybridTransformer()
model.load_state_dict(torch.load("hybrid_stock_model.pth", map_location="cpu"))
model.eval()

# =========================
# Helper Functions
# =========================
def get_random_sequence():
    start = np.random.randint(0, len(df) - SEQ_LEN)
    seq = df.iloc[start:start+SEQ_LEN][features].values
    return torch.tensor(seq, dtype=torch.float32)

def explain_prediction(seq, buy_prob):
    """
    Improved gradient × input attribution
    Produces mixed, non-uniform, signed contributions
    """

    seq = seq.unsqueeze(0)
    seq.requires_grad = True

    # Forward
    output = model(seq)
    prob = torch.softmax(output, dim=1)[0, 1]

    # Backward
    prob.backward()

    # Gradient × Input
    grads = seq.grad.detach().numpy()[0]        # (seq_len, features)
    inputs = seq.detach().numpy()[0]             # (seq_len, features)

    contrib = grads * inputs                     # element-wise
    contrib = contrib.sum(axis=0)                # aggregate over time

    # Normalize to percentage
    total = np.sum(np.abs(contrib)) + 1e-8
    contrib = (contrib / total) * 100

    explanations = {}
    for i, feat in enumerate(features):
        val = float(round(contrib[i], 2))        # Python float

        # Arrow based on sign
        arrow = "↑" if val >= 0 else "↓"
        explanations[feat] = f"{abs(val):.2f}{arrow}"

    return explanations



# =========================
# Routes
# =========================
@app.route("/predict", methods=["POST"])
def predict():
    seq = get_random_sequence()

    with torch.no_grad():
        output = model(seq.unsqueeze(0))
        probs = torch.softmax(output, dim=1)[0]

    buy_prob = probs[1].item()
    decision = "BUY" if buy_prob > 0.5 else "SELL"
    confidence = round(buy_prob if buy_prob > 0.5 else 1 - buy_prob, 2)

    factors = explain_prediction(seq, buy_prob)

    return jsonify({
        "decision": decision,
        "confidence": confidence,
        "factors": factors
    })

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Server running"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
