"""
Vanilla Graph Convolutional Network (Kipf & Welling, ICLR 2017)
=================================================================
Reference: T. N. Kipf & M. Welling, "Semi-Supervised Classification with
Graph Convolutional Networks", ICLR 2017.
           W. L. Hamilton, "Graph Representation Learning", Ch. 5.

This is a from-scratch PyTorch implementation of a single GCN layer
(no PyTorch Geometric, no DGL) built directly from the layer-wise
propagation rule:

    H^(l+1) = sigma( D~^{-1/2} A~ D~^{-1/2} H^(l) W^(l) ),   A~ = A + I

Dataset: Zachary's Karate Club -- the exact 34-node social network Kipf &
Welling used in their original blog/paper demo. It ships inside networkx,
so this script needs NO external download.

Task: semi-supervised node classification into the 2 factions the club
split into, using only ONE labeled node per class (2 labels total, out
of 34 nodes) -- the classic "extreme low-label" regime that demonstrates
how much a GCN can infer purely from graph structure.

Run:
    pip install torch networkx matplotlib
    python gcn_karate_club.py
"""
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)
np.random.seed(0)


# ---------------------------------------------------------------------------
# 1. Build Â = D~^{-1/2} (A + I) D~^{-1/2}  (the "renormalization trick")
# ---------------------------------------------------------------------------
def build_normalized_adj(G: nx.Graph) -> torch.Tensor:
    A = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
    N = A.shape[0]
    A_tilde = A + torch.eye(N)                       # add self-loops
    deg = A_tilde.sum(dim=1)                          # D~_ii
    D_inv_sqrt = torch.diag(deg.pow(-0.5))
    return D_inv_sqrt @ A_tilde @ D_inv_sqrt           # Â, shape (N, N)


# ---------------------------------------------------------------------------
# 2. A single Vanilla GCN layer:  H_out = sigma( Â  H_in  W )
# ---------------------------------------------------------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, A_hat: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        return A_hat @ self.W(H)


class VanillaGCN(nn.Module):
    """Two-layer GCN. The final layer is 2-dimensional so its output
    doubles as both the classification logits AND a directly-plottable
    2-D node embedding (the classic Kipf & Welling visualization)."""

    def __init__(self, n_nodes: int, hidden_dim: int = 4, n_classes: int = 2):
        super().__init__()
        self.gc1 = GCNLayer(n_nodes, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, n_classes)

    def forward(self, A_hat, X):
        H1 = F.relu(self.gc1(A_hat, X))
        Z = self.gc2(A_hat, H1)          # logits == 2-D embedding
        return Z


# ---------------------------------------------------------------------------
# 3. Data
# ---------------------------------------------------------------------------
G = nx.karate_club_graph()
N = G.number_of_nodes()
A_hat = build_normalized_adj(G)
X = torch.eye(N)                          # no node features -> learn from structure alone
labels = torch.tensor(
    [0 if G.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(N)]
)

train_idx = torch.tensor([0, 33])         # 1 labeled node per class
train_lbl = labels[train_idx]

# ---------------------------------------------------------------------------
# 4. Train (Adam, cross-entropy on just the 2 labeled nodes)
# ---------------------------------------------------------------------------
model = VanillaGCN(n_nodes=N, hidden_dim=4, n_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

for epoch in range(1, 301):
    model.train()
    optimizer.zero_grad()
    Z = model(A_hat, X)
    loss = F.cross_entropy(Z[train_idx], train_lbl)
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0 or epoch == 1:
        with torch.no_grad():
            preds = Z.argmax(dim=1)
            acc = (preds == labels).float().mean().item()
        print(f"epoch {epoch:3d} | loss {loss.item():.4f} | "
              f"accuracy on ALL 34 nodes (only 2 were labeled) = {acc*100:5.1f}%")

# ---------------------------------------------------------------------------
# 5. Final evaluation + the signature embedding plot
# ---------------------------------------------------------------------------
model.eval()
with torch.no_grad():
    Z = model(A_hat, X)
    preds = Z.argmax(dim=1)
    acc = (preds == labels).float().mean().item()

print(f"\nFINAL accuracy on all 34 nodes (trained on just 2 labels): {acc*100:.1f}%")
misclassified = (preds != labels).nonzero().flatten().tolist()
print("Misclassified nodes:", misclassified)

Z_np, labels_np = Z.numpy(), labels.numpy()
colors = np.where(labels_np == 0, "#4C72B0", "#DD8452")
plt.figure(figsize=(6, 5.5))
plt.scatter(Z_np[:, 0], Z_np[:, 1], c=colors, s=140, edgecolors="k", zorder=3)
for i in range(N):
    plt.annotate(str(i), (Z_np[i, 0], Z_np[i, 1]), fontsize=7, ha="center", va="center")
for i in train_idx.tolist():
    plt.scatter(*Z_np[i], s=400, facecolors="none", edgecolors="green", linewidths=2.5)
for i in misclassified:
    plt.scatter(*Z_np[i], s=400, facecolors="none", edgecolors="red", linewidths=2.5)
plt.title(f"Vanilla GCN — 2-D embedding of Karate Club\n"
          f"trained on 2 labels -> {acc*100:.1f}% accuracy")
plt.xlabel("embedding dim 1"); plt.ylabel("embedding dim 2")
plt.tight_layout()
plt.savefig("gcn_embedding_pytorch.png", dpi=160)
print("Saved gcn_embedding_pytorch.png")
