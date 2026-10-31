"""SR-GNN: directed gated message passing and a hybrid session readout.

Graphs and their first-occurrence aliases follow ``build_session_graph_batch``:
incoming and outgoing adjacency rows contain normalized neighbour weights.
Attention is the original unnormalized sigmoid attention, not a softmax.
"""

from __future__ import annotations

import math
from typing import ClassVar

import torch
from torch import nn
from torch.nn import functional as F

from seqrec.data.dataset import (
    EvalBatch,
    SessionGraphTrainBatch,
    build_session_graph_batch,
)
from seqrec.models.base import (
    Paradigm,
    SequentialRecommender,
    gather_last_valid,
    prepend_pad_column,
)


class SRGNN(SequentialRecommender):
    """Session-graph recommender with shared input/output item embeddings.

    ``ggnn_steps`` controls graph propagation depth. Training consumes the
    prebuilt graphs from ``SessionGraphDataset``; evaluation builds graphs from
    the current history only. No future or target item enters the graph.
    """

    paradigm: ClassVar[Paradigm] = "graph"

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 50,
        hidden_size: int = 64,
        ggnn_steps: int = 1,
        dropout: float = 0.2,
        loss_type: str = "ce",
        share_item_embeddings: bool = True,
    ) -> None:
        super().__init__(num_items)
        if min(num_items, max_seq_len, hidden_size, ggnn_steps) < 1:
            raise ValueError(
                "Item count, window, hidden size and GGNN steps must be positive."
            )
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1).")
        if loss_type != "ce" or not share_item_embeddings:
            raise ValueError("SR-GNN requires CE loss and shared item embeddings.")
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.ggnn_steps = ggnn_steps
        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.incoming = nn.Linear(hidden_size, hidden_size)
        self.outgoing = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.GRUCell(2 * hidden_size, hidden_size)
        self.local_attention = nn.Linear(hidden_size, hidden_size)
        self.node_attention = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attention_score = nn.Linear(hidden_size, 1, bias=False)
        self.hybrid = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Use the SR-GNN uniform initialization and keep the PAD row zero."""
        bound = 1 / math.sqrt(self.hidden_size)
        for parameter in self.parameters():
            nn.init.uniform_(parameter, -bound, bound)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def encode_graph(
        self,
        graph_items: torch.Tensor,
        alias_inputs: torch.Tensor,
        adj_in: torch.Tensor,
        adj_out: torch.Tensor,
        seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[B, d]`` hybrid session states using batched graph operations."""
        batch_size, width = graph_items.shape
        if width < 1 or bool(((seq_len < 1) | (seq_len > width)).any()):
            raise ValueError(
                "SR-GNN requires nonempty right-padded histories of length <= N."
            )
        node_mask = (graph_items != 0).unsqueeze(-1)
        nodes = self.dropout(self.item_embedding(graph_items))
        for _ in range(self.ggnn_steps):
            incoming = self.incoming(torch.bmm(adj_in, nodes))
            outgoing = self.outgoing(torch.bmm(adj_out, nodes))
            messages = torch.cat((incoming, outgoing), dim=-1)
            nodes = self.gate(
                messages.reshape(-1, 2 * self.hidden_size),
                nodes.reshape(-1, self.hidden_size),
            ).reshape(batch_size, width, self.hidden_size)
            nodes = nodes * node_mask
        states = nodes.gather(
            1, alias_inputs.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        )
        local = gather_last_valid(states, seq_len)
        attention = self.attention_score(
            torch.sigmoid(
                self.local_attention(local).unsqueeze(1) + self.node_attention(states)
            )
        )
        valid = torch.arange(width, device=seq_len.device).unsqueeze(
            0
        ) < seq_len.unsqueeze(1)
        global_state = (attention * states * valid.unsqueeze(-1)).sum(dim=1)
        return self.dropout(self.hybrid(torch.cat((local, global_state), dim=-1)))

    def project(self, sessions: torch.Tensor) -> torch.Tensor:
        """Score real items using the same projection during training and inference."""
        return sessions @ self.item_embedding.weight[1:].T

    def compute_loss(self, batch: SessionGraphTrainBatch) -> torch.Tensor:
        """Compute next-item full-vocabulary CE over prebuilt session graphs."""
        sessions = self.encode_graph(
            batch["graph_items"],
            batch["alias_inputs"],
            batch["adj_in"],
            batch["adj_out"],
            batch["seq_len"],
        )
        return F.cross_entropy(self.project(sessions).float(), batch["target_id"] - 1)

    def score_items(self, batch: EvalBatch) -> torch.Tensor:
        """Build graphs from evaluation histories and return ``[B, V+1]`` logits."""
        graph_items, aliases, adj_in, adj_out = build_session_graph_batch(
            batch["input_ids"], batch["seq_len"]
        )
        sessions = self.encode_graph(
            graph_items, aliases, adj_in, adj_out, batch["seq_len"]
        )
        return prepend_pad_column(self.project(sessions))
