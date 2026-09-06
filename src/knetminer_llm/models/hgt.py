from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch_geometric.nn import HGTConv


@dataclass(frozen=True)
class HGTConfig:
    input_dims: dict[str, int]
    hidden_dim: int = 128
    heads: int = 4
    layers: int = 2

    def __post_init__(self) -> None:
        if self.hidden_dim != 128 or self.heads != 4 or self.layers != 2:
            raise ValueError("Project 06 HGT requires 128 hidden dimensions, four heads, and two layers")
        if set(self.input_dims) != {"disease", "target", "drug"}:
            raise ValueError("HGT input dimensions must cover exactly the three node types")
        if any(dimension != 384 for dimension in self.input_dims.values()):
            raise ValueError("Project 06 HGT requires 384-dimensional inputs")


class HGTReranker(nn.Module):
    """HGT scorer over an existing heterogeneous graph; it never creates edges."""

    def __init__(self, metadata, config: HGTConfig) -> None:
        super().__init__()
        self.config = config
        self.projections = nn.ModuleDict(
            {
                node_type: nn.Linear(input_dim, config.hidden_dim)
                for node_type, input_dim in config.input_dims.items()
            }
        )
        self.convs = nn.ModuleList(
            [
                HGTConv(
                    in_channels=config.hidden_dim,
                    out_channels=config.hidden_dim,
                    metadata=metadata,
                    heads=config.heads,
                )
                for _ in range(config.layers)
            ]
        )
        self.residuals = nn.ModuleDict(
            {
                node_type: nn.Linear(config.hidden_dim, config.hidden_dim)
                for node_type in config.input_dims
            }
        )

    def forward(
        self,
        x_dict: dict[str, Tensor],
        edge_index_dict: dict[tuple[str, str, str], Tensor],
    ) -> dict[str, Tensor]:
        hidden = {
            node_type: torch.relu(self.projections[node_type](features))
            for node_type, features in x_dict.items()
        }
        for conv in self.convs:
            if not edge_index_dict:
                continue
            messages = conv(hidden, edge_index_dict)
            hidden = {
                node_type: torch.relu(
                    messages.get(node_type, torch.zeros_like(features))
                    + self.residuals[node_type](features)
                )
                if node_type in messages
                else features
                for node_type, features in hidden.items()
            }
        return hidden

    def score_observed_pairs(
        self,
        embeddings: dict[str, Tensor],
        pairs: tuple[tuple[str, int, str, int], ...],
    ) -> Tensor:
        scores: list[Tensor] = []
        for source_type, source_index, target_type, target_index in pairs:
            source = embeddings[source_type][source_index]
            target = embeddings[target_type][target_index]
            scores.append(torch.sum(source * target).reshape(1))
        if not scores:
            return next(iter(embeddings.values())).new_empty((0,))
        return torch.cat(scores)
