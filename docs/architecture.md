# Architecture

The package is divided into contracts, data normalization, graph views and splits, retrieval, optional
HGT scoring, synthesis validation, evaluation helpers, and frozen-bundle rendering. Each boundary
accepts typed values and fails before passing malformed or unsupported state downstream.

`NormalizedSnapshot` is the shared source of truth. `build_graph_views()` derives both NetworkX and
PyTorch Geometric representations and reconciles their stable node and forward-edge identities.
Forward edges are citable; generated reverse twins are transport-only. The link-prediction split uses
only disease-target associations and retains reverse twins only for training edges.

Retrieval enumerates explicit typed relation sequences within fixed caps. Optional HGT scores may
reorder returned observed paths but cannot add candidates. Synthesis receives only the validated
intent, entity IDs, and evidence-path fields. The output gate permits one conservative supported claim
or a structured abstention.

The static surface loads a frozen bundle against a trusted expected snapshot hash. The bundle owns an
inventory of forward edge IDs and evidence paths, so answers cannot cite missing or reverse-only edges.
No network request, model load, graph build, or training operation occurs while rendering a bundle.
