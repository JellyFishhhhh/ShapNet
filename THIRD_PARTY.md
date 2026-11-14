# Upstream attribution

`shapnet/vendor/` contains selected model files from the user's RRT-MIL checkout, based on commit `17141aa`. Source and copied-file hashes are in `docs/upstream_files.json`. Changes are relative package imports, a CPU-safe registered position-index buffer in `emb_position.py`, and honoring input_dim in MaxMIL.

Original project: Feature Re-Embedding: Towards Foundation Model-Level Performance in Computational Pathology (Tang et al., CVPR 2024), https://github.com/DearCaat/RRT-MIL . Existing source comments and attributions have been preserved. No new blanket license is assigned to upstream code; consult upstream authors for applicable terms. This refactor does not claim ownership of those implementations.
