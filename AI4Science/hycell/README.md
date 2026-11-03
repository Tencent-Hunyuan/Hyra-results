# HyCell

This directory contains Hyra's HyCell, a single-cell embedding model with 16.1 million parameters that maps
the raw counts of a human cell to a 512-dimensional embedding, and the `hycell` Python package that runs it.
HyCell is pretrained without cell-type labels on 79.5 million cells of the CZ CELLxGENE Census and is used as a
frozen encoder: embeddings of new cells serve for clustering, integration across batches,
annotation transfer and comparison of cell states. The decoder of the model turns embeddings back into gene
expression.

Technical report: [paper/HyCell_Technical_Report.pdf](paper/HyCell_Technical_Report.pdf).

## Recorded result

Frozen embeddings on seven atlases excluded from pretraining (immune, pancreas and lung from the scIB
benchmark, PBMC, prefrontal cortex, a cross-tissue immune atlas and the Asian Immune Diversity Atlas), every
model run with its released weights on the same cells and scored by the same code. Bio and Batch are the scIB
biological-conservation and batch-correction scores, LR and kNN the macro-F1 of annotation transfer to a
held-out batch, and S = 0.3 Bio + 0.2 Batch + 0.25 LR + 0.25 kNN. Means over the seven atlases:

| Model | Parameters | S | Bio | Batch | LR | kNN |
|---|---:|---:|---:|---:|---:|---:|
| **HyCell** | 16 M | 0.735 | **0.700** | 0.527 | 0.847 | 0.833 |
| Stack (batch-grouped inference) | 218 M | **0.736** | 0.657 | **0.596** | 0.841 | **0.839** |
| SCimilarity (trained with cell-type labels) | 31 M | 0.726 | 0.699 | 0.542 | 0.839 | 0.791 |
| scFoundation | 100 M | 0.702 | 0.655 | 0.436 | **0.849** | 0.823 |
| UCE | 675 M | 0.694 | 0.662 | 0.432 | 0.823 | 0.814 |
| PCA of the counts | – | 0.676 | 0.654 | 0.443 | 0.827 | 0.738 |

Among the eleven models, HyCell has the highest biological-conservation score (0.700; SCimilarity, trained with
cell-type labels, 0.699) and an overall score level with Stack, which has 14 times more parameters and uses batch
labels at inference to group cells.
Tahoe-x1, TranscriptFormer, scGPT, Geneformer V2, STATE and scPRINT score between 0.607 and 0.683; the full
comparison is in the [technical report](paper/HyCell_Technical_Report.pdf). The benchmark measures cell-level
tasks; Stack also draws on the other cells of a sample at inference, which helps sample-level tasks such as
predicting a donor's disease state, and these are not part of it.

Embedding runs at about 1,500 cells per second on one NVIDIA H20 for atlases of 40,000 cells, including reading
the file and tokenization. Results were recorded with Python 3.12, torch 2.7.1 and flash-attn 2.8.1
(`constraints.txt`).

## Method

A cell is an unordered set of gene tokens. Each expressed gene contributes the sum of a learned gene embedding
and a small network applied to its log1p(counts per 10,000); up to 818 expressed genes enter (the highest by
value), together with a quarter as many measured zeros, genes the dataset measures but the cell does not
express, and a [CLS] token. A six-layer transformer (width 256, four heads, SwiGLU, no positional encoding)
encodes the set, and the embedding is a linear projection of the final [CLS] state.

During training a count decoder reconstructs the full expression profile from the embedding under a
negative-binomial likelihood, conditioned on a learned code for the cell's dataset and donor, and a donor
discriminator on the embedding is trained against the encoder. Batch and donor information is used only in
training; an embedding is computed from the cell's own counts and the list of genes its dataset measures.
The model was trained for 6.4 × 10^10 tokens (9.9 hours on two NVIDIA H20 GPUs) on the human primary cells of
the Census release of 8 November 2025, with every dataset of the evaluation atlases removed.

## Install

Python 3.10 or newer, torch 2.4 or newer. A CUDA GPU is recommended. CPU inference works and is much slower,
about 13 cells per second with 16 threads in our tests.

```bash
pip install "git+https://github.com/Tencent-Hunyuan/Hyra-results.git#subdirectory=AI4Science/hycell"
pip install flash-attn --no-build-isolation      # optional, CUDA: the attention kernel used in training
```

The weights (64 MB, `src/hycell/checkpoints/hycell-16m/`) are part of the package, so HyCell runs without
network access once installed; `hycell info --verify` checks their sha256. `hycell.load(weights=...)` loads a
`.safetensors` file with its `.json` sidecar from any path.

## Embed

```python
import anndata as ad
import hycell

adata = ad.read_h5ad("cells.h5ad")        # raw counts in X (or pass layer= / use_raw=True)
model = hycell.load()                     # CUDA when available, otherwise CPU
adata.obsm["X_hycell"] = model.embed(adata)
```

`embed` also takes a path to an `.h5ad` file, which it reads in backed mode, cell block by cell block. The
embedding can be used like any other representation, for example with scanpy:

```python
import scanpy as sc
sc.pp.neighbors(adata, use_rep="X_hycell")
sc.tl.umap(adata)
sc.tl.leiden(adata)
```

HyCell embeds every cell on its own and is trained to discount donor and dataset effects, which suits comparing
cells across donors and studies. Properties shared by all cells of a donor, such as sex or disease state, are
better read from embeddings aggregated per donor or sample, for example their mean.

Command line:

```bash
hycell embed cells.h5ad -o embeddings.npy          # array + embeddings.obs_names.txt
hycell embed cells.h5ad -o cells_hycell.h5ad       # the input with obsm["X_hycell"]
```

## Decode

The decoder maps embeddings back to expected expression. `decode` returns counts per 10,000 over the genes
it is given, decoded with the model's code for an unknown dataset and donor:

```python
genes = model.measured_genes(adata)                 # vocabulary genes the dataset measures
expr = model.decode(adata.obsm["X_hycell"], genes)  # (cells, genes), counts per 10,000
```

Averaging decoded expression within two groups of cells gives a decoded fold change for every gene;
[docs/examples/decode_programs.py](docs/examples/decode_programs.py) does this and compares it with the fold
change of the counts.

## Fine-tune

With a labelled reference, HyCell can be fine-tuned into a cell-type annotator: the encoder and a linear head
on the embedding are trained with cross-entropy, starting from the pretrained weights.

```python
tuned = model.finetune(reference, "cell_type")     # one to three minutes on one GPU
query.obs["predicted"] = tuned.predict(query)
tuned.save("hycell_annotator")                      # hycell.load("hycell_annotator")
```

```bash
hycell finetune reference.h5ad --label-key cell_type -o hycell_annotator/
hycell predict query.h5ad --weights hycell_annotator/ -o predictions.csv
```

Annotating a held-out batch from the other batches of four benchmark atlases (macro-F1, mean over batches),
the fine-tuned annotator is ahead of the benchmark classifiers on frozen embeddings on every atlas:

| Atlas | Batches | Fine-tuned | Frozen + logistic regression | Frozen + kNN |
|---|---:|---:|---:|---:|
| Pancreas | 5 | **0.907** | 0.832 | 0.838 |
| Lung | 4 | **0.794** | 0.768 | 0.758 |
| PBMC | 2 | **0.966** | 0.961 | 0.936 |
| Immune | 9 | **0.877** | 0.857 | 0.873 |

See [docs/finetune.md](docs/finetune.md) for the procedure and its options.

## Inputs

HyCell reads raw integer counts. Genes are matched to its vocabulary of 36,495 human genes by Ensembl gene id
(a `var` column such as `ensembl_id`, `gene_ids` or `feature_id`, or Ensembl ids as `var_names`) and otherwise
by gene symbol; genes outside the vocabulary are dropped and duplicates are summed. Pass all measured genes,
not a highly-variable subset: the measured zeros are drawn from the genes present in the input. See
[docs/inputs.md](docs/inputs.md) for the details, determinism and the two attention implementations.

## Examples

[docs/examples](docs/examples): `embed_and_umap.py` (embedding and UMAP), `label_transfer.py` (kNN annotation
transfer between batches), `finetune_annotation.py` (fine-tuned annotator for a query batch),
`decode_programs.py` (decoded gene programs). `tests/data/lung_512.h5ad`, 512 cells of the scIB lung atlas with
cell types and batches, works with all four:

```bash
python docs/examples/decode_programs.py tests/data/lung_512.h5ad --groupby cell_type --a Macrophage --b "Type 2"
```

lists the surfactant genes SFTPC, SFTPB and SFTPA1 as the genes decoded highest in type 2 pneumocytes, and LYZ,
S100A8 and S100A9 as the highest in macrophages.

## Reproducibility

On a CUDA GPU with flash-attn, `hycell` reproduces the embeddings scored in the technical report bit for bit
(verified on all seven evaluation atlases, 167,804 cells). Without flash-attn, PyTorch attention gives the
same embeddings up to rounding (per-cell cosine similarity above 0.9999). `pytest tests` checks both against
reference embeddings of `tests/data/lung_512.h5ad`.

## Citation

See [CITATION.cff](CITATION.cff): Hyra Team, "HyCell: a Compact Embedding Model for Single-Cell
Transcriptomes", technical report, 2026 ([PDF](paper/HyCell_Technical_Report.pdf)).

## License

Apache-2.0, see the repository [LICENSE](../../LICENSE). The weights were trained on CZ CELLxGENE Census data
and the test cells come from the scIB lung atlas, both under CC BY 4.0; attributions are in [NOTICE](NOTICE).
