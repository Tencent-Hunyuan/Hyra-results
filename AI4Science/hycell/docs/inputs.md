# Inputs and determinism

## Counts

HyCell reads raw integer counts, one row per cell: `adata.X` by default, a layer with `layer="counts"`, or
`adata.raw` with `use_raw=True`. Sparse (CSR/CSC) and dense matrices are accepted, and an `.h5ad` path is read in
backed mode one block of 4,096 cells at a time. Negative values are rejected; non-integer values (normalised or
log-transformed data) raise a warning, because the model was trained on counts.

## Genes

The vocabulary (`hycell.load_genes()`) has 36,495 human genes: every protein-coding gene and every other gene
detected in at least 0.1% of the pretraining cells. Input genes are matched by Ensembl gene id, with version
suffixes ignored. HyCell looks for Ensembl ids in the `var` columns `ensembl_id`, `gene_ids`, `gene_id`,
`feature_id`, `ensembl`, `ensembl_gene_id` and `gene_ensembl`, then in `var_names`. Without Ensembl ids it
matches gene symbols (`var_names` or a `feature_name`, `gene_symbol`, `gene_symbols`, `gene_name` or `symbol`
column), exactly and then case-insensitively; a symbol shared by several vocabulary genes resolves to the
protein-coding one. `gene_key=` names the column explicitly (`"index"` for `var_names`).

Genes outside the vocabulary are dropped, and the counts of input genes that map to the same vocabulary gene
are summed. Library size and counts per 10,000 are computed over the vocabulary genes.

## Tokens

Each expressed gene becomes a token whose value is log1p(counts per 10,000). A cell with more than 818
expressed genes keeps the 818 with the highest values. Measured zeros, a quarter as many as the expressed genes
(rounded), are drawn from the core genes, the 10,131 genes measured in at least 90% of the pretraining datasets,
that the input measures and the cell does not express. Restricting zeros to core genes keeps a dataset's gene
panel from marking its cells. A [CLS] token leads the set, so a cell has at most 1,024 tokens. A cell without
any vocabulary counts is encoded from the [CLS] token alone.

Because the zeros come from the genes present in the input, pass all measured genes rather than a subset of
highly variable genes.

## Determinism

The tie breaks among equal values and the measured-zero draws come from a random generator seeded with
`(seed, row)`, where `row` is the cell's position in the input (`seed=0` by default). An embedding therefore
does not depend on the other cells or on how they are batched, and the same file always gives the same
embeddings. A subset of a file gives the embeddings of the full file when the cells' row numbers are passed
as `rows=`. A different seed draws other zeros. On the lung and pancreas atlases of the benchmark, seeds 0 and 1
give per-cell cosine similarities of at least 0.998 (median 0.9999).

## Attention and precision

`hycell.load(attention=...)` selects one of two implementations of the same network:

- `flash` (default on CUDA when flash-attn is installed): cells are packed into sequences of up to
  `batch_tokens` tokens (65,536) and attention runs per cell with flash-attention's variable-length kernel, in
  bfloat16 autocast. This is the configuration of training and evaluation; it reproduces the benchmark
  embeddings bit for bit.
- `sdpa` (any device): cells of similar length are padded into batches and attention runs with PyTorch's
  `scaled_dot_product_attention`. On CUDA it uses bfloat16 autocast, on CPU float32 (`precision=` overrides).
  The embeddings equal the `flash` ones up to rounding: on the seven evaluation atlases the per-cell cosine
  similarity is at least 0.99999 in bfloat16 and 0.99998 in float32.

## Decoder

`decode(z, genes)` computes the decoder logits of the given genes, normalises them with a softmax over those
genes and scales them to `library_size` (10,000 by default), so the output is the expected expression in counts
per 10,000 over that gene set. In training the softmax ran over the genes each dataset measured; decoding over
`model.measured_genes(adata)` reproduces this for a new dataset. The decoder uses the learned code for an
unknown dataset and donor, the code the model was trained to use for data outside its corpus.
`dispersion(genes)` returns the negative-binomial inverse dispersion of each gene.
