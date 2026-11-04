# Fine-tuning for cell-type annotation

`model.finetune(reference, label_key)` turns HyCell into a cell-type annotator for a labelled reference
dataset. The encoder and a linear classification head on the embedding are trained together with
cross-entropy, starting from the pretrained weights:

- each step draws 64 labelled cells and tokenizes them with the pretraining augmentations (a random gene
  panel with probability 0.3, binomial thinning with probability 0.5, and the 818-gene cap drawn by value);
- AdamW without weight decay, learning rate 1e-5 for the network and 1e-3 for the head (initialised at zero),
  gradient clipping at 1.0, bfloat16 autocast on CUDA;
- 10% of the labelled cells are held out; their loss is computed every 100 steps, the weights with the lowest
  loss are kept, and training stops after five evaluations without improvement or after 2,000 steps.

Cells without a label are ignored. The returned model embeds cells like the base model (with the fine-tuned
encoder) and predicts labels:

```python
import hycell

model = hycell.load()
tuned = model.finetune(reference, "cell_type")
query.obs["predicted"] = tuned.predict(query)
probabilities = tuned.predict(query, return_probabilities=True)   # DataFrame, cells x classes
tuned.save("hycell_annotator")                                    # weights.safetensors + weights.json
annotator = hycell.load("hycell_annotator")
```

Command line:

```bash
hycell finetune reference.h5ad --label-key cell_type -o hycell_annotator/
hycell predict query.h5ad --weights hycell_annotator/ -o predictions.csv
```

`steps`, `cells_per_step`, `lr`, `head_lr`, `validation_fraction`, `eval_every`, `patience` and `seed` change
the procedure; the defaults are the settings validated below.

## Validation

Leave-one-batch-out annotation on four atlases of the benchmark: for every batch, the annotator is fine-tuned
on the other batches with the default settings and predicts the held-out batch. The baselines are the
classifiers of the benchmark on the frozen embeddings of the same cells, logistic regression (C = 1, features
scaled) and 15-nearest neighbours. Macro-F1 over the cell types of the held-out batch, mean over batches:

| Atlas | Batches | Fine-tuned | Frozen + logistic regression | Frozen + kNN |
|---|---:|---:|---:|---:|
| Pancreas (inDrop, Smart-seq2) | 5 | **0.907** | 0.832 | 0.838 |
| Lung (Drop-seq) | 4 | **0.794** | 0.768 | 0.758 |
| PBMC (10x) | 2 | **0.966** | 0.961 | 0.936 |
| Immune (bone marrow and blood, 5 studies) | 9 | **0.877** | 0.857 | 0.873 |

The fine-tuned annotator is ahead of logistic regression on 19 of the 20 held-out batches and of kNN on 17.
The gain is largest on the pancreas atlas, whose batches come from different technologies. Fine-tuning took
one to three minutes per fold on one NVIDIA H20 (10,000 to 31,000 reference cells).
