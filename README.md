
# PCNet
# Personalized Complementary Network for Personalized High-order Complementary Service Recommendation
⭐ This code has been released for reproducibility ⭐

⭐ Overall framework of the PCNet model ⭐



Overall framework of the PCNet model.

⭐ The HGA dataset refers to our previous work: https://github.com/528Lab/CAData ⭐  
⭐ The PWA dataset refers to: https://github.com/kkfletch/API-Dataset ⭐

---

### Environment

- **Python**: 3.9.16  
- **PyTorch**: 2.0.1  
- **Transformers**: 4.3x.x  
- Common dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `tqdm`


### BERT Embeddings
We extract textual embeddings for Mashups/APIs using `bert_embedder.py` (HuggingFace `AutoTokenizer` / `AutoModel`).
- Default model: `bert-base-uncased.` 
- Tokenization: padding + truncation with `--max-length 128`.
- Representation: `--layer last`.
- Pooling: `--pooling mean` (masked mean pooling) or `cls`; `--exclude-special-tokens` is optional.
- Empty text: outputs a zero vector (`--on-empty zero`).
- Outputs: `bert_mashup_des.json` and `bert_api_des.json` (key -> embedding list).

Example:
```bash
python bert_embedder.py \
  --mashup-json data/mashup.json \
  --api-json data/api.json \
  --model-name bert-base-uncased \
  --layer last \
  --pooling mean \
  --max-length 128 \
  --batch-size 64 \
  --seed 42
```

### Random Seed (Sample Generation & Training)
To facilitate fair comparison, we fix the random seed in the released code:
- Sample generation / data split: `seed=2024`.
- Training: `seed=3088` (sets Python/NumPy/PyTorch seeds and enables deterministic cuDNN).
