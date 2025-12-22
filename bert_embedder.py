from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


LOGGER = logging.getLogger("bert_embedder")


def set_global_seed(seed: int) -> None:
    """Set seeds for Python, NumPy, and Torch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(verbosity: int) -> None:
    """Configure logging level."""
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_rows(json_path: str) -> List[Dict[str, Any]]:
    """
    Load rows from JSON file.

    Accepts:
      - {"rows": [...]}  (dict with "rows")
      - [...]            (list)
    """
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    if isinstance(obj, dict) and "rows" in obj and isinstance(obj["rows"], list):
        rows = obj["rows"]
    elif isinstance(obj, list):
        rows = obj
    else:
        raise ValueError(
            f"Unsupported JSON structure in {json_path}. Expected list or dict with 'rows'."
        )

    # Ensure each row is a dict (best-effort).
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        if isinstance(r, dict):
            out.append(r)
        else:
            LOGGER.warning("Skipping non-dict row at index %d in %s", i, json_path)
    return out


def build_text_map(
    rows: Sequence[Mapping[str, Any]],
    key_field: str,
    text_field: str,
    *,
    deduplicate: bool = True,
    cast_key_to_str: bool = True,
    on_empty: str = "zero",  # {"zero","skip","error"}
) -> Dict[str, str]:
    """
    Build a mapping from key -> text, with optional deduplication and empty handling.

    on_empty:
      - "zero": keep entry; downstream will emit a zero vector
      - "skip": drop entry
      - "error": raise
    """
    out: Dict[str, str] = {}
    seen: set[str] = set()

    for r in rows:
        if key_field not in r:
            continue
        key = r[key_field]
        if key is None:
            continue
        k = str(key) if cast_key_to_str else key  # type: ignore[assignment]
        if deduplicate and k in seen:
            continue

        text = r.get(text_field, "")
        if text is None:
            text = ""
        if not isinstance(text, str):
            text = str(text)

        if len(text.strip()) == 0:
            if on_empty == "skip":
                continue
            if on_empty == "error":
                raise ValueError(f"Empty text for key={k} (field '{text_field}')")
            # on_empty == "zero": keep; will encode to zeros later
            LOGGER.info("Empty text for key=%s; will emit zero vector.", k)

        out[k] = text
        seen.add(k)

    return out


@dataclass(frozen=True)
class EncodingConfig:
    model_name: str
    pooling: str  # {"mean", "cls"}
    max_length: int
    batch_size: int
    exclude_special_tokens: bool
    layer: str  # {"last", "last4_mean"}
    device: str
    fp16: bool


class BERTTextEncoder:
    """
    A thin, explicit wrapper around a BERT-family encoder for embedding extraction.

    Output embedding dimension equals the encoder hidden size.
    """

    def __init__(self, cfg: EncodingConfig) -> None:
        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(cfg.model_name)
        self.model.eval()

        self.device = torch.device(cfg.device)
        self.model.to(self.device)

        # Token IDs for special tokens (may be None depending on tokenizer/model).
        self.cls_id = self.tokenizer.cls_token_id
        self.sep_id = self.tokenizer.sep_token_id

        LOGGER.info("Loaded model=%s on device=%s", cfg.model_name, cfg.device)

    def _pool(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pool token embeddings into a single vector per sequence.

        hidden: [B, T, H]
        attention_mask: [B, T] (1=valid token, 0=padding)

        Returns: [B, H]
        """
        pooling = self.cfg.pooling.lower().strip()
        if pooling == "cls":
            # By convention, [CLS] is position 0.
            return hidden[:, 0, :]

        if pooling != "mean":
            raise ValueError(f"Unsupported pooling: {self.cfg.pooling}")

        mask = attention_mask

        if self.cfg.exclude_special_tokens:
            # Exclude [CLS] and [SEP] from pooling if token ids are available.
            # This is a common practice when you want the embedding to represent
            # the content tokens rather than special markers.
            if self.cls_id is not None:
                mask = mask * (input_ids != self.cls_id).long()
            if self.sep_id is not None:
                mask = mask * (input_ids != self.sep_id).long()

        # Mean pooling with mask:
        # e = sum_i (m_i * h_i) / sum_i m_i
        mask_f = mask.unsqueeze(-1).type_as(hidden)  # [B, T, 1]
        summed = torch.sum(hidden * mask_f, dim=1)   # [B, H]
        counts = torch.clamp(mask_f.sum(dim=1), min=1e-9)  # [B, 1]
        return summed / counts

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """
        Encode a list of texts into a matrix of embeddings.

        Returns:
          np.ndarray of shape [N, H], dtype float32 (or float16 if fp16)
        """
        if len(texts) == 0:
            return np.zeros((0, self.model.config.hidden_size), dtype=np.float32)

        all_vecs: List[np.ndarray] = []

        dtype_ctx = torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=(self.cfg.fp16 and self.device.type == "cuda")
        )

        for start in range(0, len(texts), self.cfg.batch_size):
            batch_texts = texts[start : start + self.cfg.batch_size]

            # Tokenize with padding/truncation.
            enc = self.tokenizer(
                list(batch_texts),
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc["attention_mask"].to(self.device)

            with dtype_ctx:
                if self.cfg.layer == "last":
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
                    hidden = out.last_hidden_state  # [B, T, H]
                elif self.cfg.layer == "last4_mean":
                    out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                    hs = out.hidden_states
                    if hs is None or len(hs) < 5:
                        raise RuntimeError("Model did not return enough hidden states for last4_mean.")
                    # hidden_states includes embeddings + each layer; we average the last 4 transformer layers.
                    last4 = torch.stack(hs[-4:], dim=0)  # [4, B, T, H]
                    hidden = torch.mean(last4, dim=0)    # [B, T, H]
                else:
                    raise ValueError(f"Unsupported layer option: {self.cfg.layer}")

                pooled = self._pool(input_ids=input_ids, attention_mask=attention_mask, hidden=hidden)  # [B, H]

            vecs = pooled.detach().cpu().numpy()
            # Use float32 for stable JSON export and downstream numeric stability.
            vecs = vecs.astype(np.float32, copy=False)
            all_vecs.append(vecs)

        return np.concatenate(all_vecs, axis=0)


def write_json_mapping(path: str, mapping: Mapping[str, Sequence[float]], *, indent: int = 2) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=indent)


def encode_mapping(
    encoder: BERTTextEncoder,
    key_to_text: Mapping[str, str],
    *,
    on_empty: str = "zero",
) -> Dict[str, List[float]]:
    """
    Encode a key->text mapping to key->embedding.

    on_empty:
      - "zero": empty/whitespace-only strings -> zero vectors
      - "skip": omit keys whose text is empty
      - "error": raise
    """
    keys: List[str] = []
    texts: List[str] = []

    for k, t in key_to_text.items():
        if t is None:
            t = ""
        if len(t.strip()) == 0:
            if on_empty == "skip":
                continue
            if on_empty == "error":
                raise ValueError(f"Empty text for key={k}")
        keys.append(k)
        texts.append(t)

    if len(keys) == 0:
        return {}

    # Encode non-empty texts in one pass.
    vecs = encoder.encode(texts)  # [N, H]
    dim = vecs.shape[1]

    out: Dict[str, List[float]] = {}
    for i, k in enumerate(keys):
        t = texts[i]
        if len(t.strip()) == 0 and on_empty == "zero":
            out[k] = [0.0] * dim
        else:
            out[k] = vecs[i].tolist()
    return out


def resolve_device(device: Optional[str]) -> str:
    if device:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract academic-grade BERT embeddings from mashup/api JSON datasets."
    )

    # Matching your notebook-like I/O:
    p.add_argument("--mashup-json", type=str, default=None, help="Path to mashup JSON (e.g., app.json).")
    p.add_argument("--api-json", type=str, default=None, help="Path to api JSON (e.g., api.json).")
    p.add_argument("--out-mashup", type=str, default="bert_mashup_des.json", help="Output JSON for mashup embeddings.")
    p.add_argument("--out-api", type=str, default="bert_api_des.json", help="Output JSON for api embeddings.")

    # Generic mode:
    p.add_argument("--input", type=str, default=None, help="Generic single input JSON path.")
    p.add_argument("--output", type=str, default=None, help="Generic single output JSON path.")
    p.add_argument("--key-field", type=str, default=None, help="Generic key field for --input (e.g., id/name).")
    p.add_argument("--text-field", type=str, default=None, help="Generic text field for --input (e.g., category).")

    # Field defaults for notebook-like datasets:
    p.add_argument("--mashup-key-field", type=str, default="id", help="Key field in mashup rows.")
    p.add_argument("--api-key-field", type=str, default="name", help="Key field in api rows.")
    p.add_argument("--default-text-field", type=str, default="category", help="Text field name (default for both).")

    # Model / encoding config
    p.add_argument("--model-name", type=str, default="bert-base-uncased",
                   help='HF model name. For Chinese, consider "bert-base-chinese".')
    p.add_argument("--pooling", type=str, default="mean", choices=["mean", "cls"],
                   help="Pooling strategy to produce a single vector per text.")
    p.add_argument("--exclude-special-tokens", action="store_true",
                   help="Exclude [CLS]/[SEP] tokens from mean pooling.")
    p.add_argument("--layer", type=str, default="last", choices=["last", "last4_mean"],
                   help="Which layer representation to pool.")
    p.add_argument("--max-length", type=int, default=128, help="Max sequence length for tokenizer truncation.")
    p.add_argument("--batch-size", type=int, default=64, help="Batch size for encoding.")
    p.add_argument("--device", type=str, default=None, help='Device string ("cpu", "cuda", "cuda:0").')
    p.add_argument("--fp16", action="store_true", help="Use fp16 autocast on CUDA for speed.")

    # Data handling
    p.add_argument("--on-empty", type=str, default="zero", choices=["zero", "skip", "error"],
                   help="How to handle empty text fields.")

    # Repro / logging
    p.add_argument("--seed", type=int, default=42, help="Global random seed.")
    p.add_argument("-v", "--verbose", action="count", default=0, help="Increase verbosity (-v or -vv).")

    args = p.parse_args()

    # Basic validation for generic mode.
    if args.input or args.output or args.key_field or args.text_field:
        if not (args.input and args.output and args.key_field and args.text_field):
            raise SystemExit(
                "Generic mode requires: --input --output --key-field --text-field"
            )

    # Notebook-like mode requires at least one of mashup/api.
    if not any([args.mashup_json, args.api_json, args.input]):
        raise SystemExit("Provide either --mashup-json/--api-json or generic --input/--output.")

    return args


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)
    set_global_seed(args.seed)

    device = resolve_device(args.device)

    cfg = EncodingConfig(
        model_name=args.model_name,
        pooling=args.pooling,
        max_length=args.max_length,
        batch_size=args.batch_size,
        exclude_special_tokens=bool(args.exclude_special_tokens),
        layer=args.layer,
        device=device,
        fp16=bool(args.fp16),
    )

    encoder = BERTTextEncoder(cfg)

    # Generic single-file mode
    if args.input:
        rows = load_rows(args.input)
        key_to_text = build_text_map(
            rows,
            key_field=args.key_field,
            text_field=args.text_field,
            on_empty=args.on_empty,
        )
        LOGGER.info("Generic input: %d entries after filtering.", len(key_to_text))
        out_map = encode_mapping(encoder, key_to_text, on_empty=args.on_empty)
        write_json_mapping(args.output, out_map)
        LOGGER.info("Wrote %d embeddings to %s", len(out_map), args.output)
        return

    # Notebook-like mode (mashup/api)
    text_field = args.default_text_field

    if args.mashup_json:
        mashup_rows = load_rows(args.mashup_json)
        mashup_map = build_text_map(
            mashup_rows,
            key_field=args.mashup_key_field,
            text_field=text_field,
            on_empty=args.on_empty,
        )
        LOGGER.info("Mashup: %d entries after filtering.", len(mashup_map))
        mashup_emb = encode_mapping(encoder, mashup_map, on_empty=args.on_empty)
        write_json_mapping(args.out_mashup, mashup_emb)
        LOGGER.info("Wrote %d mashup embeddings to %s", len(mashup_emb), args.out_mashup)

    if args.api_json:
        api_rows = load_rows(args.api_json)
        api_map = build_text_map(
            api_rows,
            key_field=args.api_key_field,
            text_field=text_field,
            on_empty=args.on_empty,
        )
        LOGGER.info("API: %d entries after filtering.", len(api_map))
        api_emb = encode_mapping(encoder, api_map, on_empty=args.on_empty)
        write_json_mapping(args.out_api, api_emb)
        LOGGER.info("Wrote %d api embeddings to %s", len(api_emb), args.out_api)


if __name__ == "__main__":
    main()
