#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np


def l2_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return vec
    return vec / norm


def load_axes(axis_json_path: Path):
    with axis_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    axes = data.get("axes", [])
    if not axes:
        raise ValueError(f"No axes found in {axis_json_path}")

    return axes


def build_encoder(model_path: str):
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_path)

        def encode(sentences):
            # normalize_embeddings=False so we can explicitly normalize per spec.
            return model.encode(
                sentences,
                convert_to_numpy=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )

        return encode
    except Exception:
        pass

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModel.from_pretrained(model_path)
        model.eval()

        def mean_pooling(last_hidden_state, attention_mask):
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            summed = torch.sum(last_hidden_state * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            return summed / counts

        def encode(sentences):
            with torch.no_grad():
                inputs = tokenizer(
                    sentences,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                outputs = model(**inputs)
                pooled = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
                return pooled.cpu().numpy()

        return encode
    except Exception as e:
        raise RuntimeError(
            "Cannot build encoder. Install sentence-transformers or transformers+torch."
        ) from e


def main():
    parser = argparse.ArgumentParser(description="Build orthogonalized philosophy base axes.")
    parser.add_argument(
        "--axis-json",
        default=r"..\\100_axis.json",
        help="Path to axis json file.",
    )
    parser.add_argument(
        "--model-path",
        default=r"..\\bge-large-zh-v1.5",
        help="Local embedding model path.",
    )
    parser.add_argument(
        "--output",
        default=r"..\\100_base_d8.npy",
        help="Output npy path.",
    )
    args = parser.parse_args()

    axis_json_path = Path(args.axis_json)
    output_path = Path(args.output)

    axes = load_axes(axis_json_path)
    encode = build_encoder(args.model_path)

    axis_raw_list = []

    for axis in axes:
        a_sentences = axis.get("A", {}).get("sentences", [])
        b_sentences = axis.get("B", {}).get("sentences", [])
        axis_name = axis.get("axis_name", f"axis_{axis.get('axis_id', '?')}")

        if not a_sentences or not b_sentences:
            raise ValueError(f"Axis {axis_name} missing A/B sentences")

        a_vecs = np.asarray(encode(a_sentences), dtype=np.float32)
        b_vecs = np.asarray(encode(b_sentences), dtype=np.float32)

        a_vecs = np.vstack([l2_normalize(v) for v in a_vecs])
        b_vecs = np.vstack([l2_normalize(v) for v in b_vecs])

        axis_raw = b_vecs.mean(axis=0) - a_vecs.mean(axis=0)
        axis_raw = l2_normalize(axis_raw)
        axis_raw_list.append(axis_raw.astype(np.float32))

    axis_raw_mat = np.vstack(axis_raw_list)

    # QR orthogonalization on transposed matrix.
    q, _ = np.linalg.qr(axis_raw_mat.T)
    axis_ortho = q.T.astype(np.float32)

    if axis_ortho.shape[0] != len(axes):
        raise RuntimeError(
            f"Unexpected axis count after orthogonalization: {axis_ortho.shape}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(output_path), axis_ortho)

    # Basic sanity check for orthogonality.
    gram = axis_ortho @ axis_ortho.T
    max_offdiag = float(np.max(np.abs(gram - np.eye(gram.shape[0], dtype=gram.dtype))))

    print(f"Saved: {output_path}")
    print(f"AxisRaw shape: {axis_raw_mat.shape}")
    print(f"AxisOrtho shape: {axis_ortho.shape}")
    print(f"Max |Q*Q^T - I|: {max_offdiag:.6e}")


if __name__ == "__main__":
    main()
