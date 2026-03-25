"""Rule-based + optional semantic ICP scoring engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src.config import HR_KEYWORDS, ICP_TITLES, MODEL_NAME, REJECT_KEYWORDS


@dataclass
class SemanticArtifacts:
    model: object
    icp_vectors: np.ndarray
    index: Optional[object]
    device: str


class ICPEngine:
    def __init__(
        self,
        enable_semantic: bool = True,
        logger: Optional[Callable[[str], None]] = None,
    ):
        self.logger = logger
        self.semantic: Optional[SemanticArtifacts] = None
        if enable_semantic:
            self.semantic = self._load_semantic_stack()

    def _log(self, message: str) -> None:
        if self.logger:
            self.logger(message)
        else:
            print(message)

    def _detect_device(self) -> str:
        try:
            import torch
        except Exception:
            self._log("PyTorch not available, using CPU for semantic scoring.")
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_semantic_stack(self) -> Optional[SemanticArtifacts]:
        try:
            from sentence_transformers import SentenceTransformer
        except Exception as exc:
            self._log(f"Semantic stack unavailable, using rules only: {exc}")
            return None

        device = self._detect_device()
        self._log(f"Loading semantic model on device: {device}")
        model = SentenceTransformer(MODEL_NAME, device=device)
        icp_vecs = model.encode(ICP_TITLES, normalize_embeddings=True).astype("float32")
        index: Optional[object] = None

        try:
            import faiss

            index = faiss.IndexFlatIP(icp_vecs.shape[1])
            if device == "cuda" and hasattr(faiss, "StandardGpuResources"):
                try:
                    res = faiss.StandardGpuResources()
                    index = faiss.index_cpu_to_gpu(res, 0, index)
                    self._log("Using FAISS GPU index (NVIDIA acceleration enabled).")
                except Exception:
                    self._log("FAISS GPU transfer failed, using FAISS CPU index.")
            index.add(icp_vecs)
        except Exception:
            self._log("FAISS not available; using NumPy cosine fallback for similarity.")

        return SemanticArtifacts(
            model=model,
            icp_vectors=icp_vecs,
            index=index,
            device=device,
        )

    @staticmethod
    def _base_rule_score(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["title"] = out["title"].fillna("").astype(str).str.lower().str.strip()
        out["industry"] = out["industry"].fillna("").astype(str).str.lower().str.strip()

        hr_pat = "|".join(HR_KEYWORDS)
        out["hr_in_title"] = out["title"].str.contains(hr_pat, na=False, regex=True) | out[
            "title"
        ].str.contains(r"\bhuman resource\b", na=False, regex=True)

        reject_pat = "|".join(REJECT_KEYWORDS)
        out["bad_title"] = out["title"].str.contains(reject_pat, na=False, regex=True) & ~out[
            "hr_in_title"
        ]

        senior_pat = (
            r"\bceo\b|\bcoo\b|\bcfo\b|\bcto\b|\bcio\b|\bchro\b|\bcpo\b|"
            r"founder|owner|co-owner|\bvp\b|\bsvp\b|\bavp\b|\bevp\b|"
            r"vice president|director|head of|president|managing director|general manager|"
            r"manager|partner"
        )
        out["is_senior"] = out["title"].str.contains(senior_pat, na=False, regex=True)

        out["struct_score"] = (
            out["hr_in_title"].astype(int) * 8
            + out["is_senior"].astype(int) * 3
            - out["bad_title"].astype(int) * 15
        )
        return out

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        scored = self._base_rule_score(df)

        hard_reject_mask = scored["bad_title"]
        rejects = scored[hard_reject_mask].copy()
        rejects["sem_score"] = 0.0
        rejects["final_score"] = 0.0
        rejects["bucket"] = "REJECT"

        to_score = scored[~hard_reject_mask].copy()
        if len(to_score) == 0:
            return rejects

        to_score["sem_score"] = 0.0
        if self.semantic is not None:
            self._log(f"Embedding {len(to_score)} titles...")
            vecs = self.semantic.model.encode(
                to_score["title"].tolist(),
                batch_size=256,
                normalize_embeddings=True,
                show_progress_bar=True,
            ).astype("float32")
            if self.semantic.index is not None:
                scores, _ = self.semantic.index.search(vecs, k=1)
                to_score["sem_score"] = scores[:, 0]
            else:
                sims = np.matmul(vecs, self.semantic.icp_vectors.T)
                to_score["sem_score"] = sims.max(axis=1)

        max_s = to_score["struct_score"].clip(lower=0).max() or 1
        if self.semantic is None:
            to_score["final_score"] = to_score["struct_score"].clip(lower=0) / max_s
        else:
            to_score["final_score"] = (0.65 * to_score["sem_score"]) + (
                0.35 * (to_score["struct_score"].clip(lower=0) / max_s)
            )

        core_hr_ind = to_score["industry"].str.contains(
            r"staffing|recruiting|human resources|hr tech", na=False, regex=True
        )
        founder_owner_mask = to_score["title"].str.contains(
            r"\bfounder\b|co-founder|\bowner\b|co-owner", na=False, regex=True
        )
        # C-suite pattern (titles that are hiring decision makers regardless of dept)
        csuite_mask = to_score["title"].str.contains(
            r"\bceo\b|\bcoo\b|\bcfo\b|\bchro\b|\bcpo\b"
            r"|chief executive|chief operating|chief financial"
            r"|chief technology|chief information(?! security)"
            r"|managing director|general manager"
            r"|\bpresident\b",
            na=False, regex=True,
        )

        # Modest penalty for non-HR founders/owners outside HR industry
        to_score.loc[
            founder_owner_mask & ~to_score["hr_in_title"] & ~core_hr_ind, "final_score"
        ] -= 0.10

        # Leadership rescue: keep strong senior profiles out of hard reject.
        # Excludes obviously non-buyer/student-type profiles.
        leadership_mask = to_score["title"].str.contains(
            r"\bsvp\b|\bavp\b|\bevp\b|\bvp\b|vice president|director|head of|"
            r"\bcfo\b|\bcto\b|\bcio\b|\bcoo\b|\bceo\b|chief|president|managing director|general manager",
            na=False,
            regex=True,
        )
        non_buyer_mask = to_score["title"].str.contains(
            r"student|intern|assistant|coach|rabbi|professor|teacher|advisor|consultant",
            na=False,
            regex=True,
        )
        leadership_rescue_mask = leadership_mask & ~non_buyer_mask & ~to_score["bad_title"]

        to_score["bucket"] = "REJECT"
        to_score.loc[to_score["final_score"] >= 0.47, "bucket"] = "REVIEW"
        to_score.loc[to_score["final_score"] >= 0.54, "bucket"] = "ACCEPT"

        # HR-keyword title → ACCEPT directly
        to_score.loc[
            to_score["hr_in_title"] & (to_score["final_score"] >= 0.45), "bucket"
        ] = "ACCEPT"
        # Confirmed C-suite → ACCEPT (they own hiring budgets)
        to_score.loc[csuite_mask & ~to_score["bad_title"], "bucket"] = "ACCEPT"
        # Strong leadership should at least be REVIEW, often ACCEPT.
        to_score.loc[leadership_rescue_mask & (to_score["sem_score"] >= 0.58), "bucket"] = "REVIEW"
        to_score.loc[leadership_rescue_mask & (to_score["sem_score"] >= 0.66), "bucket"] = "ACCEPT"
        # Founders/owners in HR/staffing industry → ACCEPT
        to_score.loc[founder_owner_mask & core_hr_ind, "bucket"] = "ACCEPT"

        return pd.concat([to_score, rejects], ignore_index=True)

