"""Data cleaning and cross-source deduplication utilities."""

from __future__ import annotations

import pandas as pd


class DataCleaner:
    """Normalizes contact keys and deduplicates records by trust order."""

    @staticmethod
    def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
        clean = df.copy()
        clean["name"] = clean["name"].fillna("").astype(str).str.strip()
        clean["company"] = clean["company"].fillna("").astype(str).str.strip()
        clean["email"] = clean["email"].fillna("").astype(str).str.lower().str.strip()
        clean["linkedin"] = (
            clean["linkedin"].fillna("").astype(str).str.lower().str.strip()
        )

        # Priority: email > linkedin > (name + company)
        has_email = clean[clean["email"] != ""].drop_duplicates("email")
        no_email = clean[clean["email"] == ""]

        has_linkedin = no_email[no_email["linkedin"] != ""].drop_duplicates("linkedin")
        no_linkedin = no_email[no_email["linkedin"] == ""].drop_duplicates(
            ["name", "company"]
        )

        return pd.concat([has_email, has_linkedin, no_linkedin], ignore_index=True)

