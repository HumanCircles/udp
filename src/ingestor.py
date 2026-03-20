"""Source detection and schema mapping into a universal contact format."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd


class DataIngestor:
    """Detects known source schemas and maps them into one universal schema."""

    UNIVERSAL_COLS: List[str] = [
        "name",
        "title",
        "company",
        "email",
        "phone",
        "linkedin",
        "industry",
        "num_employees",
        "source_type",
    ]

    @staticmethod
    def _log(logger: Optional[Callable[[str], None]], message: str) -> None:
        if logger:
            logger(message)
        else:
            print(message)

    @staticmethod
    def extract_employees(value: object) -> float:
        if pd.isna(value) or str(value).strip() == "":
            return np.nan
        nums = re.findall(r"\d+", str(value).replace(",", ""))
        return float(nums[-1]) if nums else np.nan

    @classmethod
    def load_and_map(
        cls, filepath: Path, logger: Optional[Callable[[str], None]] = None
    ) -> pd.DataFrame:
        df = pd.read_csv(filepath, low_memory=False, on_bad_lines="skip")
        cols = set(df.columns)

        # PhantomBuster / LinkedIn Sales Navigator exports
        if "fullPositions[0].title" in cols or "linkedinId" in cols:
            df = df.rename(
                columns={
                    "firstName": "first_name",
                    "lastName": "last_name",
                    "fullPositions[0].title": "title",
                    "fullPositions[0].companyName": "company",
                    "publicProfileUrl": "linkedin",
                    "fullPositions[0].companyIndustry": "industry",
                    "fullPositions[0].companyStaffCountRange": "num_employees",
                }
            )
            df["name"] = (
                df.get("first_name", "").fillna("").astype(str).str.strip()
                + " "
                + df.get("last_name", "").fillna("").astype(str).str.strip()
            ).str.strip()
            df["email"] = ""
            df["phone"] = ""
            df["source_type"] = "PhantomBuster"

        # Apollo exports
        elif "Departments" in cols and "Seniority" in cols:
            df = df.rename(
                columns={
                    "Name": "name",
                    "Title": "title",
                    "Company": "company",
                    "Email": "email",
                    "Linkedin Profile Link": "linkedin",
                    "Industry": "industry",
                    "# Employees": "num_employees",
                }
            )
            df["phone"] = ""
            df["source_type"] = "Apollo"

        # Aggregated exports (EasySource / RocketReach style)
        elif "Current Position" in cols and "Preferred Email" in cols:
            df = df.rename(
                columns={
                    "Name": "name",
                    "Current Position": "title",
                    "Current Organization": "company",
                    "Preferred Email": "email",
                    "Preferred Phone": "phone",
                    "LinkedIn": "linkedin",
                    "Industry": "industry",
                }
            )
            df["num_employees"] = np.nan
            df["source_type"] = "Aggregated"
        else:
            cls._log(logger, f"Skipping {os.path.basename(filepath)} - Unknown schema")
            return pd.DataFrame(columns=cls.UNIVERSAL_COLS)

        for col in cls.UNIVERSAL_COLS:
            if col not in df.columns:
                df[col] = ""

        if "num_employees" in df.columns and df["num_employees"].dtype == object:
            df["num_employees"] = df["num_employees"].apply(cls.extract_employees)

        return df[cls.UNIVERSAL_COLS]

