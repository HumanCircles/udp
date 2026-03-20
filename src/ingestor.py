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

        # Apollo exports (full — has Departments + Seniority columns)
        elif "Departments" in cols and "Seniority" in cols:
            df = df.rename(
                columns={
                    "Title": "title",
                    "Company": "company",
                    "Email": "email",
                    "Linkedin Profile Link": "linkedin",
                    "Person Linkedin Url": "linkedin",
                    "Industry": "industry",
                    "# Employees": "num_employees",
                }
            )
            # Build name from First Name + Last Name (full Apollo has no combined Name col)
            if "Name" in df.columns:
                df["name"] = df["Name"].fillna("").astype(str).str.strip()
            else:
                first = df.get("First Name", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
                last = df.get("Last Name", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
                df["name"] = (first + " " + last).str.strip()
            df["phone"] = df.get(
                "Work Direct Phone",
                df.get("Mobile Phone", pd.Series("", index=df.index))
            ).fillna("").astype(str)
            df["source_type"] = "Apollo"

        # Simplified Apollo/export — Name/Email/LinkedIn present, no Departments column
        # Handles BOM (\ufeff) variant of "First Name" that appears in many files
        elif "Email" in cols and ("LinkedIn" in cols or "Person Linkedin Url" in cols):
            # Strip BOM from column names
            df.columns = [c.lstrip("\ufeff") for c in df.columns]
            cols = set(df.columns)

            # Build name: prefer "Name" if present, else combine First + Last
            if "Name" in cols:
                df["name"] = df["Name"].fillna("").astype(str).str.strip()
            else:
                first = df.get("First Name", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
                last = df.get("Last Name", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
                df["name"] = (first + " " + last).str.strip()

            df["title"] = df.get("Title", pd.Series("", index=df.index)).fillna("").astype(str)
            df["company"] = df.get("Company", pd.Series("", index=df.index)).fillna("").astype(str)
            df["email"] = df.get("Email", pd.Series("", index=df.index)).fillna("").astype(str)
            df["phone"] = ""
            df["linkedin"] = df.get(
                "LinkedIn",
                df.get("Person Linkedin Url", pd.Series("", index=df.index))
            ).fillna("").astype(str)
            df["industry"] = df.get("Industry", pd.Series("", index=df.index)).fillna("").astype(str)
            df["num_employees"] = np.nan
            df["source_type"] = "Apollo-Simplified"

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

