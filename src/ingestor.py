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
        "emp_lower",
        "emp_upper",
        "source_type",
    ]

    @staticmethod
    def _log(logger: Optional[Callable[[str], None]], message: str) -> None:
        if logger:
            logger(message)
        else:
            print(message)

    @staticmethod
    def parse_employee_range(value: object) -> tuple[float, float]:
        """Return (lower, upper) numeric bounds from a free-text headcount field.

        Handles common formats:
          "1001-5000", "1,001-5,000" → (1001.0, 5000.0)
          "10001+"                   → (10001.0, inf)
          "1500"                     → (1500.0, 1500.0)
          NaN / ""                   → (nan, nan)
        """
        if pd.isna(value) or str(value).strip() in ("", "nan"):
            return np.nan, np.nan
        s = str(value).replace(",", "").strip()
        if s.endswith("+"):
            nums = re.findall(r"\d+", s)
            return (float(nums[0]), float("inf")) if nums else (np.nan, np.nan)
        nums = re.findall(r"\d+", s)
        if not nums:
            return np.nan, np.nan
        lower = float(nums[0])
        upper = float(nums[-1]) if len(nums) > 1 else lower
        return lower, upper

    @staticmethod
    def extract_employees(value: object) -> float:
        """Legacy single-value extraction — returns the upper bound of a range."""
        if pd.isna(value) or str(value).strip() == "":
            return np.nan
        nums = re.findall(r"\d+", str(value).replace(",", ""))
        return float(nums[-1]) if nums else np.nan

    @classmethod
    def load_and_map(
        cls, filepath: Path, logger: Optional[Callable[[str], None]] = None
    ) -> pd.DataFrame:
        df = pd.read_csv(filepath, low_memory=False, on_bad_lines="skip")
        # Normalize BOM-prefixed headers once for all schemas
        df.columns = [c.lstrip("\ufeff") for c in df.columns]
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
        elif "Email" in cols and (
            "LinkedIn" in cols
            or "LinkedIn Url" in cols
            or "Linkedin Url" in cols
            or "Person Linkedin Url" in cols
        ):

            # Build name: prefer "Name" if present, else combine First + Last
            if "Name" in cols:
                df["name"] = df["Name"].fillna("").astype(str).str.strip()
            else:
                first = df.get("First Name", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
                last = df.get("Last Name", pd.Series("", index=df.index)).fillna("").astype(str).str.strip()
                df["name"] = (first + " " + last).str.strip()

            df["title"] = df.get("Title", pd.Series("", index=df.index)).fillna("").astype(str)
            df["company"] = df.get(
                "Company",
                df.get("Company Name", pd.Series("", index=df.index))
            ).fillna("").astype(str)
            df["email"] = df.get("Email", pd.Series("", index=df.index)).fillna("").astype(str)
            df["phone"] = ""
            df["linkedin"] = df.get(
                "LinkedIn",
                df.get(
                    "LinkedIn Url",
                    df.get(
                        "Linkedin Url",
                        df.get("Person Linkedin Url", pd.Series("", index=df.index))
                    ),
                ),
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

        # Preserve raw num_employees string for display, then derive both bounds.
        raw_emp = df["num_employees"].copy()
        if raw_emp.dtype == object:
            df["num_employees"] = raw_emp.apply(cls.extract_employees)

        bounds = raw_emp.apply(cls.parse_employee_range)
        df["emp_lower"] = bounds.apply(lambda t: t[0])
        df["emp_upper"] = bounds.apply(lambda t: t[1])

        return df[cls.UNIVERSAL_COLS]

