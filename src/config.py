"""Central configuration for the unified ICP data pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INPUT_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "output"

# Keywords
HR_KEYWORDS: List[str] = [
    "talent acquisition",
    "talent acq",
    "recruiter",
    "recruiting",
    "recruitment",
    "headhunter",
    "human resources",
    r"\bhr\b",
    "hrbp",
    "hr business partner",
    "people ops",
    "people operations",
    "people partner",
    "people team",
    "chief.*people",
    "chief.*human",
    "chief.*talent",
    "chief.*culture",
    "chro",
    "cpo",
    "head of.*talent",
    "head of.*hr",
    "head of.*people",
    "head of.*recruiting",
    "vp.*talent",
    "vp.*hr",
    "vp.*people",
    "director.*talent",
    "director.*hr",
    "director.*human",
    "director.*people",
    "talent manager",
    "talent management",
    "talent leader",
    "workforce",
    "staffing",
    "employer branding",
    "total rewards",
    "l&d",
    "learning.*development",
    "executive recruiter",
    "technical recruiter",
    "human capital",
    "learning officer",
    "clo",
]

REJECT_KEYWORDS: List[str] = [
    "sales",
    "account executive",
    "business development",
    "marketing",
    "seo",
    "finance",
    "cfo",
    "software engineer",
    "developer",
    "cto",
    "data analyst",
    "warehouse",
    "logistics",
    "physician",
    "nurse",
    "clinical",
    "real estate agent",
    "realtor",
    "lawyer",
]

ICP_TITLES: List[str] = [
    # Core HR / Talent titles
    "Director of Talent Acquisition",
    "VP Human Resources",
    "Chief People Officer",
    "CHRO",
    "Head of Talent Acquisition",
    "HR Manager",
    "Senior Recruiter",
    "Technical Recruiter",
    "Headhunter",
    "Human Capital Strategist",
    # C-suite decision makers (budget owners / hiring decision makers)
    "CEO",
    "Chief Executive Officer",
    "Founder and CEO",
    "Co-Founder and CEO",
    "Owner",
    "President and CEO",
    "COO",
    "Chief Operating Officer",
    "CFO",
    "Chief Financial Officer",
    "CTO",
    "Chief Technology Officer",
    "CIO",
    "Chief Information Officer",
    "Managing Director",
    "General Manager",
    "President",
]

MODEL_NAME = "BAAI/bge-small-en-v1.5"

