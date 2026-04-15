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
    r"\btalent\b.*\bpartners?\b",
    r"\bpeople\b.*\bculture\b.*\bpartners?\b",
    r"\bpeople\b.*\btalent\b.*\bpartners?\b",
    r"\bpeople\b.*\bbusiness partner\b",
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

# ── Timezone mapping (US state full name → sales timezone label) ───────────
STATE_TIMEZONE: dict[str, str] = {
    # Eastern
    "Connecticut": "ET", "Delaware": "ET", "District of Columbia": "ET",
    "Florida": "ET", "Georgia": "ET", "Indiana": "ET", "Kentucky": "ET",
    "Maine": "ET", "Maryland": "ET", "Massachusetts": "ET", "Michigan": "ET",
    "New Hampshire": "ET", "New Jersey": "ET", "New York": "ET",
    "North Carolina": "ET", "Ohio": "ET", "Pennsylvania": "ET",
    "Rhode Island": "ET", "South Carolina": "ET", "Tennessee": "ET",
    "Vermont": "ET", "Virginia": "ET", "West Virginia": "ET",
    "Washington DC": "ET",
    # Central
    "Alabama": "CT", "Arkansas": "CT", "Illinois": "CT", "Iowa": "CT",
    "Kansas": "CT", "Louisiana": "CT", "Minnesota": "CT", "Mississippi": "CT",
    "Missouri": "CT", "Nebraska": "CT", "North Dakota": "CT", "Oklahoma": "CT",
    "South Dakota": "CT", "Texas": "CT", "Wisconsin": "CT",
    # Mountain
    "Arizona": "MT", "Colorado": "MT", "Idaho": "MT", "Montana": "MT",
    "New Mexico": "MT", "Utah": "MT", "Wyoming": "MT",
    # Pacific
    "California": "PT", "Nevada": "PT", "Oregon": "PT", "Washington": "PT",
    # Other US territories
    "Alaska": "AKT", "Hawaii": "HT",
}

# ── Industry groups (used by dashboard checkboxes → dynamic regex) ────────
# Derived from scanning all 142 unique `Industry` field values across raw CSVs.
# Each regex is matched against the lowercase `industry` field value.
INDUSTRY_GROUPS: dict[str, str] = {
    # ── Core ICP targets ──────────────────────────────────────────────────
    "Healthcare & Medical": (
        r"hospital\b|health care|healthcare|medical practice|mental health|"
        r"medical devices|pharmaceut|biotech|clinical|dental|optom|"
        r"health.wellness|life science|nursing|rehab|therapeut|"
        r"individual.*family service"
    ),
    "Construction & Infrastructure": (
        r"construction\b|contractor|building material|civil engineering|"
        r"architecture|surveying|landscap|plumbing|roofing|flooring|"
        r"hvac|masonry|concrete|infrastructure"
    ),
    "Manufacturing & Industrial": (
        r"manufactur|mechanical.*industrial|industrial\b|machinery\b|"
        r"automotive|aerospace|fabricat|packaging|plastics|rubber|"
        r"metal|steel|electrical.*electronic.*manufactur|"
        r"food production|furniture|apparel|fashion|wholesale|"
        r"building material|consumer goods"
    ),
    "Retail & Consumer Goods": (
        r"\bretail\b|consumer goods|consumer services|e.?commerce|"
        r"food.*beverage|grocery|supermarket|apparel|fashion|"
        r"wholesale|merchandise|department store"
    ),
    "Software & Technology": (
        r"information technology|it services|software|computer.*network|"
        r"network security|computer games|internet|cloud|saas|"
        r"semiconductor|animation|information services"
    ),
    "Telecommunications & Media": (
        r"telecom|media production|entertainment|broadcast|"
        r"wireless|cable|satellite"
    ),
    "Insurance": (
        r"insurance\b|underwriting|reinsurance"
    ),
    "Financial Services": (
        r"financial services|investment management|investment banking|"
        r"\bbanking\b|accounting\b|wealth management|credit union|"
        r"private equity|venture capital|fintech"
    ),
    "Oil, Gas & Energy": (
        r"oil.*energy|energy\b|oil.*gas|petroleum|mining|utilities\b|"
        r"environmental services|renewable|solar|wind power"
    ),
    "Real Estate": (
        r"real estate\b|property management|realty|mortgage|"
        r"facilities services"
    ),
    # ── Secondary targets ─────────────────────────────────────────────────
    "Professional Services": (
        r"management consulting|professional training|research\b|"
        r"outsourc|business consulting"
    ),
    "Legal": (
        r"law practice|legal services|attorney|litigation"
    ),
    "Education": (
        r"higher education|primary.*secondary education|education management|"
        r"e.?learning|university|college|school"
    ),
    "Transportation & Logistics": (
        r"transportation|trucking|railroad|logistics.*supply|"
        r"supply chain|freight|shipping|aviation|maritime"
    ),
    "Government & Non-Profit": (
        r"government|public.*administration|military|nonprofit|"
        r"non.profit|ngo|public safety|civil service"
    ),
    "Hospitality & Arts": (
        r"hospitality\b|hotel|performing arts|fine art|arts.*crafts|"
        r"sports\b|recreation|museum|gallery"
    ),
    "Staffing & Recruiting": (
        r"staffing|recruiting|recruitment|executive search|"
        r"human resources\b|workforce solution|rpo\b|peo\b|"
        r"professional employer|talent agency|placement"
    ),
}

# ── Industry filters ───────────────────────────────────────────────────────
# Regex applied against the normalised `industry` field (lowercase).
# Records whose industry matches EXCLUDED_INDUSTRIES_PAT are hard-rejected.
# Records in TARGET_INDUSTRIES_PAT receive a scoring bonus.
EXCLUDED_INDUSTRIES_PAT: str = (
    r"staffing|recruiting|recruitment|executive search|"
    r"outsourc|\brpo\b|\bpeo\b|professional employer|"
    r"talent agency|placement agency|workforce solution"
)

TARGET_INDUSTRIES_PAT: str = (
    r"\bhealthcare\b|\bhealth care\b|\bhospital\b|\bmedical\b|dental|pharma|biotech|clinical|wellness|"
    r"\bconstruction\b|contractor|civil engineering|"
    r"\bsoftware\b|\btechnology\b|\btech\b|saas|it services|computer|internet|cloud|"
    r"\binsurance\b|"
    r"\bretail\b|consumer goods|e[\-\s]?commerce|wholesale|"
    r"manufactur|\bindustrial\b|fabricat|\bmachinery\b|\bautomotive\b"
)

