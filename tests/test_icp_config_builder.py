"""Tests for ICPConfig builder logic (mirrors streamlit_app._build_icp_config)."""
from src.config import INDUSTRY_GROUPS
from src.scorer import ICPConfig

SENIORITY_OPTIONS = [
    ("C-Suite",  "c suite"),
    ("Founder",  "founder"),
    ("Owner",    "owner"),
    ("VP",       "vp"),
    ("Director", "director"),
    ("Head",     "head"),
    ("Manager",  "manager"),
    ("Senior",   "senior"),
    ("Partner",  "partner"),
    ("Entry",    "entry"),
]


def build_icp_config(state: dict) -> ICPConfig:
    """Pure helper — same logic as streamlit_app._build_icp_config but takes explicit dict."""
    emp_min = int(state.get("emp_min", 1))
    emp_max = int(state.get("emp_max", 9_999_999))

    use_all_ind = state.get("all_industries", True)
    if use_all_ind:
        tgt_pat = None
        apply_bonus = False
    else:
        selected = [n for n in INDUSTRY_GROUPS if state.get(f"ind_{n}")]
        if selected:
            tgt_pat = "|".join(
                f"(?:{INDUSTRY_GROUPS[n]})" for n in selected
                if n != "Staffing & Recruiting"
            )
            apply_bonus = bool(tgt_pat)
        else:
            tgt_pat = None
            apply_bonus = False

    use_all_sen = state.get("all_seniority", True)
    if use_all_sen or not any(state.get(f"sen_{lbl}") for lbl, _ in SENIORITY_OPTIONS):
        seniority_include = None
    else:
        seniority_include = [
            api_val for lbl, api_val in SENIORITY_OPTIONS
            if state.get(f"sen_{lbl}")
        ]

    return ICPConfig(
        emp_min=emp_min,
        emp_max=emp_max,
        apply_industry_filter=state.get("excl_staffing", True),
        apply_industry_bonus=apply_bonus,
        target_industries_pat=tgt_pat,
        seniority_include=seniority_include,
        require_email=state.get("require_email", False),
        require_phone=state.get("require_phone", False),
        require_linkedin=state.get("require_linkedin", False),
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_defaults_no_seniority_filter():
    cfg = build_icp_config({"all_industries": True, "all_seniority": True})
    assert cfg.seniority_include is None


def test_emp_range_passed_through():
    cfg = build_icp_config({"emp_min": 200, "emp_max": 5000, "all_industries": True, "all_seniority": True})
    assert cfg.emp_min == 200
    assert cfg.emp_max == 5000


def test_selected_industries_build_pattern():
    state = {
        "all_industries": False,
        "all_seniority": True,
        "ind_Healthcare & Medical": True,
    }
    cfg = build_icp_config(state)
    assert cfg.target_industries_pat is not None
    assert "hospital" in cfg.target_industries_pat
    assert cfg.apply_industry_bonus is True


def test_all_industries_disables_bonus():
    cfg = build_icp_config({"all_industries": True, "all_seniority": True})
    assert cfg.target_industries_pat is None
    assert cfg.apply_industry_bonus is False


def test_seniority_specific_levels():
    state = {
        "all_industries": True,
        "all_seniority": False,
        "sen_C-Suite": True,
        "sen_VP": True,
        "sen_Founder": False,
        "sen_Owner": False,
        "sen_Director": False,
        "sen_Head": False,
        "sen_Manager": False,
        "sen_Senior": False,
        "sen_Partner": False,
        "sen_Entry": False,
    }
    cfg = build_icp_config(state)
    assert cfg.seniority_include == ["c suite", "vp"]


def test_contact_quality_gates():
    state = {
        "all_industries": True,
        "all_seniority": True,
        "require_email": True,
        "require_phone": False,
        "require_linkedin": True,
    }
    cfg = build_icp_config(state)
    assert cfg.require_email is True
    assert cfg.require_phone is False
    assert cfg.require_linkedin is True


def test_excl_staffing_flag():
    cfg_on  = build_icp_config({"all_industries": True, "all_seniority": True, "excl_staffing": True})
    cfg_off = build_icp_config({"all_industries": True, "all_seniority": True, "excl_staffing": False})
    assert cfg_on.apply_industry_filter is True
    assert cfg_off.apply_industry_filter is False


def test_no_industries_selected_no_bonus():
    state = {"all_industries": False, "all_seniority": True}
    # No ind_* keys set → no selected industries
    cfg = build_icp_config(state)
    assert cfg.target_industries_pat is None
    assert cfg.apply_industry_bonus is False
