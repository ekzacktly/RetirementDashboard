import streamlit as st
import pandas as pd
import scipy.optimize as opt
import numpy_financial as npf
import plotly.express as px
import numpy as np
import json
import os
import sys
import uuid
from typing import List, Dict, Union, Any, Tuple

# Hide sidebar by default
st.set_page_config(page_title="Retirement Dashboard", layout="wide", initial_sidebar_state="collapsed")

# ==========================================
# CONSTANTS & CONFIGURATION LOADER
# ==========================================
US_STATES = [
    "Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware",
    "Florida", "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky",
    "Louisiana", "Maine", "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi",
    "Missouri", "Montana", "Nebraska", "Nevada", "New Hampshire", "New Jersey", "New Mexico",
    "New York", "North Carolina", "North Dakota", "Ohio", "Oklahoma", "Oregon", "Pennsylvania",
    "Rhode Island", "South Carolina", "South Dakota", "Tennessee", "Texas", "Utah", "Vermont",
    "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"
]

ZERO_TAX_STATES = ["Alaska", "Florida", "Nevada", "New Hampshire", "South Dakota", "Tennessee", "Texas", "Washington",
                   "Wyoming"]
RET_EXEMPT_STATES = ["Pennsylvania", "Illinois", "Mississippi", "Iowa"]

GLIDE_OPTIONS = [
    "Safe (Standard TDF)",
    "Moderate (Taxable Bridge)",
    "Delayed (Roth Heritage)",
    "No Glide (Abrupt Drop)"
]

config = {
    "num_people": "2 People",
    "state": "Pennsylvania",
    "state_tax_rate": 3.07,
    "state_exempts_ret": True,
    "local_tax_rate": 1.0,
    "current_age": 35, "retire_age": 65, "target_lifespan": 95,
    "asset_strategy": "Unified Portfolio",
    "pre_ret_return": 7.0, "post_ret_return": 4.0, "unified_glide_profile": "Safe (Standard TDF)",
    "pre_ret_trad": 6.0, "post_ret_trad": 3.5, "trad_glide_profile": "Safe (Standard TDF)",
    "pre_ret_roth": 8.5, "post_ret_roth": 6.0, "roth_glide_profile": "Delayed (Roth Heritage)",
    "pre_ret_brok": 7.5, "post_ret_brok": 5.0, "brok_glide_profile": "Moderate (Taxable Bridge)",
    "target_gross_income": 100000,
    "use_ss": True, "ss_payout_scenario": "100% (Scheduled Benefits)",
    "p1_fra_benefit": 2500, "p1_ss_age": 67,
    "p2_fra_benefit": 2000, "p2_ss_age": 67,
    "rmd_start_age": 75, "penalty_age": 60, "penalty_pct": 10.0,
    "use_smile_model": True, "use_cpi": False, "cpi_rate": 2.5, "use_irmaa": False,
    "p1_salary": 70000, "p1_annual_raise": 2.5,
    "p1_trad_401k_start": 45000, "p1_trad_401k_cont": 6.0, "p1_trad_401k_match": 4.0, "p1_trad_401k_flat": 0,
    "p1_trad_ira_start": 0, "p1_trad_ira_mo": 0,
    "p1_roth_401k_start": 0, "p1_roth_401k_cont": 0.0, "p1_roth_401k_match": 0.0,
    "p1_roth_ira_start": 5000, "p1_roth_ira_mo": 100,
    "p1_brok_start": 2000, "p1_brok_mo": 0,
    "p2_salary": 65000, "p2_annual_raise": 2.5,
    "p2_trad_401k_start": 35000, "p2_trad_401k_cont": 6.0, "p2_trad_401k_match": 4.0, "p2_trad_401k_flat": 0,
    "p2_trad_ira_start": 0, "p2_trad_ira_mo": 0,
    "p2_roth_401k_start": 0, "p2_roth_401k_cont": 0.0, "p2_roth_401k_match": 0.0,
    "p2_roth_ira_start": 5000, "p2_roth_ira_mo": 100,
    "p2_brok_start": 2000, "p2_brok_mo": 0
}


# ==========================================
# HELPER FUNCTIONS & IRS TAX ENGINE
# ==========================================
def estimate_pia(salary: float) -> float:
    aime = salary / 12.0
    if aime <= 1286.0:
        return aime * 0.90
    elif aime <= 7749.0:
        return (1286.0 * 0.90) + ((aime - 1286.0) * 0.32)
    else:
        return (1286.0 * 0.90) + ((7749.0 - 1286.0) * 0.32) + ((aime - 7749.0) * 0.15)


def calc_ss_multiplier(claim_age: int) -> float:
    fra = 67
    if claim_age == fra:
        return 1.0
    elif claim_age < fra:
        months_early = (fra - claim_age) * 12.0
        red_36 = min(36.0, months_early) * (5.0 / 9.0) / 100.0
        red_more = max(0.0, months_early - 36.0) * (5.0 / 12.0) / 100.0
        return float(1.0 - red_36 - red_more)
    else:
        months_delayed = (claim_age - fra) * 12.0
        return float(1.0 + (months_delayed * (8.0 / 12.0) / 100.0))


def calc_fed_tax(gross: float, inf_mult: float = 1.0, is_single_filer: bool = False) -> float:
    sd = (14600.0 if is_single_filer else 29200.0) * inf_mult
    taxable = max(0.0, gross - sd)
    tax = 0.0
    if is_single_filer:
        brackets = [(11600.0 * inf_mult, 0.10), (35550.0 * inf_mult, 0.12), (53325.0 * inf_mult, 0.22),
                    (91425.0 * inf_mult, 0.24), (49650.0 * inf_mult, 0.32), (362350.0 * inf_mult, 0.35),
                    (10000000.0 * inf_mult, 0.37)]
    else:
        brackets = [(23200.0 * inf_mult, 0.10), (71100.0 * inf_mult, 0.12), (106650.0 * inf_mult, 0.22),
                    (182850.0 * inf_mult, 0.24), (99300.0 * inf_mult, 0.32), (246950.0 * inf_mult, 0.35),
                    (10000000.0 * inf_mult, 0.37)]

    cur = taxable
    for b_size, rate in brackets:
        chunk = min(cur, b_size)
        tax += chunk * rate
        cur -= chunk
        if cur <= 0:
            break
    return float(tax)


def calc_fed_cg_tax(gains: float, ordinary_gross: float, inf_mult: float = 1.0, is_single_filer: bool = False) -> float:
    sd = (14600.0 if is_single_filer else 29200.0) * inf_mult
    ord_taxable = max(0.0, ordinary_gross - sd)
    rem_sd = max(0.0, sd - ordinary_gross)
    taxable_gains = max(0.0, gains - rem_sd)
    if taxable_gains <= 0:
        return 0.0

    b0_limit = (47025.0 if is_single_filer else 94050.0) * inf_mult
    b15_limit = (518900.0 if is_single_filer else 583750.0) * inf_mult
    tax = 0.0

    space_0 = max(0.0, b0_limit - ord_taxable)
    gain_0 = min(taxable_gains, space_0)
    rem_gains = taxable_gains - gain_0
    if rem_gains <= 0:
        return float(tax)

    space_15 = max(0.0, b15_limit - max(ord_taxable, b0_limit))
    gain_15 = min(rem_gains, space_15)
    tax += gain_15 * 0.15
    rem_gains -= gain_15
    if rem_gains <= 0:
        return float(tax)

    tax += rem_gains * 0.20
    return float(tax)


def calc_taxable_ss(ss_amount: float, magi: float, is_single_filer: bool) -> float:
    if ss_amount <= 0.0:
        return 0.0

    t1 = 25000.0 if is_single_filer else 32000.0
    t2 = 34000.0 if is_single_filer else 44000.0
    prov_inc = magi + 0.50 * ss_amount

    if prov_inc <= t1:
        return 0.0

    half_ss = 0.50 * ss_amount
    tier1_taxable = min(0.50 * (prov_inc - t1), half_ss)

    if prov_inc <= t2:
        return float(min(tier1_taxable, 0.85 * ss_amount))

    tier2_base = min(half_ss, 4500.0 if is_single_filer else 6000.0)
    taxable_raw = 0.85 * (prov_inc - t2) + tier2_base
    return float(min(taxable_raw, 0.85 * ss_amount))


def calc_irmaa_surcharge(magi: float, inf_mult: float = 1.0, is_single_filer: bool = False) -> float:
    t1 = (103000.0 if is_single_filer else 206000.0) * inf_mult
    t2 = (129000.0 if is_single_filer else 258000.0) * inf_mult
    t3 = (161000.0 if is_single_filer else 322000.0) * inf_mult
    t4 = (193000.0 if is_single_filer else 386000.0) * inf_mult
    t5 = (500000.0 if is_single_filer else 750000.0) * inf_mult

    mult = 1.0 if is_single_filer else 2.0

    if magi <= t1:
        return 0.0
    elif magi <= t2:
        return 838.80 * mult * inf_mult
    elif magi <= t3:
        return 2096.40 * mult * inf_mult
    elif magi <= t4:
        return 3354.00 * mult * inf_mult
    elif magi <= t5:
        return 4611.60 * mult * inf_mult
    else:
        return 5031.60 * mult * inf_mult


@st.cache_data
def get_gross_for_net_total(current_spend: float, ss_amount: float, state_tax_dec: float, state_exempts: bool,
                            is_single_filer: bool, inf_mult: float = 1.0, is_penalized: bool = False,
                            pen_pct: float = 0.10) -> float:
    def net_cash_total(g: float) -> float:
        taxable_ss = calc_taxable_ss(ss_amount, g, is_single_filer)
        fed_tax = calc_fed_tax(g + taxable_ss, inf_mult, is_single_filer)
        penalty = (g * pen_pct) if is_penalized else 0.0
        state_tax = 0.0 if state_exempts else (g * state_tax_dec)
        return (g + ss_amount) - fed_tax - penalty - state_tax

    if net_cash_total(0.0) >= current_spend:
        return 0.0

    try:
        res = opt.root_scalar(lambda g: net_cash_total(g) - current_spend, bracket=[0.0, current_spend * 3.0])
        return float(res.root)
    except ValueError:
        return float(current_spend * 1.5)


def get_smile_spending(age: int, base: float) -> float:
    if age <= 65:
        return base
    elif age <= 75:
        return base * (1.0 - 0.015 * (age - 65))
    elif age <= 85:
        return base * (0.85 - 0.005 * (age - 75))
    elif age <= 95:
        return base * (0.80 + 0.010 * (age - 85))
    else:
        return base * 0.90


def get_glide_return_custom(age: int, glide_retire_age: int, pre_ret: float, post_ret: float, profile: str) -> float:
    if profile == "No Glide (Abrupt Drop)":
        return pre_ret if age < glide_retire_age else post_ret

    ytr = glide_retire_age - age

    if profile == "Safe (Standard TDF)":
        start_ytr, end_ytr = 25, -7
    elif profile == "Moderate (Taxable Bridge)":
        start_ytr, end_ytr = 15, -10
    elif profile == "Delayed (Roth Heritage)":
        start_ytr, end_ytr = 0, -20
    else:
        return pre_ret if age < glide_retire_age else post_ret

    if ytr >= start_ytr:
        return pre_ret
    elif ytr <= end_ytr:
        return post_ret
    else:
        window = start_ytr - end_ytr
        progress = (start_ytr - ytr) / window
        return float(pre_ret - (pre_ret - post_ret) * progress)


def get_rmd_divisor(age: int) -> float:
    irs_table = {
        72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
        80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2, 87: 14.4,
        88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5, 95: 8.9,
        96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4, 101: 6.0, 102: 5.6, 103: 5.2,
        104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1, 108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4,
        112: 3.3, 113: 3.1, 114: 3.0, 115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3, 120: 2.0
    }
    if age < 72:
        return 999.0
    return float(irs_table.get(age, 2.0))


def withdraw_proportional(target: float, b1: float, b2: float) -> Tuple[float, float]:
    total = b1 + b2
    if total <= 0:
        return 0.0, 0.0
    if target >= total:
        return b1, b2
    d1_target = target * (b1 / total)
    return d1_target, target - d1_target


def withdraw_from_dual_brokerage_dynamic(net_needed: float, b1: float, basis1: float, b2: float, basis2: float,
                                         ordinary_gross: float, ss_amount: float, state_tax_dec: float, inf_mult: float,
                                         is_single_filer: bool, allow_neg: bool) -> Tuple[
    float, float, float, float, float, float, float, float, float]:
    total_bal = b1 + b2
    if total_bal <= 0:
        if allow_neg:
            return net_needed, 0.0, 0.0, 0.0, 0.0, b1 - (net_needed / 2.0), basis1, b2 - (net_needed / 2.0), basis2
        return 0.0, 0.0, 0.0, 0.0, 0.0, b1, basis1, b2, basis2

    ratio1 = b1 / total_bal
    ratio2 = 1.0 - ratio1

    gr1 = max(0.0, (b1 - basis1) / b1) if b1 > 0 else 0.0
    gr2 = max(0.0, (b2 - basis2) / b2) if b2 > 0 else 0.0

    def net_yield(gross_draw: float) -> float:
        d1_test = min(gross_draw, total_bal) * ratio1
        d2_test = min(gross_draw, total_bal) * ratio2
        gains = (d1_test * gr1) + (d2_test * gr2)

        new_taxable_ss = calc_taxable_ss(ss_amount, ordinary_gross + gains, is_single_filer)
        new_ord_tax = calc_fed_tax(ordinary_gross + new_taxable_ss, inf_mult, is_single_filer)
        base_ord_tax = calc_fed_tax(ordinary_gross + calc_taxable_ss(ss_amount, ordinary_gross, is_single_filer),
                                    inf_mult, is_single_filer)

        marginal_ord_tax = new_ord_tax - base_ord_tax
        fed_cg_tax = calc_fed_cg_tax(gains, ordinary_gross + new_taxable_ss, inf_mult, is_single_filer)
        state_cg_tax = gains * state_tax_dec

        return gross_draw - fed_cg_tax - state_cg_tax - marginal_ord_tax

    max_net = net_yield(total_bal)

    if net_needed <= max_net:
        if net_yield(net_needed) >= net_needed:
            target_gross = net_needed
        else:
            try:
                res = opt.root_scalar(lambda g: net_yield(g) - net_needed, bracket=[net_needed, total_bal])
                target_gross = float(res.root)
            except ValueError:
                target_gross = total_bal

        d1_draw = target_gross * ratio1
        d2_draw = target_gross * ratio2
        gains_realized = (d1_draw * gr1) + (d2_draw * gr2)

        new_taxable_ss = calc_taxable_ss(ss_amount, ordinary_gross + gains_realized, is_single_filer)
        new_ord_tax = calc_fed_tax(ordinary_gross + new_taxable_ss, inf_mult, is_single_filer)
        base_ord_tax = calc_fed_tax(ordinary_gross + calc_taxable_ss(ss_amount, ordinary_gross, is_single_filer),
                                    inf_mult, is_single_filer)
        marginal_ord = new_ord_tax - base_ord_tax

        fed_tax = calc_fed_cg_tax(gains_realized, ordinary_gross + new_taxable_ss, inf_mult, is_single_filer)
        state_tax = gains_realized * state_tax_dec
        net_provided = net_needed
    else:
        target_gross = total_bal
        d1_draw = b1
        d2_draw = b2
        gains_realized = (d1_draw * gr1) + (d2_draw * gr2)

        new_taxable_ss = calc_taxable_ss(ss_amount, ordinary_gross + gains_realized, is_single_filer)
        new_ord_tax = calc_fed_tax(ordinary_gross + new_taxable_ss, inf_mult, is_single_filer)
        base_ord_tax = calc_fed_tax(ordinary_gross + calc_taxable_ss(ss_amount, ordinary_gross, is_single_filer),
                                    inf_mult, is_single_filer)
        marginal_ord = new_ord_tax - base_ord_tax

        fed_tax = calc_fed_cg_tax(gains_realized, ordinary_gross + new_taxable_ss, inf_mult, is_single_filer)
        state_tax = gains_realized * state_tax_dec
        net_provided = max_net

        if allow_neg:
            shortfall = net_needed - max_net
            d1_draw += shortfall / 2.0
            d2_draw += shortfall / 2.0
            net_provided = net_needed

    nb1 = b1 - d1_draw
    nbasis1 = basis1 - (d1_draw * (basis1 / b1)) if b1 > 0 else basis1

    nb2 = b2 - d2_draw
    nbasis2 = basis2 - (d2_draw * (basis2 / b2)) if b2 > 0 else basis2

    return float(net_provided), float(fed_tax), float(state_tax), float(marginal_ord), float(gains_realized), float(
        nb1), float(nbasis1), float(nb2), float(nbasis2)


# ==========================================
# UI RENDERING & SIMULATION EXECUTION
# ==========================================
# (Skipped entirely when pytest loads the file)
if not getattr(sys, 'testing', False):

    if 'uploader_key' not in st.session_state:
        st.session_state['uploader_key'] = str(uuid.uuid4())
    if 'bypass_local' not in st.session_state:
        st.session_state['bypass_local'] = False

    # 1. Local Command-Line Profile Parsing
    base_dir = os.path.dirname(os.path.abspath(__file__))
    profile_filename = "my_profile.json"

    if "--profile" in sys.argv:
        try:
            idx = sys.argv.index("--profile")
            profile_filename = sys.argv[idx + 1]
        except IndexError:
            pass

    profile_path = profile_filename if os.path.isabs(profile_filename) else os.path.join(base_dir, profile_filename)
    profile_loaded_name = None

    if not st.session_state['bypass_local'] and os.path.exists(profile_path):
        with open(profile_path, 'r') as f:
            custom_config = json.load(f)
            if "target_net_income" in custom_config and "target_gross_income" not in custom_config:
                custom_config["target_gross_income"] = custom_config["target_net_income"]
            config.update(custom_config)
            profile_loaded_name = profile_filename

    # SIDEBAR: PROFILE MANAGEMENT & SETTINGS
    profile_container = st.sidebar.container()
    with profile_container:
        st.header("📂 Profile Management")
        uploaded_file = st.file_uploader(
            "Upload my_profile.json",
            type=["json"],
            key=st.session_state['uploader_key'],
            help="Select a previously exported my_profile.json file to configure all parameters."
        )

        if uploaded_file is not None and not isinstance(uploaded_file, list):
            try:
                file_content = uploaded_file.getvalue().decode("utf-8")
                ui_config = json.loads(file_content)
                if "target_net_income" in ui_config and "target_gross_income" not in ui_config:
                    ui_config["target_gross_income"] = ui_config["target_net_income"]
                config.update(ui_config)
                profile_loaded_name = f"Uploaded File: {uploaded_file.name}"
                st.success("Profile successfully applied!")
            except Exception as e:
                st.error(f"Failed to read JSON file: {e}")

        if st.button("🔄 Revert to Defaults",
                     help="Clear all custom inputs and loaded profiles to instantly revert back to the generic baseline couple."):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.session_state['uploader_key'] = str(uuid.uuid4())
            st.session_state['bypass_local'] = True
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("🌍 Household & Location")

    num_people_choice = st.sidebar.radio(
        "Household Setup",
        options=["1 Person", "2 People"],
        index=1 if config.get("num_people", "2 People") == "2 People" else 0,
        help="Select whether this plan models a single filer or a married couple filing jointly (MFJ)."
    )
    is_single = (num_people_choice == "1 Person")

    default_state = str(config.get("state", "Pennsylvania"))
    if default_state not in US_STATES:
        default_state = "Pennsylvania"
    state_choice = st.sidebar.selectbox("State of Residence", US_STATES, index=US_STATES.index(default_state),
                                        help="Determines specific state income tax rules applied to wages and retirement.")

    if state_choice in ZERO_TAX_STATES:
        state_tax_rate = 0.0
        state_exempts_ret = True
        local_tax_rate = 0.0
    else:
        if state_choice != config.get("state"):
            default_rate = 3.07 if state_choice == "Pennsylvania" else (4.95 if state_choice == "Illinois" else 5.0)
            default_exempt = state_choice in RET_EXEMPT_STATES
        else:
            default_rate = float(config.get("state_tax_rate", 5.0))
            default_exempt = bool(config.get("state_exempts_ret", False))

        state_tax_rate = st.sidebar.number_input("Effective State Income Tax (%)", value=default_rate, step=0.1,
                                                 help="Flat or progressive tax rate applied to state taxable income.")
        state_exempts_ret = st.sidebar.checkbox("State Exempts Retirement Income", value=default_exempt,
                                                help="If checked, distributions from Pre-Tax 401(k)s and IRAs are exempt from state income tax.")
        local_tax_rate = st.sidebar.number_input("Local Earned Income Tax (EIT) (%)",
                                                 value=float(config.get("local_tax_rate", 1.0)), step=0.1,
                                                 help="Local municipality tax applied exclusively to working wages.")

    state_tax_decimal = state_tax_rate / 100.0
    local_tax_decimal = local_tax_rate / 100.0

    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Engine Rules & Constraints")

    rmd_start_age = st.sidebar.number_input("RMD Start Age", value=int(config.get("rmd_start_age", 75)), step=1,
                                            help="Age when mandatory IRS distributions start from pre-tax accounts (Currently age 75 per SECURE 2.0).")
    penalty_age = st.sidebar.number_input("Early Withdrawal Penalty Age", value=int(config.get("penalty_age", 60)),
                                          step=1,
                                          help="Age when the 10% IRS early withdrawal penalty drops off (Typically 59.5, rounded to 60).")
    penalty_pct = st.sidebar.number_input("Early Withdrawal Penalty (%)", value=float(config.get("penalty_pct", 10.0)),
                                          step=1.0,
                                          help="Statutory excise tax rate assessed by the IRS on early distributions.") / 100.0

    use_smile_model = st.sidebar.checkbox("Use Retirement Spending Smile",
                                          value=bool(config.get("use_smile_model", True)),
                                          help="Modulates annual spending through Go-Go, Slow-Go, and Care phases to model real-world spending declines.")

    st.sidebar.markdown("### Advanced Institutional Toggles")
    use_cpi = st.sidebar.checkbox("Model Inflation (Nominal Dollars & CPI)", value=bool(config.get("use_cpi", False)),
                                  help="Converts Real Returns to Nominal Returns, exponentially inflates spending, and indexes IRS tax brackets to prevent artificial bracket creep.")
    cpi_rate = st.sidebar.number_input("CPI Inflation Rate (%)", value=float(config.get("cpi_rate", 2.5)), step=0.1,
                                       help="Expected annual inflation rate applied to the tax code and living expenses.") / 100.0 if use_cpi else 0.0

    use_irmaa = st.sidebar.checkbox("Calculate Medicare IRMAA Surcharges (Age 65+)",
                                    value=bool(config.get("use_irmaa", False)),
                                    help="Evaluates your MAGI against Medicare cliffs and deducts required surcharges for Part B and Part D.")

    # DASHBOARD UI SETUP
    st.title("Auto-Optimized Retirement Engine")

    if profile_loaded_name:
        st.success(f"✅ Successfully loaded configuration: `{profile_loaded_name}`")
    else:
        st.info(
            "ℹ️ Using standard generic defaults. You can upload a custom `my_profile.json` in the sidebar to load your exact numbers.")

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Retirement Dashboard", "📖 User Manual & Explainer", "📜 The Financial Story", "🎲 Monte Carlo Stress Test"])

    with tab1:
        st.header("1. Macro Assumptions & Goals")

        mac1, mac2, mac3, mac4 = st.columns(4)
        with mac1:
            current_age = st.number_input("Current Age", value=int(config["current_age"]), step=1,
                                          help="Current age of the primary earner for timeline tracking.")
        with mac2:
            retire_age = st.number_input("Retirement Age", value=int(config["retire_age"]), step=1,
                                         help="Age when you stop working and begin drawing from your portfolio.")
        with mac3:
            target_lifespan = st.number_input("Target Lifespan", value=int(config["target_lifespan"]), step=1,
                                              help="Terminal simulation age where balances are evaluated for depletion or preservation.")
        with mac4:
            target_gross_income = st.number_input("Target Gross Ret. Spend ($)",
                                                  value=int(config["target_gross_income"]), step=5000,
                                                  help="Target gross annual retirement spending (in today's dollars).")

        st.markdown("### Social Security Benefits")
        use_ss = st.checkbox("Enable Social Security Modeling", value=bool(config.get("use_ss", True)),
                             help="Toggle to completely exclude Social Security from your retirement cash flows.")

        if use_ss:
            ss_payout_scenario = st.selectbox(
                "Benefit Payout Projection",
                ["100% (Scheduled Benefits)", "78% (2032 SSA Depletion Cut)", "74% (2032 CBO Depletion Cut)"],
                index=["100% (Scheduled Benefits)", "78% (2032 SSA Depletion Cut)",
                       "74% (2032 CBO Depletion Cut)"].index(
                    str(config.get("ss_payout_scenario", "100% (Scheduled Benefits)"))),
                help="If the trust fund depletes in 2032 as projected, scheduled benefits will be statutorily reduced unless Congress intervenes. Select a haircut to stress-test your plan."
            )
            if "100%" in ss_payout_scenario:
                ss_multiplier_base = 1.0
            elif "78%" in ss_payout_scenario:
                ss_multiplier_base = 0.78
            else:
                ss_multiplier_base = 0.74

            p1_salary_base = float(config.get("p1_salary", 70000.0))
            p2_salary_base = float(config.get("p2_salary", 65000.0))

            ss_col1, ss_col2 = st.columns(2)
            with ss_col1:
                st.markdown("#### Person 1")
                p1_est_pia = estimate_pia(p1_salary_base)
                p1_fra_benefit = st.number_input(
                    "P1 FRA Base Benefit ($/mo)",
                    value=int(config.get("p1_fra_benefit", int(p1_est_pia))),
                    step=100,
                    help=f"Your Primary Insurance Amount (PIA) at exactly age 67. Based on your current salary of \\${p1_salary_base:,.0f}, an IRS Bend Point estimate is \\${int(p1_est_pia):,.0f}/mo. The engine uses this purely as a starting base—it will mathematically reduce or increase this amount based on your Claiming Age slider below."
                )
                p1_ss_age = st.slider(
                    "P1 Claiming Age",
                    62, 70,
                    int(config.get("p1_ss_age", 67)),
                    help="Claiming early at 62 permanently reduces your check by 30%. Delaying to 70 earns 8% annual credits, permanently boosting your check by 24%."
                )
                p1_final_mo = p1_fra_benefit * calc_ss_multiplier(p1_ss_age) * ss_multiplier_base

            with ss_col2:
                if is_single:
                    p2_fra_benefit = 0.0
                    p2_ss_age = 67
                    p2_final_mo = 0.0
                else:
                    st.markdown("#### Person 2")
                    p2_est_pia = estimate_pia(p2_salary_base)
                    p2_fra_benefit = st.number_input(
                        "P2 FRA Base Benefit ($/mo)",
                        value=int(config.get("p2_fra_benefit", int(p2_est_pia))),
                        step=100,
                        help=f"Your Primary Insurance Amount (PIA) at exactly age 67. Based on your current salary of \\${p2_salary_base:,.0f}, an IRS Bend Point estimate is \\${int(p2_est_pia):,.0f}/mo. The engine uses this purely as a starting base—it will mathematically reduce or increase this amount based on your Claiming Age slider below."
                    )
                    p2_ss_age = st.slider(
                        "P2 Claiming Age",
                        62, 70,
                        int(config.get("p2_ss_age", 67)),
                        help="Claiming early at 62 permanently reduces your check by 30%. Delaying to 70 earns 8% annual credits, permanently boosting your check by 24%."
                    )
                    p2_final_mo = p2_fra_benefit * calc_ss_multiplier(p2_ss_age) * ss_multiplier_base

            st.info(
                f"**Combined Initial Benefit:** \\${p1_final_mo + p2_final_mo:,.0f} / mo\n\n"
                f"*(P1 starts at Age {p1_ss_age} with \\${p1_final_mo:,.0f}/mo)*\n\n" +
                (f"*(P2 starts at Age {p2_ss_age} with \\${p2_final_mo:,.0f}/mo)*" if not is_single else "")
            )
        else:
            ss_payout_scenario = "100% (Scheduled Benefits)"
            ss_multiplier_base = 0.0
            p1_fra_benefit = 0.0
            p1_ss_age = 67
            p2_fra_benefit = 0.0
            p2_ss_age = 67

        st.markdown("### Investment Returns & Asset Location")
        asset_strategy = st.radio(
            "Return Assumption Strategy",
            ["Unified Portfolio", "Asset Location (By Tax Bucket)"],
            index=0 if str(config.get("asset_strategy", "Unified Portfolio")) == "Unified Portfolio" else 1,
            horizontal=True,
            help="Select whether your entire portfolio grows at the same average rate, or assign specific high/low-growth profiles to Pre-Tax, Roth, and Brokerage accounts."
        )

        if asset_strategy == "Unified Portfolio":
            u_col1, u_col2, u_col3 = st.columns(3)
            with u_col1:
                pre_ret_return = st.number_input("Pre-Retirement Return (Real %)",
                                                 value=float(config["pre_ret_return"]), step=0.1,
                                                 help="Expected real investment return before retirement.") / 100.0
            with u_col2:
                post_ret_return = st.number_input("Post-Retirement Return (Real %)",
                                                  value=float(config["post_ret_return"]), step=0.1,
                                                  help="Expected real investment return during retirement decumulation.") / 100.0
            with u_col3:
                unified_glide_profile = st.selectbox("Unified Glide Profile", GLIDE_OPTIONS, index=GLIDE_OPTIONS.index(
                    str(config.get("unified_glide_profile", "Safe (Standard TDF)"))),
                                                     help="Select the mathematical transition curve used to slowly shift your returns from Pre-Retirement to Post-Retirement.")

            pre_ret_trad = pre_ret_roth = pre_ret_brok = pre_ret_return
            post_ret_trad = post_ret_roth = post_ret_brok = post_ret_return
            trad_glide_profile = roth_glide_profile = brok_glide_profile = unified_glide_profile

        else:
            st.info(
                "💡 **Common Sense Setup:** To avoid forced RMD tax bombs, planners typically load Pre-Tax accounts with conservative bonds (Safe Glide), load Roth accounts with aggressive stocks for tax-free growth (Delayed/No Glide), and load Brokerage with tax-efficient broad market ETFs (Moderate Glide).")
            al_col1, al_col2, al_col3 = st.columns(3)
            with al_col1:
                st.markdown("**Pre-Tax (Trad 401k/IRA)**")
                pre_ret_trad = st.number_input("Pre-Ret Return (%) ", value=float(config.get("pre_ret_trad", 6.0)),
                                               step=0.1, help="Expected real return for Pre-Tax accounts.") / 100.0
                post_ret_trad = st.number_input("Post-Ret Return (%) ", value=float(config.get("post_ret_trad", 3.5)),
                                                step=0.1,
                                                help="Expected real return for Pre-Tax accounts in retirement.") / 100.0
                trad_glide_profile = st.selectbox("Trad Glide Profile", GLIDE_OPTIONS, index=GLIDE_OPTIONS.index(
                    str(config.get("trad_glide_profile", "Safe (Standard TDF)"))),
                                                  help="Starts shifting 25 years before retirement. Best for Pre-Tax accounts to mitigate Sequence of Returns Risk.")
            with al_col2:
                st.markdown("**Post-Tax (Roth)**")
                pre_ret_roth = st.number_input("Pre-Ret Return (%)  ", value=float(config.get("pre_ret_roth", 8.5)),
                                               step=0.1,
                                               help="Expected real return for tax-free Roth accounts.") / 100.0
                post_ret_roth = st.number_input("Post-Ret Return (%)  ", value=float(config.get("post_ret_roth", 6.0)),
                                                step=0.1,
                                                help="Expected real return for tax-free Roth accounts in retirement.") / 100.0
                roth_glide_profile = st.selectbox("Roth Glide Profile", GLIDE_OPTIONS, index=GLIDE_OPTIONS.index(
                    str(config.get("roth_glide_profile", "Delayed (Roth Heritage)"))),
                                                  help="Doesn't start shifting until retirement, gliding over 20 years. Best for Roth accounts left to compound tax-free.")
            with al_col3:
                st.markdown("**Taxable (Brokerage)**")
                pre_ret_brok = st.number_input("Pre-Ret Return (%)   ", value=float(config.get("pre_ret_brok", 7.5)),
                                               step=0.1, help="Expected real return for Taxable accounts.") / 100.0
                post_ret_brok = st.number_input("Post-Ret Return (%)   ", value=float(config.get("post_ret_brok", 5.0)),
                                                step=0.1,
                                                help="Expected real return for Taxable accounts in retirement.") / 100.0
                brok_glide_profile = st.selectbox("Brok Glide Profile", GLIDE_OPTIONS, index=GLIDE_OPTIONS.index(
                    str(config.get("brok_glide_profile", "Moderate (Taxable Bridge)"))),
                                                  help="Starts shifting 15 years before retirement. Best for Brokerage accounts providing flexible bridge income.")

            pre_ret_return = float(config.get("pre_ret_return", 7.0)) / 100.0
            post_ret_return = float(config.get("post_ret_return", 4.0)) / 100.0
            unified_glide_profile = "Safe (Standard TDF)"

        st.markdown("---")

        # ----------------------------------------
        # SECTION 2: Income & Salaries
        # ----------------------------------------
        st.header("2. Base Salaries")
        col2a, col2b = st.columns(2)
        with col2a:
            p1_salary = st.number_input("Person 1 Salary ($)", value=float(config["p1_salary"]), step=5000.0,
                                        help="Gross annual salary for Person 1.")
            p1_annual_raise = st.number_input("P1 Annual Raise (%)", value=float(config.get("p1_annual_raise", 2.5)),
                                              step=0.1,
                                              help="Expected annual percentage increase in wage income for Person 1.") / 100.0
        with col2b:
            p2_salary = st.number_input("Person 2 Salary ($)", value=float(config["p2_salary"]), step=5000.0,
                                        disabled=is_single, help="Gross annual salary for Person 2.")
            p2_annual_raise = st.number_input("P2 Annual Raise (%)", value=float(config.get("p2_annual_raise", 2.5)),
                                              step=0.1, disabled=is_single,
                                              help="Expected annual percentage increase in wage income for Person 2.") / 100.0

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------------------------------------
        # SECTION 3: Pre-Tax Accounts
        # ----------------------------------------
        st.header("3. Pre-Tax Accounts (Traditional 401k & IRA)")
        col3a, col3b = st.columns(2)
        with col3a:
            st.subheader("Person 1")
            p1_trad_401k_start = st.number_input("P1 Trad 401(k) Start Bal", value=float(config["p1_trad_401k_start"]),
                                                 step=10000.0,
                                                 help="Current balance in Person 1's Traditional Pre-Tax 401(k).")
            p1_trad_401k_cont = st.number_input("P1 Trad 401(k) Contrib (%)", value=float(config["p1_trad_401k_cont"]),
                                                step=1.0,
                                                help="Percentage of Person 1's salary contributed to their Pre-Tax 401(k).") / 100.0
            p1_trad_401k_match = st.number_input("P1 Trad 401(k) Match (%)", value=float(config["p1_trad_401k_match"]),
                                                 step=1.0,
                                                 help="Employer matching percentage deposited as Pre-Tax Traditional.") / 100.0
            p1_trad_401k_flat = st.number_input("P1 401(k) Flat Bonus/Yr ($)", value=float(config["p1_trad_401k_flat"]),
                                                step=1000.0,
                                                help="Fixed annual non-elective employer contribution or profit sharing.")
            p1_trad_ira_start = st.number_input("P1 Trad IRA Start Bal", value=float(config["p1_trad_ira_start"]),
                                                step=5000.0, help="Current balance in Person 1's Traditional IRA.")
            p1_trad_ira_mo = st.number_input("P1 Trad IRA Monthly ($)", value=float(config["p1_trad_ira_mo"]),
                                             step=100.0,
                                             help="Fixed monthly out-of-pocket contribution to Person 1's Traditional IRA.")
        with col3b:
            st.subheader("Person 2")
            p2_trad_401k_start = st.number_input("P2 Trad 401(k) Start Bal", value=float(config["p2_trad_401k_start"]),
                                                 step=10000.0, disabled=is_single,
                                                 help="Current balance in Person 2's Traditional Pre-Tax 401(k).")
            p2_trad_401k_cont = st.number_input("P2 Trad 401(k) Contrib (%)", value=float(config["p2_trad_401k_cont"]),
                                                step=1.0, disabled=is_single,
                                                help="Percentage of Person 2's salary contributed to their Pre-Tax 401(k).") / 100.0
            p2_trad_401k_match = st.number_input("P2 Trad 401(k) Match (%)", value=float(config["p2_trad_401k_match"]),
                                                 step=1.0, disabled=is_single,
                                                 help="Employer matching percentage deposited as Pre-Tax Traditional.") / 100.0
            p2_trad_401k_flat = st.number_input("P2 401(k) Flat Bonus/Yr ($)", value=float(config["p2_trad_401k_flat"]),
                                                step=1000.0, disabled=is_single,
                                                help="Fixed annual non-elective employer contribution or profit sharing.")
            p2_trad_ira_start = st.number_input("P2 Trad IRA Start Bal", value=float(config["p2_trad_ira_start"]),
                                                step=5000.0, disabled=is_single,
                                                help="Current balance in Person 2's Traditional IRA.")
            p2_trad_ira_mo = st.number_input("P2 Trad IRA Monthly ($)", value=float(config["p2_trad_ira_mo"]),
                                             step=100.0, disabled=is_single,
                                             help="Fixed monthly out-of-pocket contribution to Person 2's Traditional IRA.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------------------------------------
        # SECTION 4: Post-Tax Accounts
        # ----------------------------------------
        st.header("4. Post-Tax Accounts (Roth 401k & IRA)")
        col4a, col4b = st.columns(2)
        with col4a:
            st.subheader("Person 1")
            p1_roth_401k_start = st.number_input("P1 Roth 401(k) Start Bal", value=float(config["p1_roth_401k_start"]),
                                                 step=5000.0,
                                                 help="Current balance in Person 1's designated Roth 401(k).")
            p1_roth_401k_cont = st.number_input("P1 Roth 401(k) Contrib (%)", value=float(config["p1_roth_401k_cont"]),
                                                step=1.0,
                                                help="Percentage of Person 1's salary contributed to their Roth 401(k).") / 100.0
            p1_roth_401k_match = st.number_input("P1 Roth 401(k) Match (%)",
                                                 value=float(config.get("p1_roth_401k_match", 0.0)), step=1.0,
                                                 help="Employer matching percentage legally deposited as Roth (per SECURE 2.0).") / 100.0
            p1_roth_ira_start = st.number_input("P1 Roth IRA Start Bal", value=float(config["p1_roth_ira_start"]),
                                                step=5000.0, help="Current balance in Person 1's Roth IRA.")
            p1_roth_ira_mo = st.number_input("P1 Roth IRA Monthly ($)", value=float(config["p1_roth_ira_mo"]),
                                             step=100.0,
                                             help="Fixed monthly contribution to Person 1's Roth IRA (subject to annual IRS limits).")
        with col4b:
            st.subheader("Person 2")
            p2_roth_401k_start = st.number_input("P2 Roth 401(k) Start Bal", value=float(config["p2_roth_401k_start"]),
                                                 step=5000.0, disabled=is_single,
                                                 help="Current balance in Person 2's designated Roth 401(k).")
            p2_roth_401k_cont = st.number_input("P2 Roth 401(k) Contrib (%)", value=float(config["p2_roth_401k_cont"]),
                                                step=1.0, disabled=is_single,
                                                help="Percentage of Person 2's salary contributed to their Roth 401(k).") / 100.0
            p2_roth_401k_match = st.number_input("P2 Roth 401(k) Match (%)",
                                                 value=float(config.get("p2_roth_401k_match", 0.0)), step=1.0,
                                                 disabled=is_single,
                                                 help="Employer matching percentage legally deposited as Roth (per SECURE 2.0).") / 100.0
            p2_roth_ira_start = st.number_input("P2 Roth IRA Start Bal", value=float(config["p2_roth_ira_start"]),
                                                step=5000.0, disabled=is_single,
                                                help="Current balance in Person 2's Roth IRA.")
            p2_roth_ira_mo = st.number_input("P2 Roth IRA Monthly ($)", value=float(config["p2_roth_ira_mo"]),
                                             step=100.0, disabled=is_single,
                                             help="Fixed monthly contribution to Person 2's Roth IRA.")

        st.markdown("<br>", unsafe_allow_html=True)

        # ----------------------------------------
        # SECTION 5: Taxable Accounts
        # ----------------------------------------
        st.header("5. Taxable Accounts (Brokerage)")
        col5a, col5b = st.columns(2)
        with col5a:
            st.subheader("Person 1")
            p1_brok_start = st.number_input("P1 Brokerage Start Bal", value=float(config["p1_brok_start"]), step=5000.0,
                                            help="Current balance in Person 1's Non-Qualified Taxable Brokerage account.")
            p1_brok_mo = st.number_input("P1 Brokerage Monthly ($)", value=float(config["p1_brok_mo"]), step=100.0,
                                         help="Monthly post-tax contribution to Person 1's Taxable Brokerage.")
        with col5b:
            st.subheader("Person 2")
            p2_brok_start = st.number_input("P2 Brokerage Start Bal", value=float(config["p2_brok_start"]), step=5000.0,
                                            disabled=is_single,
                                            help="Current balance in Person 2's Non-Qualified Taxable Brokerage account.")
            p2_brok_mo = st.number_input("P2 Brokerage Monthly ($)", value=float(config["p2_brok_mo"]), step=100.0,
                                         disabled=is_single,
                                         help="Monthly post-tax contribution to Person 2's Taxable Brokerage.")

        # ==========================================
        # JSON PROFILE EXPORTER
        # ==========================================
        current_settings = {
            "num_people": num_people_choice,
            "state": state_choice,
            "state_tax_rate": round(state_tax_rate, 2),
            "state_exempts_ret": state_exempts_ret,
            "local_tax_rate": round(local_tax_rate, 2),
            "current_age": current_age,
            "retire_age": retire_age,
            "target_lifespan": target_lifespan,
            "asset_strategy": asset_strategy,
            "pre_ret_return": round(pre_ret_return * 100, 2),
            "post_ret_return": round(post_ret_return * 100, 2),
            "unified_glide_profile": unified_glide_profile,
            "pre_ret_trad": round(pre_ret_trad * 100, 2),
            "post_ret_trad": round(post_ret_trad * 100, 2),
            "trad_glide_profile": trad_glide_profile,
            "pre_ret_roth": round(pre_ret_roth * 100, 2),
            "post_ret_roth": round(post_ret_roth * 100, 2),
            "roth_glide_profile": roth_glide_profile,
            "pre_ret_brok": round(pre_ret_brok * 100, 2),
            "post_ret_brok": round(post_ret_brok * 100, 2),
            "brok_glide_profile": brok_glide_profile,
            "rmd_start_age": rmd_start_age,
            "p1_salary": p1_salary,
            "p1_annual_raise": round(p1_annual_raise * 100, 2),
            "use_ss": use_ss,
            "ss_payout_scenario": ss_payout_scenario if use_ss else "100% (Scheduled Benefits)",
            "p1_fra_benefit": p1_fra_benefit,
            "p1_ss_age": p1_ss_age,
            "p1_trad_401k_start": p1_trad_401k_start,
            "p1_trad_401k_cont": round(p1_trad_401k_cont * 100, 2),
            "p1_trad_401k_match": round(p1_trad_401k_match * 100, 2),
            "p1_trad_401k_flat": p1_trad_401k_flat,
            "p1_trad_ira_start": p1_trad_ira_start,
            "p1_trad_ira_mo": p1_trad_ira_mo,
            "p1_roth_401k_start": p1_roth_401k_start,
            "p1_roth_401k_cont": round(p1_roth_401k_cont * 100, 2),
            "p1_roth_401k_match": round(p1_roth_401k_match * 100, 2),
            "p1_roth_ira_start": p1_roth_ira_start,
            "p1_roth_ira_mo": p1_roth_ira_mo,
            "p1_brok_start": p1_brok_start,
            "p1_brok_mo": p1_brok_mo,
            "p2_salary": p2_salary if not is_single else 0.0,
            "p2_annual_raise": round(p2_annual_raise * 100, 2) if not is_single else 0.0,
            "p2_fra_benefit": p2_fra_benefit if not is_single else 0.0,
            "p2_ss_age": p2_ss_age if not is_single else 67,
            "p2_trad_401k_start": p2_trad_401k_start if not is_single else 0.0,
            "p2_trad_401k_cont": round(p2_trad_401k_cont * 100, 2) if not is_single else 0.0,
            "p2_trad_401k_match": round(p2_trad_401k_match * 100, 2) if not is_single else 0.0,
            "p2_trad_401k_flat": p2_trad_401k_flat if not is_single else 0.0,
            "p2_trad_ira_start": p2_trad_ira_start if not is_single else 0.0,
            "p2_trad_ira_mo": p2_trad_ira_mo if not is_single else 0.0,
            "p2_roth_401k_start": p2_roth_401k_start if not is_single else 0.0,
            "p2_roth_401k_cont": round(p2_roth_401k_cont * 100, 2) if not is_single else 0.0,
            "p2_roth_401k_match": round(p2_roth_401k_match * 100, 2) if not is_single else 0.0,
            "p2_roth_ira_start": p2_roth_ira_start if not is_single else 0.0,
            "p2_roth_ira_mo": p2_roth_ira_mo if not is_single else 0.0,
            "p2_brok_start": p2_brok_start if not is_single else 0.0,
            "p2_brok_mo": p2_brok_mo if not is_single else 0.0,
            "target_gross_income": target_gross_income,
            "penalty_age": penalty_age,
            "penalty_pct": round(penalty_pct * 100, 2),
            "use_smile_model": use_smile_model,
            "use_cpi": use_cpi,
            "cpi_rate": round(cpi_rate * 100, 2) if use_cpi else 2.5,
            "use_irmaa": use_irmaa
        }

        json_export = json.dumps(current_settings, indent=4)
        profile_container.download_button(
            label="💾 Save Current Profile",
            data=json_export,
            file_name="my_profile.json",
            mime="application/json",
            help="Click to save all current inputs, state selections, and settings into a standardized configuration file."
        )


        def run_simulation(test_gross_income: float, return_data: bool = False, allow_negative_brokerage: bool = False,
                           mc_mode: bool = False, mc_vols: dict = None) -> Union[float, List[float], Dict[str, Any]]:
            sim_data = []

            p1_trad_bal = p1_trad_401k_start + p1_trad_ira_start
            p2_trad_bal = p2_trad_401k_start + p2_trad_ira_start if not is_single else 0.0
            p1_roth_bal = p1_roth_401k_start + p1_roth_ira_start
            p2_roth_bal = p2_roth_401k_start + p2_roth_ira_start if not is_single else 0.0
            p1_brok_bal = p1_brok_start
            p1_brok_basis = p1_brok_start
            p2_brok_bal = p2_brok_start if not is_single else 0.0
            p2_brok_basis = p2_brok_start if not is_single else 0.0

            curr_p1_sal = p1_salary
            curr_p2_sal = p2_salary if not is_single else 0.0

            total_penalties = 0.0
            total_roth_conv_net = 0.0
            total_rmd_overflow_net = 0.0
            total_tax_living = 0.0
            total_tax_roth = 0.0
            total_tax_rmd = 0.0
            total_tax_brokerage = 0.0
            port_at_retire = 0.0
            total_state_tax_retired = 0.0
            total_irmaa_paid = 0.0
            total_ss_received = 0.0

            ret_balances_dict = {}
            mc_balance_history = []
            dep_age = "Never"
            ending_bal = 0.0

            max_sim_age = max(151, target_lifespan + 1)

            inf_mult_yr0 = 1.0
            base_net_spend = test_gross_income - calc_fed_tax(test_gross_income, inf_mult_yr0, is_single)
            if not state_exempts_ret:
                base_net_spend -= (test_gross_income * state_tax_decimal)

            for age in range(current_age, max_sim_age):
                is_retired = age >= retire_age
                is_penalized = retire_age <= age < penalty_age

                inf_mult = (1.0 + cpi_rate) ** (age - current_age) if use_cpi else 1.0

                trad_bal_tot = p1_trad_bal + p2_trad_bal
                roth_bal_tot = p1_roth_bal + p2_roth_bal
                brok_bal_tot = p1_brok_bal + p2_brok_bal

                if mc_mode:
                    mc_balance_history.append(trad_bal_tot + roth_bal_tot + brok_bal_tot)

                if return_data and dep_age == "Never" and (trad_bal_tot + roth_bal_tot + brok_bal_tot) < 1:
                    dep_age = str(age)

                if age == retire_age or (age == current_age and current_age >= retire_age):
                    port_at_retire = trad_bal_tot + roth_bal_tot + brok_bal_tot
                    ret_balances_dict = {
                        "P1 Pre-Tax": p1_trad_bal, "P2 Pre-Tax": p2_trad_bal,
                        "P1 Post-Tax (Roth)": p1_roth_bal, "P2 Post-Tax (Roth)": p2_roth_bal,
                        "P1 Brokerage": p1_brok_bal, "P2 Brokerage": p2_brok_bal
                    }

                b_trad_ret = get_glide_return_custom(age, retire_age, pre_ret_trad, post_ret_trad, trad_glide_profile)
                b_roth_ret = get_glide_return_custom(age, retire_age, pre_ret_roth, post_ret_roth, roth_glide_profile)
                b_brok_ret = get_glide_return_custom(age, retire_age, pre_ret_brok, post_ret_brok, brok_glide_profile)

                if mc_mode and mc_vols:
                    vol = mc_vols['pre'] if not is_retired else mc_vols['post']
                    ret_trad = float(np.random.normal(b_trad_ret, vol))
                    ret_roth = float(np.random.normal(b_roth_ret, vol))
                    ret_brok = float(np.random.normal(b_brok_ret, vol))
                else:
                    ret_trad, ret_roth, ret_brok = b_trad_ret, b_roth_ret, b_brok_ret

                if use_cpi:
                    ret_trad = ((1.0 + ret_trad) * (1.0 + cpi_rate)) - 1.0
                    ret_roth = ((1.0 + ret_roth) * (1.0 + cpi_rate)) - 1.0
                    ret_brok = ((1.0 + ret_brok) * (1.0 + cpi_rate)) - 1.0

                current_spend = get_smile_spending(age, base_net_spend) if use_smile_model and is_retired else (
                    base_net_spend if is_retired else 0.0)
                if use_cpi and is_retired:
                    current_spend *= inf_mult

                fed_tax_paid_yr = state_tax_paid_yr = cg_tax_paid_yr = penalty_paid_yr = rmd_amt_yr = 0.0
                pre_tax_in_yr = pre_tax_out_yr = roth_in_yr = roth_out_yr = brok_in_yr = brok_out_yr = 0.0
                irmaa_yr = ss_yr = 0.0

                if not is_retired:
                    p1_trad_in = (curr_p1_sal * p1_trad_401k_cont) + (
                                curr_p1_sal * p1_trad_401k_match) + p1_trad_401k_flat + (p1_trad_ira_mo * 12.0)
                    p2_trad_in = (curr_p2_sal * p2_trad_401k_cont) + (
                                curr_p2_sal * p2_trad_401k_match) + p2_trad_401k_flat + (
                                             p2_trad_ira_mo * 12.0) if not is_single else 0.0

                    p1_trad_bal = (p1_trad_bal + p1_trad_in) * (1.0 + ret_trad)
                    p2_trad_bal = (p2_trad_bal + p2_trad_in) * (1.0 + ret_trad) if not is_single else 0.0

                    p1_roth_in = (curr_p1_sal * p1_roth_401k_cont) + (curr_p1_sal * p1_roth_401k_match) + (
                                p1_roth_ira_mo * 12.0)
                    p2_roth_in = (curr_p2_sal * p2_roth_401k_cont) + (curr_p2_sal * p2_roth_401k_match) + (
                                p2_roth_ira_mo * 12.0) if not is_single else 0.0

                    p1_roth_bal = (p1_roth_bal + p1_roth_in) * (1.0 + ret_roth)
                    p2_roth_bal = (p2_roth_bal + p2_roth_in) * (1.0 + ret_roth) if not is_single else 0.0

                    p1_brok_in = p1_brok_mo * 12.0
                    p2_brok_in = p2_brok_mo * 12.0 if not is_single else 0.0
                    p1_brok_bal = (p1_brok_bal + p1_brok_in) * (1.0 + ret_brok)
                    p1_brok_basis += p1_brok_in
                    p2_brok_bal = (p2_brok_bal + p2_brok_in) * (1.0 + ret_brok) if not is_single else 0.0
                    p2_brok_basis += p2_brok_in

                    pre_tax_in_yr = p1_trad_in + p2_trad_in
                    roth_in_yr = p1_roth_in + p2_roth_in
                    brok_in_yr = p1_brok_in + p2_brok_in

                    curr_p1_sal *= (1.0 + p1_annual_raise)
                    curr_p2_sal *= (1.0 + p2_annual_raise) if not is_single else 0.0

                    status_label = "Working"

                elif is_penalized:
                    if use_ss and age >= p1_ss_age:
                        ss_yr += (p1_fra_benefit * 12.0) * calc_ss_multiplier(p1_ss_age) * ss_multiplier_base
                    if use_ss and not is_single and age >= p2_ss_age:
                        ss_yr += (p2_fra_benefit * 12.0) * calc_ss_multiplier(p2_ss_age) * ss_multiplier_base
                    if use_cpi:
                        ss_yr *= inf_mult
                    total_ss_received += ss_yr

                    taxable_ss_base = calc_taxable_ss(ss_yr, 0.0, is_single)
                    base_tax_fed = calc_fed_tax(taxable_ss_base, inf_mult, is_single)
                    net_ss = ss_yr - base_tax_fed
                    fed_tax_paid_yr += base_tax_fed

                    if net_ss >= current_spend:
                        surplus = net_ss - current_spend
                        brok_in_yr += surplus
                        brok_tot = p1_brok_bal + p2_brok_bal
                        p1_ratio = p1_brok_bal / brok_tot if brok_tot > 0 else 0.5
                        p1_brok_bal += surplus * p1_ratio
                        p1_brok_basis += surplus * p1_ratio
                        p2_brok_bal += surplus * (1.0 - p1_ratio)
                        p2_brok_basis += surplus * (1.0 - p1_ratio)
                        total_tax_living += base_tax_fed
                    else:
                        total_tax_living += base_tax_fed
                        shortfall = current_spend - net_ss
                        take_roth = min(roth_bal_tot, shortfall)
                        r1, r2 = withdraw_proportional(take_roth, p1_roth_bal, p2_roth_bal)
                        p1_roth_bal -= r1
                        p2_roth_bal -= r2
                        shortfall -= take_roth
                        roth_out_yr = take_roth

                        ordinary_gross = 0.0
                        gains_realized = 0.0

                        if shortfall > 0:
                            net_brok_drawn, brok_tax_fed, brok_tax_state, marginal_ord_tax, gains_realized, p1_brok_bal, p1_brok_basis, p2_brok_bal, p2_brok_basis = withdraw_from_dual_brokerage_dynamic(
                                shortfall, p1_brok_bal, p1_brok_basis, p2_brok_bal, p2_brok_basis, ordinary_gross,
                                ss_yr, state_tax_decimal, inf_mult, is_single, allow_negative_brokerage)
                            shortfall -= net_brok_drawn

                            brok_out_yr = net_brok_drawn + brok_tax_fed + brok_tax_state + marginal_ord_tax
                            cg_tax_paid_yr += brok_tax_fed
                            fed_tax_paid_yr += marginal_ord_tax
                            state_tax_paid_yr += brok_tax_state

                            total_tax_brokerage += brok_tax_fed
                            total_state_tax_retired += brok_tax_state

                        if shortfall > 0 and trad_bal_tot > 0:
                            def yield_trad(g: float) -> float:
                                new_taxable_ss = calc_taxable_ss(ss_yr, ordinary_gross + g + gains_realized, is_single)
                                new_ord_tax = calc_fed_tax(ordinary_gross + g + new_taxable_ss, inf_mult, is_single)
                                base_ord_tax_existing = calc_fed_tax(
                                    ordinary_gross + calc_taxable_ss(ss_yr, ordinary_gross + gains_realized, is_single),
                                    inf_mult, is_single)
                                marginal_ord = new_ord_tax - base_ord_tax_existing
                                penalty = g * penalty_pct
                                state_tax = 0.0 if state_exempts_ret else (g * state_tax_decimal)
                                return g - marginal_ord - penalty - state_tax

                            try:
                                res = opt.root_scalar(lambda g: yield_trad(g) - shortfall,
                                                      bracket=[shortfall, shortfall * 3.0])
                                take_trad = min(trad_bal_tot, float(res.root))
                            except ValueError:
                                take_trad = min(trad_bal_tot, shortfall * 1.5)

                            t1, t2 = withdraw_proportional(take_trad, p1_trad_bal, p2_trad_bal)
                            p1_trad_bal -= t1
                            p2_trad_bal -= t2

                            new_taxable_ss = calc_taxable_ss(ss_yr, ordinary_gross + take_trad + gains_realized,
                                                             is_single)
                            new_ord_tax = calc_fed_tax(ordinary_gross + take_trad + new_taxable_ss, inf_mult, is_single)
                            base_ord_tax_existing = calc_fed_tax(
                                ordinary_gross + calc_taxable_ss(ss_yr, ordinary_gross + gains_realized, is_single),
                                inf_mult, is_single)
                            marginal_ord_final = new_ord_tax - base_ord_tax_existing

                            trad_tax_state = 0.0 if state_exempts_ret else (take_trad * state_tax_decimal)

                            pre_tax_out_yr = take_trad
                            fed_tax_paid_yr += marginal_ord_final
                            state_tax_paid_yr += trad_tax_state
                            penalty_paid_yr = take_trad * penalty_pct

                            total_penalties += penalty_paid_yr
                            total_tax_living += marginal_ord_final
                            total_state_tax_retired += trad_tax_state

                    if use_irmaa and age >= 65:
                        magi = ordinary_gross + pre_tax_out_yr + gains_realized + calc_taxable_ss(ss_yr,
                                                                                                  ordinary_gross + pre_tax_out_yr + gains_realized,
                                                                                                  is_single)
                        irmaa_bill = calc_irmaa_surcharge(magi, inf_mult, is_single)
                        if irmaa_bill > 0:
                            irmaa_yr = irmaa_bill
                            total_irmaa_paid += irmaa_bill
                            pull_roth = min(p1_roth_bal + p2_roth_bal, irmaa_bill)
                            r1, r2 = withdraw_proportional(pull_roth, p1_roth_bal, p2_roth_bal)
                            p1_roth_bal -= r1
                            p2_roth_bal -= r2
                            roth_out_yr += pull_roth

                            rem_irmaa = irmaa_bill - pull_roth
                            if rem_irmaa > 0:
                                _, irmaa_cg_fed, irmaa_cg_state, marginal_ord_tax, gains_realized_irmaa, p1_brok_bal, p1_brok_basis, p2_brok_bal, p2_brok_basis = withdraw_from_dual_brokerage_dynamic(
                                    rem_irmaa, p1_brok_bal, p1_brok_basis, p2_brok_bal, p2_brok_basis,
                                    ordinary_gross + pre_tax_out_yr + gains_realized, ss_yr, state_tax_decimal,
                                    inf_mult, is_single, allow_negative_brokerage)
                                brok_out_yr += (rem_irmaa + irmaa_cg_fed + irmaa_cg_state + marginal_ord_tax)
                                cg_tax_paid_yr += irmaa_cg_fed
                                fed_tax_paid_yr += marginal_ord_tax
                                state_tax_paid_yr += irmaa_cg_state
                                total_tax_brokerage += irmaa_cg_fed
                                total_state_tax_retired += irmaa_cg_state

                    p1_trad_bal *= (1.0 + ret_trad)
                    p2_trad_bal *= (1.0 + ret_trad)
                    p1_roth_bal *= (1.0 + ret_roth)
                    p2_roth_bal *= (1.0 + ret_roth)
                    p1_brok_bal *= (1.0 + ret_brok)
                    p2_brok_bal *= (1.0 + ret_brok)
                    status_label = "Phase 1 (Penalty)"
                else:
                    if use_ss and age >= p1_ss_age:
                        ss_yr += (p1_fra_benefit * 12.0) * calc_ss_multiplier(p1_ss_age) * ss_multiplier_base
                    if use_ss and not is_single and age >= p2_ss_age:
                        ss_yr += (p2_fra_benefit * 12.0) * calc_ss_multiplier(p2_ss_age) * ss_multiplier_base
                    if use_cpi:
                        ss_yr *= inf_mult
                    total_ss_received += ss_yr

                    years_remaining = max(1, target_lifespan - age + 1)
                    if age > target_lifespan:
                        years_remaining = max(1, 120 - age + 1)

                    pmt = (-npf.pmt(b_trad_ret, years_remaining,
                                    trad_bal_tot / inf_mult) * inf_mult) if trad_bal_tot > 0 else 0.0

                    rmd = 0.0
                    if age >= rmd_start_age and trad_bal_tot > 0:
                        rmd = trad_bal_tot / get_rmd_divisor(age)
                        rmd_amt_yr = rmd

                    gross_need = get_gross_for_net_total(current_spend, ss_yr, state_tax_decimal, state_exempts_ret,
                                                         is_single, inf_mult, False, 0.0)
                    gross_trad = min(trad_bal_tot, max(rmd, float(pmt), gross_need))
                    pre_tax_out_yr = gross_trad
                    ordinary_gross = gross_trad

                    taxable_ss = calc_taxable_ss(ss_yr, gross_trad, is_single)
                    trad_tax_fed = calc_fed_tax(gross_trad + taxable_ss, inf_mult, is_single)
                    trad_tax_state = 0.0 if state_exempts_ret else (gross_trad * state_tax_decimal)

                    net_trad_and_ss = gross_trad + ss_yr - trad_tax_fed - trad_tax_state
                    fed_tax_paid_yr += trad_tax_fed
                    state_tax_paid_yr += trad_tax_state

                    t1, t2 = withdraw_proportional(gross_trad, p1_trad_bal, p2_trad_bal)
                    p1_trad_bal -= t1
                    p2_trad_bal -= t2

                    gains_realized = 0.0

                    if net_trad_and_ss >= current_spend:
                        surplus = net_trad_and_ss - current_spend
                        ratio_living = current_spend / net_trad_and_ss if net_trad_and_ss > 0 else 0.0
                        ratio_surplus = surplus / net_trad_and_ss if net_trad_and_ss > 0 else 0.0

                        total_tax_living += trad_tax_fed * ratio_living
                        total_state_tax_retired += trad_tax_state

                        ss_tax_isolated = calc_fed_tax(calc_taxable_ss(ss_yr, 0.0, is_single), inf_mult, is_single)
                        net_ss_only = ss_yr - ss_tax_isolated
                        net_trad_yield = net_trad_and_ss - net_ss_only

                        if age < rmd_start_age:
                            roth_in_yr = max(0.0, min(surplus, net_trad_yield))
                            brok_in_yr = surplus - roth_in_yr

                            p1_roth_bal += roth_in_yr / 2.0
                            p2_roth_bal += roth_in_yr / 2.0
                            total_roth_conv_net += roth_in_yr

                            total_tax_roth += trad_tax_fed * (
                                        roth_in_yr / net_trad_and_ss) if net_trad_and_ss > 0 else 0.0

                            if brok_in_yr > 0:
                                brok_tot = p1_brok_bal + p2_brok_bal
                                p1_ratio = p1_brok_bal / brok_tot if brok_tot > 0 else 0.5
                                p1_brok_bal += brok_in_yr * p1_ratio
                                p1_brok_basis += brok_in_yr * p1_ratio
                                p2_brok_bal += brok_in_yr * (1.0 - p1_ratio)
                                p2_brok_basis += brok_in_yr * (1.0 - p1_ratio)
                        else:
                            brok_tot = p1_brok_bal + p2_brok_bal
                            p1_ratio = p1_brok_bal / brok_tot if brok_tot > 0 else 0.5
                            p1_brok_bal += surplus * p1_ratio
                            p1_brok_basis += surplus * p1_ratio
                            p2_brok_bal += surplus * (1.0 - p1_ratio)
                            p2_brok_basis += surplus * (1.0 - p1_ratio)

                            brok_in_yr = surplus
                            total_rmd_overflow_net += surplus
                            total_tax_rmd += trad_tax_fed * ratio_surplus
                    else:
                        total_tax_living += trad_tax_fed
                        total_state_tax_retired += trad_tax_state

                        shortfall = current_spend - net_trad_and_ss
                        take_roth = min(roth_bal_tot, shortfall)
                        r1, r2 = withdraw_proportional(take_roth, p1_roth_bal, p2_roth_bal)
                        p1_roth_bal -= r1
                        p2_roth_bal -= r2
                        shortfall -= take_roth

                        roth_out_yr = take_roth

                        if shortfall > 0:
                            net_brok_drawn, brok_tax_fed, brok_tax_state, marginal_ord_tax, gains_realized, p1_brok_bal, p1_brok_basis, p2_brok_bal, p2_brok_basis = withdraw_from_dual_brokerage_dynamic(
                                shortfall, p1_brok_bal, p1_brok_basis, p2_brok_bal, p2_brok_basis, ordinary_gross,
                                ss_yr, state_tax_decimal, inf_mult, is_single, allow_negative_brokerage)
                            shortfall -= net_brok_drawn

                            brok_out_yr = net_brok_drawn + brok_tax_fed + brok_tax_state + marginal_ord_tax
                            cg_tax_paid_yr += brok_tax_fed
                            fed_tax_paid_yr += marginal_ord_tax
                            state_tax_paid_yr += brok_tax_state

                            total_tax_brokerage += brok_tax_fed
                            total_state_tax_retired += brok_tax_state

                    if use_irmaa and age >= 65:
                        magi = ordinary_gross + gains_realized + calc_taxable_ss(ss_yr, ordinary_gross + gains_realized,
                                                                                 is_single)
                        irmaa_bill = calc_irmaa_surcharge(magi, inf_mult, is_single)
                        if irmaa_bill > 0:
                            irmaa_yr = irmaa_bill
                            total_irmaa_paid += irmaa_bill
                            pull_roth = min(p1_roth_bal + p2_roth_bal, irmaa_bill)
                            r1, r2 = withdraw_proportional(pull_roth, p1_roth_bal, p2_roth_bal)
                            p1_roth_bal -= r1
                            p2_roth_bal -= r2
                            roth_out_yr += pull_roth

                            rem_irmaa = irmaa_bill - pull_roth
                            if rem_irmaa > 0:
                                _, irmaa_cg_fed, irmaa_cg_state, marginal_ord_tax, gains_realized_irmaa, p1_brok_bal, p1_brok_basis, p2_brok_bal, p2_brok_basis = withdraw_from_dual_brokerage_dynamic(
                                    rem_irmaa, p1_brok_bal, p1_brok_basis, p2_brok_bal, p2_brok_basis,
                                    ordinary_gross + gains_realized, ss_yr, state_tax_decimal, inf_mult, is_single,
                                    allow_negative_brokerage)
                                brok_out_yr += (rem_irmaa + irmaa_cg_fed + irmaa_cg_state + marginal_ord_tax)
                                cg_tax_paid_yr += irmaa_cg_fed
                                fed_tax_paid_yr += marginal_ord_tax
                                state_tax_paid_yr += irmaa_cg_state
                                total_tax_brokerage += irmaa_cg_fed
                                total_state_tax_retired += irmaa_cg_state

                    p1_trad_bal *= (1.0 + ret_trad)
                    p2_trad_bal *= (1.0 + ret_trad)
                    p1_roth_bal *= (1.0 + ret_roth)
                    p2_roth_bal *= (1.0 + ret_roth)
                    p1_brok_bal *= (1.0 + ret_brok)
                    p2_brok_bal *= (1.0 + ret_brok)
                    status_label = "Phase 2 (Levelized)" if age < rmd_start_age else "Phase 3 (RMDs)"

                if return_data and not mc_mode and age <= target_lifespan:
                    ret_display = f"{b_trad_ret * 100:.1f}%" if asset_strategy == "Unified Portfolio" else f"T:{b_trad_ret * 100:.1f}% | R:{b_roth_ret * 100:.1f}% | B:{b_brok_ret * 100:.1f}%"

                    sim_data.append([
                        age, status_label, ret_display, round(current_spend),
                        round(p1_trad_bal), round(p2_trad_bal), round(p1_roth_bal), round(p2_roth_bal),
                        round(p1_brok_bal), round(p2_brok_bal),
                        round(pre_tax_in_yr), round(pre_tax_out_yr),
                        round(roth_in_yr), round(roth_out_yr),
                        round(brok_in_yr), round(brok_out_yr), round(ss_yr),
                        round(rmd_amt_yr), round(irmaa_yr), round(fed_tax_paid_yr), round(state_tax_paid_yr),
                        round(cg_tax_paid_yr), round(penalty_paid_yr)
                    ])

                if age == target_lifespan:
                    ending_bal = p1_trad_bal + p2_trad_bal + p1_roth_bal + p2_roth_bal + p1_brok_bal + p2_brok_bal
                    if mc_mode:
                        return mc_balance_history
                    if return_data:
                        break

            if dep_age is None:
                dep_age = "Never"

            if return_data:
                return {
                    "data": sim_data,
                    "ending_bal": ending_bal,
                    "portfolio_at_retire": port_at_retire,
                    "tot_penalties": total_penalties,
                    "tot_roth_conv_net": total_roth_conv_net,
                    "tot_rmd_overflow_net": total_rmd_overflow_net,
                    "tot_tax_living": total_tax_living,
                    "tot_tax_roth": total_tax_roth,
                    "tot_tax_rmd": total_tax_rmd,
                    "tot_tax_brokerage": total_tax_brokerage,
                    "tot_state_tax_retired": total_state_tax_retired,
                    "tot_irmaa_paid": total_irmaa_paid,
                    "tot_ss_received": total_ss_received,
                    "depletion_age": dep_age,
                    "retire_balances_dict": ret_balances_dict
                }
            return float(ending_bal)


        # --- RUN SOLVERS ---
        sim_results = run_simulation(target_gross_income, return_data=True)
        if isinstance(sim_results, dict):
            sim_data = sim_results["data"]
            portfolio_at_retire_val = sim_results["portfolio_at_retire"]
            tot_penalties_val = sim_results["tot_penalties"]
            tot_roth_conv_net_val = sim_results["tot_roth_conv_net"]
            tot_rmd_overflow_net_val = sim_results["tot_rmd_overflow_net"]
            tot_tax_living_val = sim_results["tot_tax_living"]
            tot_tax_roth_val = sim_results["tot_tax_roth"]
            tot_tax_rmd_val = sim_results["tot_tax_rmd"]
            tot_tax_brokerage_val = sim_results["tot_tax_brokerage"]
            tot_state_tax_retired_val = sim_results["tot_state_tax_retired"]
            tot_irmaa_paid_val = sim_results["tot_irmaa_paid"]
            tot_ss_received_val = sim_results["tot_ss_received"]
            depletion_age_val = sim_results["depletion_age"]
            retire_balances_dict_val = sim_results["retire_balances_dict"]
        else:
            sim_data = []
            portfolio_at_retire_val = tot_penalties_val = tot_roth_conv_net_val = tot_rmd_overflow_net_val = 0.0
            tot_tax_living_val = tot_tax_roth_val = tot_tax_rmd_val = tot_tax_brokerage_val = 0.0
            tot_state_tax_retired_val = tot_irmaa_paid_val = tot_ss_received_val = 0.0
            depletion_age_val = "Never"
            retire_balances_dict_val = {}

        if 'max_zero' not in st.session_state:
            st.session_state.max_zero = 0.0
        if 'max_stable' not in st.session_state:
            st.session_state.max_stable = 0.0

        # ==========================================
        # DISPLAY RESULTS
        # ==========================================
        df = pd.DataFrame(sim_data, columns=[
            "Age", "Status", "Effective Return(s)", "Net Spend Target",
            "P1 Pre-Tax", "P2 Pre-Tax", "P1 Post-Tax (Roth)", "P2 Post-Tax (Roth)",
            "P1 Brokerage", "P2 Brokerage",
            "Pre-Tax Additions", "Pre-Tax Withdrawals",
            "Roth Additions", "Roth Withdrawals",
            "Brokerage Additions", "Brokerage Withdrawals", "Social Security",
            "RMD Amount", "IRMAA Surcharge", "Fed Ordinary Tax", "State Tax", "Fed Cap Gains Tax", "10% Penalty"
        ])

        if is_single:
            if "P2 Pre-Tax" in df.columns:
                df.drop(columns=["P2 Pre-Tax", "P2 Post-Tax (Roth)", "P2 Brokerage"], inplace=True)
                df['Total_Bal'] = df['P1 Pre-Tax'] + df['P1 Post-Tax (Roth)'] + df['P1 Brokerage']
        else:
            if "P2 Pre-Tax" in df.columns:
                df['Total_Bal'] = df['P1 Pre-Tax'] + df['P2 Pre-Tax'] + df['P1 Post-Tax (Roth)'] + df[
                    'P2 Post-Tax (Roth)'] + \
                                  df['P1 Brokerage'] + df['P2 Brokerage']

        end_of_life_balance = df[df['Age'] == target_lifespan]['Total_Bal'].iloc[0] if not df.empty else 0.0

        st.markdown("---")
        st.header("6. Lifetime Summary & KPIs")

        # -----------------------------------------------------
        # SUBSECTION: MONTHLY SAVINGS BREAKDOWN
        # -----------------------------------------------------
        st.subheader("Current Monthly Savings Breakdown")
        p1_trad_401k_mo = (p1_salary * p1_trad_401k_cont) / 12.0
        p1_roth_401k_mo = (p1_salary * p1_roth_401k_cont) / 12.0
        p2_trad_401k_mo = (p2_salary * p2_trad_401k_cont) / 12.0
        p2_roth_401k_mo = (p2_salary * p2_roth_401k_cont) / 12.0

        p1_trad_match_mo = (p1_salary * p1_trad_401k_match) / 12.0
        p1_roth_match_mo = (p1_salary * p1_roth_401k_match) / 12.0
        p2_trad_match_mo = (p2_salary * p2_trad_401k_match) / 12.0
        p2_roth_match_mo = (p2_salary * p2_roth_401k_match) / 12.0

        p1_flat_mo = p1_trad_401k_flat / 12.0
        p2_flat_mo = p2_trad_401k_flat / 12.0

        total_employee = p1_trad_401k_mo + p1_roth_401k_mo + p1_trad_ira_mo + p1_roth_ira_mo + p1_brok_mo + \
                         p2_trad_401k_mo + p2_roth_401k_mo + p2_trad_ira_mo + p2_roth_ira_mo + p2_brok_mo
        total_employer = p1_trad_match_mo + p1_roth_match_mo + p1_flat_mo + p2_trad_match_mo + p2_roth_match_mo + p2_flat_mo
        total_saved = total_employee + total_employer

        if is_single:
            cont_col1, cont_col3 = st.columns(2)
            with cont_col1:
                st.markdown("**Person 1 Monthly Savings:**")
                st.markdown(f"- Trad 401(k): \\${p1_trad_401k_mo:,.0f} *(+ \\${p1_trad_match_mo:,.0f} Match)*")
                st.markdown(f"- Trad IRA: \\${p1_trad_ira_mo:,.0f}")
                st.markdown(f"- Roth 401(k): \\${p1_roth_401k_mo:,.0f} *(+ \\${p1_roth_match_mo:,.0f} Match)*")
                st.markdown(f"- Roth IRA: \\${p1_roth_ira_mo:,.0f}")
                st.markdown(f"- Brokerage: \\${p1_brok_mo:,.0f}")
            with cont_col3:
                st.markdown("**Total Household Monthly:**")
                st.markdown(f"- **Total Employee (Out of Pocket):** \\${total_employee:,.0f}")
                st.markdown(f"- **Total Employer (% Matches):** \\${p1_trad_match_mo + p1_roth_match_mo:,.0f}")
                st.markdown(f"- **Total Employer (Flat/Bonus):** \\${p1_flat_mo:,.0f}")
                st.metric("Total Monthly Saved", f"${total_saved:,.0f}")
        else:
            cont_col1, cont_col2, cont_col3 = st.columns(3)
            with cont_col1:
                st.markdown("**Person 1 Monthly Savings:**")
                st.markdown(f"- Trad 401(k): \\${p1_trad_401k_mo:,.0f} *(+ \\${p1_trad_match_mo:,.0f} Match)*")
                st.markdown(f"- Trad IRA: \\${p1_trad_ira_mo:,.0f}")
                st.markdown(f"- Roth 401(k): \\${p1_roth_401k_mo:,.0f} *(+ \\${p1_roth_match_mo:,.0f} Match)*")
                st.markdown(f"- Roth IRA: \\${p1_roth_ira_mo:,.0f}")
                st.markdown(f"- Brokerage: \\${p1_brok_mo:,.0f}")
            with cont_col2:
                st.markdown("**Person 2 Monthly Savings:**")
                st.markdown(f"- Trad 401(k): \\${p2_trad_401k_mo:,.0f} *(+ \\${p2_trad_match_mo:,.0f} Match)*")
                st.markdown(f"- Trad IRA: \\${p2_trad_ira_mo:,.0f}")
                st.markdown(f"- Roth 401(k): \\${p2_roth_401k_mo:,.0f} *(+ \\${p2_roth_match_mo:,.0f} Match)*")
                st.markdown(f"- Roth IRA: \\${p2_roth_ira_mo:,.0f}")
                st.markdown(f"- Brokerage: \\${p2_brok_mo:,.0f}")
            with cont_col3:
                st.markdown("**Total Household Monthly:**")
                st.markdown(f"- **Total Employee (Out of Pocket):** \\${total_employee:,.0f}")
                st.markdown(
                    f"- **Total Employer (% Matches):** \\${p1_trad_match_mo + p2_trad_match_mo + p1_roth_match_mo + p2_roth_match_mo:,.0f}")
                st.markdown(f"- **Total Employer (Flat/Bonus):** \\${p1_flat_mo + p2_flat_mo:,.0f}")
                st.metric("Total Monthly Saved", f"${total_saved:,.0f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # -----------------------------------------------------
        # SUBSECTION: PORTFOLIO MILESTONES & SOLVERS
        # -----------------------------------------------------
        st.subheader("Portfolio Milestones & Solvers")
        kpi1, kpi2, kpi3 = st.columns(3)
        with kpi1:
            st.metric("Total Social Security Received", f"${tot_ss_received_val:,.0f}",
                      help="Lifetime cash flow generated by Social Security benefits (accounting for COLA if CPI is enabled).")
        with kpi2:
            st.metric("Portfolio Balance at Retirement", f"${portfolio_at_retire_val:,.0f}",
                      help="Projected value of all accounts at the exact retirement age.")
        with kpi3:
            st.metric(f"Portfolio Balance at Age {target_lifespan}", f"${end_of_life_balance:,.0f}",
                      help=f"Projected net worth across all accounts at your target lifespan age ({target_lifespan}).")

        st.markdown("##### Heavy Mathematics Solvers")
        if st.button("Calculate Maximum Spend Limits"):
            with st.spinner("Running heavy optimization solvers..."):
                try:
                    res_zero = opt.root_scalar(lambda x: float(run_simulation(x, allow_negative_brokerage=True)),
                                               bracket=[0, 3000000], method='brentq')
                    st.session_state.max_zero = float(res_zero.root)
                except ValueError:
                    st.session_state.max_zero = 0.0

                try:
                    res_stable = opt.root_scalar(
                        lambda x: float(run_simulation(x, allow_negative_brokerage=True)) - portfolio_at_retire_val,
                        bracket=[0, 3000000], method='brentq')
                    st.session_state.max_stable = float(res_stable.root)
                except ValueError:
                    st.session_state.max_stable = 0.0

        kpi4, kpi5 = st.columns(2)
        with kpi4:
            st.metric(
                "Max Allowable Pre-Tax Ret. Spend (Die at Zero)",
                f"${st.session_state.max_zero:,.0f}",
                help="The maximum sustainable gross withdrawal that depletes the entire portfolio to exactly $0 at the target lifespan age."
            )
        with kpi5:
            st.metric(
                "Stable Portfolio Pre-Tax Spend (Preserve Principal)",
                f"${st.session_state.max_stable:,.0f}",
                help="The exact sustainable gross withdrawal where terminal balance at target lifespan equals starting balance at retirement."
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # -----------------------------------------------------
        # SUBSECTION: TAX ANALYSIS & ADVANCED STRATEGY
        # -----------------------------------------------------
        st.subheader("Tax Analysis & Advanced Strategy")
        tax_col1, tax_col2, tax_col3, tax_col4 = st.columns(4)
        with tax_col1:
            st.metric("Federal Tax (Retirement Living)", f"${tot_tax_living_val:,.0f}",
                      help="Cumulative federal income taxes paid on withdrawals used directly to fund your baseline living expenses.")
            st.metric("Federal Cap Gains (Retirement)", f"${tot_tax_brokerage_val:,.0f}",
                      help="Cumulative federal long-term capital gains taxes automatically evaluated across progressive IRS brackets.")
        with tax_col2:
            st.metric("State Tax (Retirement)", f"${tot_state_tax_retired_val:,.0f}",
                      help="Total state income taxes paid on pre-tax distributions and capital gains during the decumulation phase.")
            st.metric("Total Early Penalties Paid", f"${tot_penalties_val:,.0f}",
                      help="Total 10% IRS penalties incurred from forced early pre-tax withdrawals prior to penalty age.")
        with tax_col3:
            st.metric("Total Tax-Free Roth Conversions", f"${tot_roth_conv_net_val:,.0f}",
                      help="Cumulative surplus net withdrawals successfully rolled into Roth accounts during Phase 2 (ages 60 to 74).")
            st.metric("Medicare IRMAA Surcharges", f"${tot_irmaa_paid_val:,.0f}",
                      help="Total extra premiums paid for Medicare Part B and D due to high MAGI.")
        with tax_col4:
            st.metric("Total RMD Overflows to Brokerage", f"${tot_rmd_overflow_net_val:,.0f}",
                      help="Cumulative after-tax distributions forced by IRS RMDs (age 75+) that exceeded living expenses and swept into Taxable accounts.")
            st.metric("Taxes: RMD Overflows", f"${tot_tax_rmd_val:,.0f}",
                      help="Cumulative federal taxes paid on mandatory RMD withdrawals that exceeded your lifestyle spending requirements.")

        st.markdown("---")
        st.subheader("Account Balances Over Time")

        y_cols = ["P1 Pre-Tax", "P1 Post-Tax (Roth)", "P1 Brokerage"]
        if not is_single:
            y_cols.extend(["P2 Pre-Tax", "P2 Post-Tax (Roth)", "P2 Brokerage"])

        y_cols.sort(key=lambda x: retire_balances_dict_val.get(x, 0.0), reverse=True)

        chart_data = df[["Age"] + y_cols] if not df.empty else pd.DataFrame()
        if not chart_data.empty:
            fig = px.area(
                chart_data, x="Age", y=y_cols,
                labels={"value": "Account Balance ($)", "variable": "Account Bucket"}
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("7. Lifetime Projection & Automated Waterfall")
        st.dataframe(df.drop(columns=['Total_Bal']) if not df.empty else df, use_container_width=True)

    # ==========================================
    # MONTE CARLO TAB
    # ==========================================
    with tab4:
        st.title("🎲 Monte Carlo Stress Test")
        st.markdown("""
        Deterministic projections assume you earn exactly your target return every single year. The real world is far more chaotic. 
        This engine runs your current financial profile through a stochastic simulator, introducing randomized volatility to test the mathematical durability of your retirement plan against **Sequence of Returns Risk** (e.g., suffering a major bear market right when you retire).
        """)

        mc_c1, mc_c2, mc_c3 = st.columns(3)
        with mc_c1:
            n_simulations = st.slider("Number of Simulations", min_value=100, max_value=1000, value=250, step=50,
                                      help="Higher numbers create smoother probability bands but take longer to process.")
        with mc_c2:
            vol_pre = st.number_input("Pre-Retirement Volatility (Standard Deviation %)", value=15.0, step=1.0,
                                      help="Expected volatility during accumulation (15% is standard for an S&P 500 equivalent portfolio).") / 100.0
        with mc_c3:
            vol_post = st.number_input("Post-Retirement Volatility (Standard Deviation %)", value=8.0, step=1.0,
                                       help="Expected volatility during decumulation (8% is standard for a conservative 60/40 blended portfolio).") / 100.0

        if st.button("🚀 Run Monte Carlo Simulation", type="primary"):
            with st.spinner(f"Simulating {n_simulations} alternative timelines. Please wait..."):
                all_balances = []
                successes = 0

                for i in range(n_simulations):
                    mc_vols = {'pre': vol_pre, 'post': vol_post}
                    bal_history = run_simulation(target_gross_income, return_data=True, mc_mode=True, mc_vols=mc_vols)
                    if isinstance(bal_history, list):
                        all_balances.append(bal_history)
                        if len(bal_history) > 0 and bal_history[-1] > 0:
                            successes += 1

                if all_balances:
                    df_mc = pd.DataFrame(all_balances).T
                    df_mc.index = list(range(current_age, target_lifespan + 1))

                    p10 = df_mc.quantile(0.10, axis=1)
                    p50 = df_mc.quantile(0.50, axis=1)
                    p90 = df_mc.quantile(0.90, axis=1)

                    df_plot = pd.DataFrame({
                        "10th Percentile (Pessimistic)": p10,
                        "50th Percentile (Median)": p50,
                        "90th Percentile (Optimistic)": p90
                    })

                    success_rate = (successes / n_simulations) * 100.0

                    st.markdown("---")
                    kpi_mc1, kpi_mc2 = st.columns(2)
                    with kpi_mc1:
                        if success_rate >= 85.0:
                            st.success(f"**Probability of Success:** {success_rate:.1f}%")
                        elif success_rate >= 70.0:
                            st.warning(f"**Probability of Success:** {success_rate:.1f}%")
                        else:
                            st.error(f"**Probability of Success:** {success_rate:.1f}%")
                    with kpi_mc2:
                        st.info(f"**Median Terminal Balance:** \\${p50.iloc[-1]:,.0f}")

                    fig_mc = px.line(
                        df_plot,
                        title="Stochastic Portfolio Projections (Confidence Intervals)",
                        labels={"index": "Age", "value": "Total Portfolio Balance ($)", "variable": "Scenario Outcome"}
                    )
                    st.plotly_chart(fig_mc, use_container_width=True)