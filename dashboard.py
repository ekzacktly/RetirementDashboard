import streamlit as st
import pandas as pd
import scipy.optimize as opt
import numpy_financial as npf
import plotly.express as px
import json
import os
import sys

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

config = {
    "num_people": "2 People",
    "state": "Pennsylvania",
    "state_tax_rate": 3.07,
    "state_exempts_ret": True,
    "local_tax_rate": 1.0,
    "current_age": 36, "retire_age": 55, "target_lifespan": 95,
    "pre_ret_return": 7.0, "post_ret_return": 4.0,
    "target_gross_income": 150000,
    "rmd_start_age": 75, "penalty_age": 60, "penalty_pct": 10.0,
    "p1_salary": 110000, "p1_annual_raise": 2.0,
    "p1_trad_401k_start": 325000, "p1_trad_401k_cont": 10.0, "p1_trad_401k_match": 6.0, "p1_trad_401k_flat": 5000,
    "p1_trad_ira_start": 0, "p1_trad_ira_mo": 0,
    "p1_roth_401k_start": 0, "p1_roth_401k_cont": 0.0, "p1_roth_401k_match": 0.0,
    "p1_roth_ira_start": 0, "p1_roth_ira_mo": 625,
    "p1_brok_start": 0, "p1_brok_mo": 488,
    "p2_salary": 100000, "p2_annual_raise": 2.0,
    "p2_trad_401k_start": 70000, "p2_trad_401k_cont": 10.0, "p2_trad_401k_match": 4.0, "p2_trad_401k_flat": 5000,
    "p2_trad_ira_start": 0, "p2_trad_ira_mo": 0,
    "p2_roth_401k_start": 0, "p2_roth_401k_cont": 0.0, "p2_roth_401k_match": 0.0,
    "p2_roth_ira_start": 0, "p2_roth_ira_mo": 625,
    "p2_brok_start": 0, "p2_brok_mo": 488,
    "use_glide_path": True, "use_smile_model": True
}

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

if os.path.exists(profile_path):
    with open(profile_path, 'r') as f:
        custom_config = json.load(f)
        if "target_net_income" in custom_config and "target_gross_income" not in custom_config:
            custom_config["target_gross_income"] = custom_config["target_net_income"]
        config.update(custom_config)
        profile_loaded_name = profile_filename

# ==========================================
# SIDEBAR: HOUSEHOLD, RULES & FILE I/O
# ==========================================
st.sidebar.header("📂 Load Custom Profile")
uploaded_file = st.sidebar.file_uploader(
    "Upload my_profile.json",
    type=["json"],
    help="Select a previously exported my_profile.json file to configure all parameters."
)

if uploaded_file is not None:
    try:
        ui_config = json.load(uploaded_file)
        if "target_net_income" in ui_config and "target_gross_income" not in ui_config:
            ui_config["target_gross_income"] = ui_config["target_net_income"]
        config.update(ui_config)
        profile_loaded_name = f"Uploaded File: {uploaded_file.name}"
        st.sidebar.success("Profile successfully applied!")
    except Exception as e:
        st.sidebar.error(f"Failed to read JSON file: {e}")

st.sidebar.markdown("---")
st.sidebar.header("🌍 Household & Location")

num_people_choice = st.sidebar.radio(
    "Household Setup",
    options=["1 Person", "2 People"],
    index=1 if config.get("num_people", "2 People") == "2 People" else 0,
    help="Select whether this plan models a single filer or a married couple filing jointly (MFJ)."
)
is_single = (num_people_choice == "1 Person")

default_state = config.get("state", "Pennsylvania")
if default_state not in US_STATES: default_state = "Pennsylvania"

state_choice = st.sidebar.selectbox(
    "State of Residence",
    US_STATES,
    index=US_STATES.index(default_state),
    help="Determines the specific state income tax rules applied to your wages and retirement distributions."
)

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
                                             help="Flat or effective progressive tax rate applied to state taxable income.")
    state_exempts_ret = st.sidebar.checkbox("State Exempts Retirement Income", value=default_exempt,
                                            help="If checked, distributions from Pre-Tax 401(k)s and IRAs are completely exempt from state income tax.")
    local_tax_rate = st.sidebar.number_input("Local Earned Income Tax (EIT) (%)",
                                             value=float(config.get("local_tax_rate", 1.0)), step=0.1,
                                             help="Local municipality tax applied exclusively to working wages (W-2 income).")

state_tax_decimal = state_tax_rate / 100
local_tax_decimal = local_tax_rate / 100

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Engine Rules & Constraints")

rmd_start_age = st.sidebar.number_input(
    "RMD Start Age",
    value=int(config.get("rmd_start_age", 75)),
    step=1,
    help="Age when mandatory IRS distributions start from pre-tax accounts (Currently age 75 per SECURE 2.0)."
)
penalty_age = st.sidebar.number_input(
    "Early Withdrawal Penalty Age",
    value=int(config.get("penalty_age", 60)),
    step=1,
    help="Age when the 10% IRS early withdrawal penalty drops off (Typically 59.5, rounded to 60)."
)
penalty_pct = st.sidebar.number_input(
    "Early Withdrawal Penalty (%)",
    value=float(config.get("penalty_pct", 10.0)),
    step=1.0,
    help="Statutory excise tax rate assessed by the IRS on early distributions."
) / 100

use_glide_path = st.sidebar.checkbox(
    "Use Glide Path for Returns",
    value=bool(config.get("use_glide_path", True)),
    help="Smoothly transitions returns from the pre-retirement rate down to the post-retirement rate over a 32-year curve."
)
use_smile_model = st.sidebar.checkbox(
    "Use Retirement Spending Smile",
    value=bool(config.get("use_smile_model", True)),
    help="Modulates annual spending through Go-Go, Slow-Go, and Care phases to model real-world spending declines."
)


# ==========================================
# HELPER FUNCTIONS
# ==========================================
def calc_mfj_tax(gross, is_penalized=False, penalty_pct=0.10):
    taxable = max(0, gross - 32200)
    tax = 0
    brackets = [(23200, 0.10), (71100, 0.12), (104650, 0.22), (183850, 0.24), (85150, 0.32), (1000000, 0.35)]
    cur = taxable
    for b_size, rate in brackets:
        chunk = min(cur, b_size)
        tax += chunk * rate
        cur -= chunk
        if cur <= 0: break
    if is_penalized:
        tax += (gross * penalty_pct)
    return tax


def calc_mfj_cg_tax(gains, ordinary_gross):
    sd = 32200
    ord_taxable = max(0, ordinary_gross - sd)
    rem_sd = max(0, sd - ordinary_gross)
    taxable_gains = max(0, gains - rem_sd)
    if taxable_gains <= 0: return 0.0

    b0_limit = 94050
    b15_limit = 583750
    tax = 0.0

    space_0 = max(0, b0_limit - ord_taxable)
    gain_0 = min(taxable_gains, space_0)
    rem_gains = taxable_gains - gain_0
    if rem_gains <= 0: return tax

    space_15 = max(0, b15_limit - max(ord_taxable, b0_limit))
    gain_15 = min(rem_gains, space_15)
    tax += gain_15 * 0.15
    rem_gains -= gain_15
    if rem_gains <= 0: return tax

    tax += rem_gains * 0.20
    return tax


def get_gross_for_net(target_net, state_tax_dec, state_exempts, is_penalized=False, penalty_pct=0.10):
    if target_net <= 0: return 0

    def root_func(g):
        fed_tax = calc_mfj_tax(g, is_penalized, penalty_pct)
        state_tax = 0 if state_exempts else (g * state_tax_dec)
        return (g - fed_tax - state_tax) - target_net

    res = opt.root_scalar(root_func, bracket=[target_net, target_net * 2.5])
    return res.root


def get_smile_spending(age, base):
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


def get_glide_return(age, retire_age, pre_return, post_return):
    ytr = retire_age - age
    if ytr >= 25:
        return pre_return
    elif ytr <= -7:
        return post_return
    else:
        return post_return + (pre_return - post_return) * ((ytr - (-7)) / 32.0)


def get_rmd_divisor(age):
    irs_table = {
        72: 27.4, 73: 26.5, 74: 25.5, 75: 24.6, 76: 23.7, 77: 22.9, 78: 22.0, 79: 21.1,
        80: 20.2, 81: 19.4, 82: 18.5, 83: 17.7, 84: 16.8, 85: 16.0, 86: 15.2, 87: 14.4,
        88: 13.7, 89: 12.9, 90: 12.2, 91: 11.5, 92: 10.8, 93: 10.1, 94: 9.5, 95: 8.9,
        96: 8.4, 97: 7.8, 98: 7.3, 99: 6.8, 100: 6.4, 101: 6.0, 102: 5.6, 103: 5.2,
        104: 4.9, 105: 4.6, 106: 4.3, 107: 4.1, 108: 3.9, 109: 3.7, 110: 3.5, 111: 3.4,
        112: 3.3, 113: 3.1, 114: 3.0, 115: 2.9, 116: 2.8, 117: 2.7, 118: 2.5, 119: 2.3, 120: 2.0
    }
    if age < 72: return 999.0
    return irs_table.get(age, 2.0)


def withdraw_proportional(target, b1, b2):
    total = b1 + b2
    if total <= 0: return 0, 0
    if target >= total: return b1, b2
    d1 = target * (b1 / total)
    return d1, target - d1


def withdraw_from_dual_brokerage_dynamic(net_needed, b1, basis1, b2, basis2, ordinary_gross, state_tax_dec, allow_neg):
    total_bal = b1 + b2
    if total_bal <= 0:
        if allow_neg: return net_needed, 0, 0, b1 - (net_needed / 2), basis1, b2 - (net_needed / 2), basis2
        return 0, 0, 0, b1, basis1, b2, basis2

    ratio1 = b1 / total_bal
    ratio2 = 1.0 - ratio1

    gr1 = max(0, (b1 - basis1) / b1) if b1 > 0 else 0
    gr2 = max(0, (b2 - basis2) / b2) if b2 > 0 else 0

    def net_yield(gross_draw):
        d1 = min(gross_draw, total_bal) * ratio1
        d2 = min(gross_draw, total_bal) * ratio2
        gains = (d1 * gr1) + (d2 * gr2)
        fed_cg_tax = calc_mfj_cg_tax(gains, ordinary_gross)
        state_cg_tax = gains * state_tax_dec
        return gross_draw - fed_cg_tax - state_cg_tax

    max_net = net_yield(total_bal)

    if net_needed <= max_net:
        if net_yield(net_needed) >= net_needed:
            target_gross = net_needed
        else:
            res = opt.root_scalar(lambda g: net_yield(g) - net_needed, bracket=[net_needed, total_bal])
            target_gross = res.root

        d1 = target_gross * ratio1
        d2 = target_gross * ratio2
        gains = (d1 * gr1) + (d2 * gr2)
        fed_tax = calc_mfj_cg_tax(gains, ordinary_gross)
        state_tax = gains * state_tax_dec
        net_provided = net_needed
    else:
        target_gross = total_bal
        d1 = b1
        d2 = b2
        gains = (d1 * gr1) + (d2 * gr2)
        fed_tax = calc_mfj_cg_tax(gains, ordinary_gross)
        state_tax = gains * state_tax_dec
        net_provided = max_net

        if allow_neg:
            shortfall = net_needed - max_net
            d1 += shortfall / 2
            d2 += shortfall / 2
            net_provided = net_needed

    nb1 = b1 - d1
    nbasis1 = basis1 - (d1 * (basis1 / b1)) if b1 > 0 else basis1

    nb2 = b2 - d2
    nbasis2 = basis2 - (d2 * (basis2 / b2)) if b2 > 0 else basis2

    return net_provided, fed_tax, state_tax, nb1, nbasis1, nb2, nbasis2


# ==========================================
# DASHBOARD UI SETUP
# ==========================================
st.title("Auto-Optimized Retirement Engine")

if profile_loaded_name:
    st.success(f"✅ Successfully loaded configuration: `{profile_loaded_name}`")
else:
    st.warning(
        "⚠️ No custom profile detected. Using generic default settings. (Upload a 'my_profile.json' in the sidebar).")

tab1, tab2, tab3 = st.tabs(["📊 Retirement Dashboard", "📖 User Manual & Explainer", "📜 The Financial Story"])

with tab1:
    st.header("1. Macro Assumptions & Goals")
    mac1, mac2, mac3, mac4, mac5, mac6 = st.columns(6)
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
        pre_ret_return = st.number_input("Pre-Ret Return (%)", value=float(config["pre_ret_return"]), step=0.1,
                                         help="Expected annual investment return before retirement (net of annual dividend drag).") / 100
    with mac5:
        post_ret_return = st.number_input("Post-Ret Return (%)", value=float(config["post_ret_return"]), step=0.1,
                                          help="Expected annual investment return during retirement decumulation.") / 100
    with mac6:
        target_gross_income = st.number_input("Target Gross Ret. Spend ($)", value=int(config["target_gross_income"]),
                                              step=5000,
                                              help="Target gross annual retirement spending. The model solves for the net cash this produces after ordinary taxes.")

    st.markdown("---")

    # ----------------------------------------
    # SECTION 2: Income & Salaries
    # ----------------------------------------
    st.header("2. Base Salaries")
    col2a, col2b = st.columns(2)
    with col2a:
        p1_salary = st.number_input("Person 1 Salary ($)", value=int(config["p1_salary"]), step=5000,
                                    help="Gross annual salary for Person 1.")
        p1_annual_raise = st.number_input("P1 Annual Raise (%)", value=float(config.get("p1_annual_raise", 2.0)),
                                          step=0.1,
                                          help="Expected annual percentage increase in wage income for Person 1.") / 100
    with col2b:
        p2_salary = st.number_input("Person 2 Salary ($)", value=int(config["p2_salary"]), step=5000,
                                    disabled=is_single, help="Gross annual salary for Person 2.")
        p2_annual_raise = st.number_input("P2 Annual Raise (%)", value=float(config.get("p2_annual_raise", 2.0)),
                                          step=0.1, disabled=is_single,
                                          help="Expected annual percentage increase in wage income for Person 2.") / 100

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------
    # SECTION 3: Pre-Tax Accounts
    # ----------------------------------------
    st.header("3. Pre-Tax Accounts (Traditional 401k & IRA)")
    col3a, col3b = st.columns(2)
    with col3a:
        st.subheader("Person 1")
        p1_trad_401k_start = st.number_input("P1 Trad 401(k) Start Bal", value=int(config["p1_trad_401k_start"]),
                                             step=10000,
                                             help="Current balance in Person 1's Traditional Pre-Tax 401(k).")
        p1_trad_401k_cont = st.number_input("P1 Trad 401(k) Contrib (%)", value=float(config["p1_trad_401k_cont"]),
                                            step=1.0,
                                            help="Percentage of Person 1's salary contributed to their Pre-Tax 401(k).") / 100
        p1_trad_401k_match = st.number_input("P1 Trad 401(k) Match (%)", value=float(config["p1_trad_401k_match"]),
                                             step=1.0,
                                             help="Employer matching percentage deposited as Pre-Tax Traditional.") / 100
        p1_trad_401k_flat = st.number_input("P1 401(k) Flat Bonus/Yr ($)", value=int(config["p1_trad_401k_flat"]),
                                            step=1000,
                                            help="Fixed annual non-elective employer contribution or profit sharing.")
        p1_trad_ira_start = st.number_input("P1 Trad IRA Start Bal", value=int(config["p1_trad_ira_start"]), step=5000,
                                            help="Current balance in Person 1's Traditional IRA.")
        p1_trad_ira_mo = st.number_input("P1 Trad IRA Monthly ($)", value=int(config["p1_trad_ira_mo"]), step=100,
                                         help="Fixed monthly out-of-pocket contribution to Person 1's Traditional IRA.")
    with col3b:
        st.subheader("Person 2")
        p2_trad_401k_start = st.number_input("P2 Trad 401(k) Start Bal", value=int(config["p2_trad_401k_start"]),
                                             step=10000, disabled=is_single,
                                             help="Current balance in Person 2's Traditional Pre-Tax 401(k).")
        p2_trad_401k_cont = st.number_input("P2 Trad 401(k) Contrib (%)", value=float(config["p2_trad_401k_cont"]),
                                            step=1.0, disabled=is_single,
                                            help="Percentage of Person 2's salary contributed to their Pre-Tax 401(k).") / 100
        p2_trad_401k_match = st.number_input("P2 Trad 401(k) Match (%)", value=float(config["p2_trad_401k_match"]),
                                             step=1.0, disabled=is_single,
                                             help="Employer matching percentage deposited as Pre-Tax Traditional.") / 100
        p2_trad_401k_flat = st.number_input("P2 401(k) Flat Bonus/Yr ($)", value=int(config["p2_trad_401k_flat"]),
                                            step=1000, disabled=is_single,
                                            help="Fixed annual non-elective employer contribution or profit sharing.")
        p2_trad_ira_start = st.number_input("P2 Trad IRA Start Bal", value=int(config["p2_trad_ira_start"]), step=5000,
                                            disabled=is_single, help="Current balance in Person 2's Traditional IRA.")
        p2_trad_ira_mo = st.number_input("P2 Trad IRA Monthly ($)", value=int(config["p2_trad_ira_mo"]), step=100,
                                         disabled=is_single,
                                         help="Fixed monthly out-of-pocket contribution to Person 2's Traditional IRA.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------
    # SECTION 4: Post-Tax Accounts
    # ----------------------------------------
    st.header("4. Post-Tax Accounts (Roth 401k & IRA)")
    col4a, col4b = st.columns(2)
    with col4a:
        st.subheader("Person 1")
        p1_roth_401k_start = st.number_input("P1 Roth 401(k) Start Bal", value=int(config["p1_roth_401k_start"]),
                                             step=5000, help="Current balance in Person 1's designated Roth 401(k).")
        p1_roth_401k_cont = st.number_input("P1 Roth 401(k) Contrib (%)", value=float(config["p1_roth_401k_cont"]),
                                            step=1.0,
                                            help="Percentage of Person 1's salary contributed to their Roth 401(k).") / 100
        p1_roth_401k_match = st.number_input("P1 Roth 401(k) Match (%)",
                                             value=float(config.get("p1_roth_401k_match", 0.0)), step=1.0,
                                             help="Employer matching percentage legally deposited as Roth (per SECURE 2.0).") / 100
        p1_roth_ira_start = st.number_input("P1 Roth IRA Start Bal", value=int(config["p1_roth_ira_start"]), step=5000,
                                            help="Current balance in Person 1's Roth IRA.")
        p1_roth_ira_mo = st.number_input("P1 Roth IRA Monthly ($)", value=int(config["p1_roth_ira_mo"]), step=100,
                                         help="Fixed monthly contribution to Person 1's Roth IRA (subject to annual IRS limits).")
    with col4b:
        st.subheader("Person 2")
        p2_roth_401k_start = st.number_input("P2 Roth 401(k) Start Bal", value=int(config["p2_roth_401k_start"]),
                                             step=5000, disabled=is_single,
                                             help="Current balance in Person 2's designated Roth 401(k).")
        p2_roth_401k_cont = st.number_input("P2 Roth 401(k) Contrib (%)", value=float(config["p2_roth_401k_cont"]),
                                            step=1.0, disabled=is_single,
                                            help="Percentage of Person 2's salary contributed to their Roth 401(k).") / 100
        p2_roth_401k_match = st.number_input("P2 Roth 401(k) Match (%)",
                                             value=float(config.get("p2_roth_401k_match", 0.0)), step=1.0,
                                             disabled=is_single,
                                             help="Employer matching percentage legally deposited as Roth (per SECURE 2.0).") / 100
        p2_roth_ira_start = st.number_input("P2 Roth IRA Start Bal", value=int(config["p2_roth_ira_start"]), step=5000,
                                            disabled=is_single, help="Current balance in Person 2's Roth IRA.")
        p2_roth_ira_mo = st.number_input("P2 Roth IRA Monthly ($)", value=int(config["p2_roth_ira_mo"]), step=100,
                                         disabled=is_single, help="Fixed monthly contribution to Person 2's Roth IRA.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ----------------------------------------
    # SECTION 5: Taxable Accounts
    # ----------------------------------------
    st.header("5. Taxable Accounts (Brokerage)")
    col5a, col5b = st.columns(2)
    with col5a:
        st.subheader("Person 1")
        p1_brok_start = st.number_input("P1 Brokerage Start Bal", value=int(config["p1_brok_start"]), step=5000,
                                        help="Current balance in Person 1's Non-Qualified Taxable Brokerage account.")
        p1_brok_mo = st.number_input("P1 Brokerage Monthly ($)", value=int(config["p1_brok_mo"]), step=100,
                                     help="Monthly post-tax contribution to Person 1's Taxable Brokerage.")
    with col5b:
        st.subheader("Person 2")
        p2_brok_start = st.number_input("P2 Brokerage Start Bal", value=int(config["p2_brok_start"]), step=5000,
                                        disabled=is_single,
                                        help="Current balance in Person 2's Non-Qualified Taxable Brokerage account.")
        p2_brok_mo = st.number_input("P2 Brokerage Monthly ($)", value=int(config["p2_brok_mo"]), step=100,
                                     disabled=is_single,
                                     help="Monthly post-tax contribution to Person 2's Taxable Brokerage.")

    raw_p2_salary = p2_salary
    raw_p2_annual_raise = p2_annual_raise
    raw_p2_trad_401k_start = p2_trad_401k_start;
    raw_p2_trad_401k_cont = p2_trad_401k_cont
    raw_p2_trad_401k_match = p2_trad_401k_match;
    raw_p2_trad_401k_flat = p2_trad_401k_flat
    raw_p2_trad_ira_start = p2_trad_ira_start;
    raw_p2_trad_ira_mo = p2_trad_ira_mo
    raw_p2_roth_401k_start = p2_roth_401k_start;
    raw_p2_roth_401k_cont = p2_roth_401k_cont;
    raw_p2_roth_401k_match = p2_roth_401k_match
    raw_p2_roth_ira_start = p2_roth_ira_start;
    raw_p2_roth_ira_mo = p2_roth_ira_mo
    raw_p2_brok_start = p2_brok_start;
    raw_p2_brok_mo = p2_brok_mo

    if is_single:
        p2_salary = 0
        p2_annual_raise = 0.0
        p2_trad_401k_start = 0;
        p2_trad_401k_cont = 0.0;
        p2_trad_401k_match = 0.0;
        p2_trad_401k_flat = 0
        p2_trad_ira_start = 0;
        p2_trad_ira_mo = 0
        p2_roth_401k_start = 0;
        p2_roth_401k_cont = 0.0;
        p2_roth_401k_match = 0.0
        p2_roth_ira_start = 0;
        p2_roth_ira_mo = 0
        p2_brok_start = 0;
        p2_brok_mo = 0

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
        "pre_ret_return": round(pre_ret_return * 100, 2),
        "post_ret_return": round(post_ret_return * 100, 2),
        "rmd_start_age": rmd_start_age,
        "p1_salary": p1_salary,
        "p1_annual_raise": round(p1_annual_raise * 100, 2),
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
        "p2_salary": raw_p2_salary,
        "p2_annual_raise": round(raw_p2_annual_raise * 100, 2),
        "p2_trad_401k_start": raw_p2_trad_401k_start,
        "p2_trad_401k_cont": round(raw_p2_trad_401k_cont * 100, 2),
        "p2_trad_401k_match": round(raw_p2_trad_401k_match * 100, 2),
        "p2_trad_401k_flat": raw_p2_trad_401k_flat,
        "p2_trad_ira_start": raw_p2_trad_ira_start,
        "p2_trad_ira_mo": raw_p2_trad_ira_mo,
        "p2_roth_401k_start": raw_p2_roth_401k_start,
        "p2_roth_401k_cont": round(raw_p2_roth_401k_cont * 100, 2),
        "p2_roth_401k_match": round(raw_p2_roth_401k_match * 100, 2),
        "p2_roth_ira_start": raw_p2_roth_ira_start,
        "p2_roth_ira_mo": raw_p2_roth_ira_mo,
        "p2_brok_start": raw_p2_brok_start,
        "p2_brok_mo": raw_p2_brok_mo,
        "target_gross_income": target_gross_income,
        "penalty_age": penalty_age,
        "penalty_pct": round(penalty_pct * 100, 2),
        "use_glide_path": use_glide_path,
        "use_smile_model": use_smile_model
    }

    json_export = json.dumps(current_settings, indent=4)
    st.sidebar.markdown("---")
    st.sidebar.header("💾 Save Current Profile")
    st.sidebar.download_button(
        label="Download my_profile.json",
        data=json_export,
        file_name="my_profile.json",
        mime="application/json",
        help="Click to save all current inputs, state selections, and settings into a standardized configuration file."
    )


    # ==========================================
    # SIMULATION ENGINE
    # ==========================================
    def run_simulation(test_gross_income, return_data=False, allow_negative_brokerage=False):
        data = []

        p1_trad_bal = p1_trad_401k_start + p1_trad_ira_start
        p2_trad_bal = p2_trad_401k_start + p2_trad_ira_start
        p1_roth_bal = p1_roth_401k_start + p1_roth_ira_start
        p2_roth_bal = p2_roth_401k_start + p2_roth_ira_start
        p1_brok_bal = p1_brok_start
        p1_brok_basis = p1_brok_start
        p2_brok_bal = p2_brok_start
        p2_brok_basis = p2_brok_start

        curr_p1_sal, curr_p2_sal = p1_salary, p2_salary

        tot_penalties = tot_roth_conv_net = tot_rmd_overflow_net = 0
        tot_tax_living = tot_tax_roth = tot_tax_rmd = tot_tax_brokerage = portfolio_at_retire = 0
        tot_state_tax_retired = 0

        retire_balances_dict = {}
        depletion_age = None
        ending_bal = 0

        max_sim_age = max(151, target_lifespan + 1)

        base_net_spend = test_gross_income - calc_mfj_tax(test_gross_income, is_penalized=False)
        if not state_exempts_ret:
            base_net_spend -= (test_gross_income * state_tax_decimal)

        for age in range(current_age, max_sim_age):
            is_retired = age >= retire_age
            is_penalized = age >= retire_age and age < penalty_age

            trad_bal_tot = p1_trad_bal + p2_trad_bal
            roth_bal_tot = p1_roth_bal + p2_roth_bal
            brok_bal_tot = p1_brok_bal + p2_brok_bal

            if return_data and depletion_age is None and (trad_bal_tot + roth_bal_tot + brok_bal_tot) < 1:
                depletion_age = age

            if age == retire_age or (age == current_age and current_age >= retire_age):
                portfolio_at_retire = trad_bal_tot + roth_bal_tot + brok_bal_tot
                retire_balances_dict = {
                    "P1 Pre-Tax": p1_trad_bal,
                    "P2 Pre-Tax": p2_trad_bal,
                    "P1 Post-Tax (Roth)": p1_roth_bal,
                    "P2 Post-Tax (Roth)": p2_roth_bal,
                    "P1 Brokerage": p1_brok_bal,
                    "P2 Brokerage": p2_brok_bal
                }

            current_return = get_glide_return(age, retire_age, pre_ret_return, post_ret_return) if use_glide_path else (
                pre_ret_return if not is_retired else post_ret_return)

            current_spend = get_smile_spending(age, base_net_spend) if use_smile_model and is_retired else (
                base_net_spend if is_retired else 0)

            # Year tracking variables for detailed table cash flows
            fed_tax_paid_yr = 0
            state_tax_paid_yr = 0
            cg_tax_paid_yr = 0
            penalty_paid_yr = 0
            rmd_amt_yr = 0

            pre_tax_in_yr = 0
            pre_tax_out_yr = 0
            roth_in_yr = 0
            roth_out_yr = 0
            brok_in_yr = 0
            brok_out_yr = 0

            if not is_retired:
                p1_trad_in = (curr_p1_sal * p1_trad_401k_cont) + (
                            curr_p1_sal * p1_trad_401k_match) + p1_trad_401k_flat + (p1_trad_ira_mo * 12)
                p2_trad_in = (curr_p2_sal * p2_trad_401k_cont) + (
                            curr_p2_sal * p2_trad_401k_match) + p2_trad_401k_flat + (p2_trad_ira_mo * 12)

                p1_trad_bal = (p1_trad_bal + p1_trad_in) * (1 + current_return)
                p2_trad_bal = (p2_trad_bal + p2_trad_in) * (1 + current_return)

                p1_roth_in = (curr_p1_sal * p1_roth_401k_cont) + (curr_p1_sal * p1_roth_401k_match) + (
                            p1_roth_ira_mo * 12)
                p2_roth_in = (curr_p2_sal * p2_roth_401k_cont) + (curr_p2_sal * p2_roth_401k_match) + (
                            p2_roth_ira_mo * 12)

                p1_roth_bal = (p1_roth_bal + p1_roth_in) * (1 + current_return)
                p2_roth_bal = (p2_roth_bal + p2_roth_in) * (1 + current_return)

                p1_brok_in = p1_brok_mo * 12
                p2_brok_in = p2_brok_mo * 12
                p1_brok_bal = (p1_brok_bal + p1_brok_in) * (1 + current_return)
                p1_brok_basis += p1_brok_in
                p2_brok_bal = (p2_brok_bal + p2_brok_in) * (1 + current_return)
                p2_brok_basis += p2_brok_in

                pre_tax_in_yr = p1_trad_in + p2_trad_in
                roth_in_yr = p1_roth_in + p2_roth_in
                brok_in_yr = p1_brok_in + p2_brok_in

                curr_p1_sal *= (1 + p1_annual_raise)
                curr_p2_sal *= (1 + p2_annual_raise)

                status_label = "Working"

            elif is_penalized:
                shortfall = current_spend
                take_roth = min(roth_bal_tot, shortfall)
                r1, r2 = withdraw_proportional(take_roth, p1_roth_bal, p2_roth_bal)
                p1_roth_bal -= r1;
                p2_roth_bal -= r2
                shortfall -= take_roth

                roth_out_yr = take_roth

                if shortfall > 0:
                    ordinary_gross = 0
                    net_brok_drawn, brok_tax_fed, brok_tax_state, p1_brok_bal, p1_brok_basis, p2_brok_bal, p2_brok_basis = withdraw_from_dual_brokerage_dynamic(
                        shortfall, p1_brok_bal, p1_brok_basis, p2_brok_bal, p2_brok_basis, ordinary_gross,
                        state_tax_decimal, allow_negative_brokerage)
                    shortfall -= net_brok_drawn

                    brok_out_yr = net_brok_drawn + brok_tax_fed + brok_tax_state
                    cg_tax_paid_yr += brok_tax_fed
                    state_tax_paid_yr += brok_tax_state

                    tot_tax_brokerage += brok_tax_fed
                    tot_state_tax_retired += brok_tax_state

                if shortfall > 0 and trad_bal_tot > 0:
                    gross_forced = get_gross_for_net(shortfall, state_tax_decimal, state_exempts_ret, is_penalized=True,
                                                     penalty_pct=penalty_pct)
                    take_trad = min(trad_bal_tot, gross_forced)

                    t1, t2 = withdraw_proportional(take_trad, p1_trad_bal, p2_trad_bal)
                    p1_trad_bal -= t1;
                    p2_trad_bal -= t2

                    trad_tax_fed = calc_mfj_tax(take_trad, is_penalized=True, penalty_pct=penalty_pct)
                    trad_tax_state = 0 if state_exempts_ret else (take_trad * state_tax_decimal)

                    pre_tax_out_yr = take_trad
                    fed_tax_paid_yr += trad_tax_fed
                    state_tax_paid_yr += trad_tax_state
                    penalty_paid_yr = take_trad * penalty_pct

                    tot_penalties += penalty_paid_yr
                    tot_tax_living += trad_tax_fed
                    tot_state_tax_retired += trad_tax_state

                p1_trad_bal *= (1 + current_return);
                p2_trad_bal *= (1 + current_return)
                p1_roth_bal *= (1 + current_return);
                p2_roth_bal *= (1 + current_return)
                p1_brok_bal *= (1 + current_return);
                p2_brok_bal *= (1 + current_return)
                status_label = "Phase 1 (Penalty)"
            else:
                years_remaining = max(1, target_lifespan - age + 1)
                if age > target_lifespan: years_remaining = max(1, 120 - age + 1)

                pmt = -npf.pmt(current_return, years_remaining, trad_bal_tot) if trad_bal_tot > 0 else 0

                rmd = 0
                if age >= rmd_start_age and trad_bal_tot > 0:
                    rmd = trad_bal_tot / get_rmd_divisor(age)
                    rmd_amt_yr = rmd

                gross_need = get_gross_for_net(current_spend, state_tax_decimal, state_exempts_ret, is_penalized=False)
                gross_trad = min(trad_bal_tot, max(rmd, pmt, gross_need))
                pre_tax_out_yr = gross_trad

                trad_tax_fed = calc_mfj_tax(gross_trad, is_penalized=False)
                trad_tax_state = 0 if state_exempts_ret else (gross_trad * state_tax_decimal)

                net_trad = gross_trad - (trad_tax_fed + trad_tax_state)
                fed_tax_paid_yr += trad_tax_fed
                state_tax_paid_yr += trad_tax_state

                t1, t2 = withdraw_proportional(gross_trad, p1_trad_bal, p2_trad_bal)
                p1_trad_bal -= t1;
                p2_trad_bal -= t2

                if net_trad >= current_spend:
                    surplus = net_trad - current_spend
                    ratio_living = current_spend / net_trad if net_trad > 0 else 0
                    ratio_surplus = surplus / net_trad if net_trad > 0 else 0

                    tot_tax_living += trad_tax_fed * ratio_living
                    tot_state_tax_retired += trad_tax_state

                    if age < rmd_start_age:
                        p1_roth_bal += surplus / 2;
                        p2_roth_bal += surplus / 2
                        roth_in_yr = surplus
                        tot_roth_conv_net += surplus
                        tot_tax_roth += trad_tax_fed * ratio_surplus
                    else:
                        brok_tot = p1_brok_bal + p2_brok_bal
                        p1_ratio = p1_brok_bal / brok_tot if brok_tot > 0 else 0.5
                        p1_brok_bal += surplus * p1_ratio
                        p1_brok_basis += surplus * p1_ratio
                        p2_brok_bal += surplus * (1 - p1_ratio)
                        p2_brok_basis += surplus * (1 - p1_ratio)

                        brok_in_yr = surplus
                        tot_rmd_overflow_net += surplus
                        tot_tax_rmd += trad_tax_fed * ratio_surplus
                else:
                    tot_tax_living += trad_tax_fed
                    tot_state_tax_retired += trad_tax_state

                    shortfall = current_spend - net_trad
                    take_roth = min(roth_bal_tot, shortfall)
                    r1, r2 = withdraw_proportional(take_roth, p1_roth_bal, p2_roth_bal)
                    p1_roth_bal -= r1;
                    p2_roth_bal -= r2
                    shortfall -= take_roth

                    roth_out_yr = take_roth

                    if shortfall > 0:
                        ordinary_gross = gross_trad
                        net_brok_drawn, brok_tax_fed, brok_tax_state, p1_brok_bal, p1_brok_basis, p2_brok_bal, p2_brok_basis = withdraw_from_dual_brokerage_dynamic(
                            shortfall, p1_brok_bal, p1_brok_basis, p2_brok_bal, p2_brok_basis, ordinary_gross,
                            state_tax_decimal, allow_negative_brokerage)
                        shortfall -= net_brok_drawn

                        brok_out_yr = net_brok_drawn + brok_tax_fed + brok_tax_state
                        cg_tax_paid_yr += brok_tax_fed
                        state_tax_paid_yr += brok_tax_state

                        tot_tax_brokerage += brok_tax_fed
                        tot_state_tax_retired += brok_tax_state

                p1_trad_bal *= (1 + current_return);
                p2_trad_bal *= (1 + current_return)
                p1_roth_bal *= (1 + current_return);
                p2_roth_bal *= (1 + current_return)
                p1_brok_bal *= (1 + current_return);
                p2_brok_bal *= (1 + current_return)
                status_label = "Phase 2 (Levelized)" if age < rmd_start_age else "Phase 3 (RMDs)"

            if return_data and age <= target_lifespan:
                data.append([
                    age, status_label, f"{current_return * 100:.2f}%", round(current_spend),
                    round(p1_trad_bal), round(p2_trad_bal), round(p1_roth_bal), round(p2_roth_bal),
                    round(p1_brok_bal), round(p2_brok_bal),
                    round(pre_tax_in_yr), round(pre_tax_out_yr),
                    round(roth_in_yr), round(roth_out_yr),
                    round(brok_in_yr), round(brok_out_yr),
                    round(rmd_amt_yr), round(fed_tax_paid_yr), round(state_tax_paid_yr), round(cg_tax_paid_yr),
                    round(penalty_paid_yr)
                ])

            if age == target_lifespan:
                ending_bal = p1_trad_bal + p2_trad_bal + p1_roth_bal + p2_roth_bal + p1_brok_bal + p2_brok_bal
                if not return_data: return ending_bal

        if depletion_age is None: depletion_age = "Never"

        if return_data:
            return data, ending_bal, portfolio_at_retire, tot_penalties, tot_roth_conv_net, tot_rmd_overflow_net, tot_tax_living, tot_tax_roth, tot_tax_rmd, tot_tax_brokerage, tot_state_tax_retired, depletion_age, retire_balances_dict
        return ending_bal


    # --- RUN SOLVERS ---
    sim_results = run_simulation(target_gross_income, return_data=True)
    data, _, portfolio_at_retire, tot_penalties, tot_roth_conv_net, tot_rmd_overflow_net, tot_tax_living, tot_tax_roth, tot_tax_rmd, tot_tax_brokerage, tot_state_tax_retired, depletion_age, retire_balances_dict = sim_results

    try:
        res_zero = opt.root_scalar(lambda x: run_simulation(x, allow_negative_brokerage=True), bracket=[0, 3000000],
                                   method='brentq')
        max_income_zero = res_zero.root
    except ValueError:
        max_income_zero = 0

    try:
        res_stable = opt.root_scalar(lambda x: run_simulation(x, allow_negative_brokerage=True) - portfolio_at_retire,
                                     bracket=[0, 3000000], method='brentq')
        max_income_stable = res_stable.root
    except ValueError:
        max_income_stable = 0

    # ==========================================
    # DISPLAY RESULTS
    # ==========================================
    df = pd.DataFrame(data, columns=[
        "Age", "Status", "Eff. Return", "Net Spend Target",
        "P1 Pre-Tax", "P2 Pre-Tax", "P1 Post-Tax (Roth)", "P2 Post-Tax (Roth)",
        "P1 Brokerage", "P2 Brokerage",
        "Pre-Tax Additions", "Pre-Tax Withdrawals",
        "Roth Additions", "Roth Withdrawals",
        "Brokerage Additions", "Brokerage Withdrawals",
        "RMD Amount", "Fed Ordinary Tax", "State Tax", "Fed Cap Gains Tax", "10% Penalty"
    ])

    if is_single:
        df.drop(columns=["P2 Pre-Tax", "P2 Post-Tax (Roth)", "P2 Brokerage"], inplace=True)
        df['Total_Bal'] = df['P1 Pre-Tax'] + df['P1 Post-Tax (Roth)'] + df['P1 Brokerage']
    else:
        df['Total_Bal'] = df['P1 Pre-Tax'] + df['P2 Pre-Tax'] + df['P1 Post-Tax (Roth)'] + df['P2 Post-Tax (Roth)'] + \
                          df['P1 Brokerage'] + df['P2 Brokerage']

    end_of_life_balance = df[df['Age'] == target_lifespan]['Total_Bal'].iloc[0] if not df.empty else 0

    st.markdown("---")
    st.header("6. Lifetime Summary & KPIs")

    # -----------------------------------------------------
    # SUBSECTION: MONTHLY SAVINGS BREAKDOWN
    # -----------------------------------------------------
    st.subheader("Current Monthly Savings Breakdown")
    p1_trad_401k_mo = (p1_salary * p1_trad_401k_cont) / 12
    p1_roth_401k_mo = (p1_salary * p1_roth_401k_cont) / 12
    p2_trad_401k_mo = (p2_salary * p2_trad_401k_cont) / 12
    p2_roth_401k_mo = (p2_salary * p2_roth_401k_cont) / 12

    p1_trad_match_mo = (p1_salary * p1_trad_401k_match) / 12
    p1_roth_match_mo = (p1_salary * p1_roth_401k_match) / 12
    p2_trad_match_mo = (p2_salary * p2_trad_401k_match) / 12
    p2_roth_match_mo = (p2_salary * p2_roth_401k_match) / 12

    p1_flat_mo = p1_trad_401k_flat / 12
    p2_flat_mo = p2_trad_401k_flat / 12

    total_employee = p1_trad_401k_mo + p1_roth_401k_mo + p1_trad_ira_mo + p1_roth_ira_mo + p1_brok_mo + \
                     p2_trad_401k_mo + p2_roth_401k_mo + p2_trad_ira_mo + p2_roth_ira_mo + p2_brok_mo
    total_employer = p1_trad_match_mo + p1_roth_match_mo + p1_flat_mo + p2_trad_match_mo + p2_roth_match_mo + p2_flat_mo
    total_saved = total_employee + total_employer

    if is_single:
        cont_col1, cont_col3 = st.columns(2)
        with cont_col1:
            st.markdown("**Person 1 Monthly Savings:**")
            st.markdown(f"- Trad 401(k): \${p1_trad_401k_mo:,.0f} *(+ \${p1_trad_match_mo:,.0f} Match)*")
            st.markdown(f"- Trad IRA: \${p1_trad_ira_mo:,.0f}")
            st.markdown(f"- Roth 401(k): \${p1_roth_401k_mo:,.0f} *(+ \${p1_roth_match_mo:,.0f} Match)*")
            st.markdown(f"- Roth IRA: \${p1_roth_ira_mo:,.0f}")
            st.markdown(f"- Brokerage: \${p1_brok_mo:,.0f}")
        with cont_col3:
            st.markdown("**Total Household Monthly:**")
            st.markdown(f"- **Total Employee (Out of Pocket):** \${total_employee:,.0f}")
            st.markdown(f"- **Total Employer (% Matches):** \${p1_trad_match_mo + p1_roth_match_mo:,.0f}")
            st.markdown(f"- **Total Employer (Flat/Bonus):** \${p1_flat_mo:,.0f}")
            st.metric("Total Monthly Saved", f"${total_saved:,.0f}")
    else:
        cont_col1, cont_col2, cont_col3 = st.columns(3)
        with cont_col1:
            st.markdown("**Person 1 Monthly Savings:**")
            st.markdown(f"- Trad 401(k): \${p1_trad_401k_mo:,.0f} *(+ \${p1_trad_match_mo:,.0f} Match)*")
            st.markdown(f"- Trad IRA: \${p1_trad_ira_mo:,.0f}")
            st.markdown(f"- Roth 401(k): \${p1_roth_401k_mo:,.0f} *(+ \${p1_roth_match_mo:,.0f} Match)*")
            st.markdown(f"- Roth IRA: \${p1_roth_ira_mo:,.0f}")
            st.markdown(f"- Brokerage: \${p1_brok_mo:,.0f}")
        with cont_col2:
            st.markdown("**Person 2 Monthly Savings:**")
            st.markdown(f"- Trad 401(k): \${p2_trad_401k_mo:,.0f} *(+ \${p2_trad_match_mo:,.0f} Match)*")
            st.markdown(f"- Trad IRA: \${p2_trad_ira_mo:,.0f}")
            st.markdown(f"- Roth 401(k): \${p2_roth_401k_mo:,.0f} *(+ \${p2_roth_match_mo:,.0f} Match)*")
            st.markdown(f"- Roth IRA: \${p2_roth_ira_mo:,.0f}")
            st.markdown(f"- Brokerage: \${p2_brok_mo:,.0f}")
        with cont_col3:
            st.markdown("**Total Household Monthly:**")
            st.markdown(f"- **Total Employee (Out of Pocket):** \${total_employee:,.0f}")
            st.markdown(
                f"- **Total Employer (% Matches):** \${p1_trad_match_mo + p2_trad_match_mo + p1_roth_match_mo + p2_roth_match_mo:,.0f}")
            st.markdown(f"- **Total Employer (Flat/Bonus):** \${p1_flat_mo + p2_flat_mo:,.0f}")
            st.metric("Total Monthly Saved", f"${total_saved:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # SUBSECTION: PORTFOLIO MILESTONES & SOLVERS
    # -----------------------------------------------------
    st.subheader("Portfolio Milestones & Solvers")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Portfolio Balance at Retirement", f"${portfolio_at_retire:,.0f}",
                  help="Projected value of all accounts at the exact retirement age.")
    with kpi2:
        st.metric(f"Portfolio Balance at Age {target_lifespan}", f"${end_of_life_balance:,.0f}",
                  help=f"Projected net worth across all accounts at your target lifespan age ({target_lifespan}).")
    with kpi3:
        st.metric("Stable Portfolio Solver (Preserve Principal)", f"${max_income_stable:,.0f}",
                  help="The exact sustainable gross withdrawal where terminal balance at target lifespan equals starting balance at retirement.")
    with kpi4:
        st.metric("Max Spend Solver (Die at Zero)", f"${max_income_zero:,.0f}",
                  help="The maximum sustainable gross withdrawal that depletes the entire portfolio to exactly $0 at the target lifespan age.")

    st.markdown("<br>", unsafe_allow_html=True)

    # -----------------------------------------------------
    # SUBSECTION: TAX ANALYSIS & ADVANCED STRATEGY
    # -----------------------------------------------------
    st.subheader("Tax Analysis & Advanced Strategy")
    tax_col1, tax_col2, tax_col3, tax_col4 = st.columns(4)
    with tax_col1:
        st.metric("Federal Tax (Retirement Living)", f"${tot_tax_living:,.0f}",
                  help="Cumulative federal income taxes paid on withdrawals used directly to fund your baseline living expenses.")
        st.metric("Federal Cap Gains (Retirement)", f"${tot_tax_brokerage:,.0f}",
                  help="Cumulative federal long-term capital gains taxes automatically evaluated across progressive 0%, 15%, and 20% IRS brackets.")
    with tax_col2:
        st.metric("State Tax (Retirement)", f"${tot_state_tax_retired:,.0f}",
                  help="Total state income taxes paid on pre-tax distributions and capital gains during the decumulation phase.")
        st.metric("Total Early Penalties Paid", f"${tot_penalties:,.0f}",
                  help="Total 10% IRS penalties incurred from forced early pre-tax withdrawals prior to penalty age.")
    with tax_col3:
        st.metric("Total Tax-Free Roth Conversions", f"${tot_roth_conv_net:,.0f}",
                  help="Cumulative surplus net withdrawals successfully rolled into Roth accounts during Phase 2 (ages 60 to 74).")
        st.metric("Taxes: Roth Conversion", f"${tot_tax_roth:,.0f}",
                  help="Cumulative federal taxes paid to convert surplus pre-tax dollars into tax-free Roth accounts during Phase 2.")
    with tax_col4:
        st.metric("Total RMD Overflows to Brokerage", f"${tot_rmd_overflow_net:,.0f}",
                  help="Cumulative after-tax distributions forced by IRS RMDs (age 75+) that exceeded living expenses and swept into Taxable accounts.")
        st.metric("Taxes: RMD Overflows", f"${tot_tax_rmd:,.0f}",
                  help="Cumulative federal taxes paid on mandatory RMD withdrawals that exceeded your lifestyle spending requirements.")

    st.markdown("---")
    st.subheader("Account Balances Over Time")

    y_cols = ["P1 Pre-Tax", "P1 Post-Tax (Roth)", "P1 Brokerage"]
    if not is_single:
        y_cols.extend(["P2 Pre-Tax", "P2 Post-Tax (Roth)", "P2 Brokerage"])

    # Sort columns so the largest balance at retirement forms the base of the area chart
    y_cols.sort(key=lambda x: retire_balances_dict.get(x, 0), reverse=True)

    chart_data = df[["Age"] + y_cols]
    fig = px.area(
        chart_data, x="Age", y=y_cols,
        labels={"value": "Account Balance ($)", "variable": "Account Bucket"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("7. Lifetime Projection & Automated Waterfall")
    st.dataframe(df.drop(columns=['Total_Bal']), use_container_width=True)

# ==========================================
# MANUAL & EXPLAINER TAB
# ==========================================
with tab2:
    st.title("📖 User Manual & Architectural Assumptions")
    st.markdown(r"""
    Welcome to the Auto-Optimized Retirement Engine. This platform models multi-asset accumulation, progressive tax brackets, state and local constraints, IRS withdrawal restrictions, and decumulation waterfalls.

    ---

    ## 1. Structural Assumptions & Modeling Scope

    ### Tax Environments & Annual "Tax Drag"
    The engine categorizes all assets into three functional tax environments:
    *   **Pre-Tax (Traditional 401(k) & Traditional IRA):** 
        *   Contributions enter gross before tax.
        *   Assets compound with **100% tax deferral** (no annual taxes on dividends, interest, or turnover).
        *   All distributions are taxed as ordinary income under Married Filing Jointly (MFJ) rates.
        *   Subject to a **10% IRS excise penalty** on withdrawals prior to the Penalty Age.
    *   **Post-Tax (Roth 401(k) & Roth IRA):** 
        *   Funded strictly with after-tax capital.
        *   Assets compound with **100% tax immunity**—no annual tax drag on dividends, capital gains distributions, or turnover.
        *   Qualified distributions are **$0 taxable**.
        *   In accordance with SECURE 2.0, Roth 401(k)s are treated as exempt from Required Minimum Distributions (RMDs), behaving identically to Roth IRAs during decumulation.
    *   **Taxable (Non-Qualified Brokerage):** 
        *   Funded with after-tax capital, which establishes the baseline **Cost Basis**.
        *   **Accumulation Phase Assumption:** The model assumes your Pre-Retirement Return input is **net of annual dividend tax drag**.
        *   **Decumulation Phase (Dynamic Capital Gains):** Upon liquidation, the engine computes a dynamic `Growth Ratio = (Balance - Basis) / Balance`. The taxable portion of the withdrawal is dynamically evaluated against the progressive IRS long-term capital gains brackets (0%, 15%, 20%). The capital gains are mathematically stacked on top of your ordinary income (Pre-Tax draws) for the given year, ensuring you fully exploit the 0% Capital Gains bracket up to the $94,050 threshold.

    ---

    ## 2. State & Local Taxes (SALT)
    To accurately model geographic liabilities, the engine supports four distinct state tax archetypes across the Accumulation and Decumulation phases:

    1.  **Zero-Tax States (e.g., FL, TX, WA, NV, TN, WY, SD, AK):** 0% tax on earned income, retirement distributions, and capital gains.
    2.  **Retirement-Exempt Flat States (e.g., Pennsylvania, Illinois, Mississippi, Iowa):** Imposes a statutory tax on wages and capital gains during working years (e.g., 3.07% PA state tax), but provides **100% tax exemption on qualified retirement plan distributions (401k, 403b, IRA)** during decumulation. Local Earned Income Taxes (EIT) apply exclusively to wages and exempt retirement distributions.
    3.  **Fully Taxable Progressive States (e.g., California, New York, New Jersey):** Retirement withdrawals are taxed identically to ordinary wage income, layered on top of federal brackets.
    4.  **Partial Exclusion States (e.g., Georgia, Colorado, South Carolina):** Flat or progressive rates with a capped annual exclusion deduction for individuals age 65+.

    ---

    ## 3. Decumulation Waterfall: The Three-Phase Lifecycle

    The engine dynamically sequences withdrawals based on age milestones to minimize lifetime tax liability:

    ```
    [ Accumulation: Age < Retire Age ]
                   │
                   ▼
    [ Phase 1: Penalty Shield (Retire Age to Age 59) ]
      - Protect Pre-Tax: Gross Pre-Tax Draw = $0
      - Order: Drain Roth IRAs -> Drain Taxable Brokerage -> (Forced Pre-Tax with 10% penalty only if insolvent)
                   │
                   ▼
    [ Phase 2: Levelized Smoothing & Roth Conversions (Age 60 to 74) ]
      - Target Gross = max(Levelized PMT Annuity, Gross Need for Net Lifestyle)
      - Deficit Funding: Pull from Roth -> Pull from Brokerage
      - Surplus Absorption: Net excess above lifestyle is converted into Roth IRAs
                   │
                   ▼
    [ Phase 3: Statutory RMDs & Taxable Spillover (Age 75+) ]
      - Target Gross = max(Statutory RMD, Levelized PMT Annuity, Gross Need)
      - Deficit Funding: Pull from Roth -> Pull from Brokerage
      - Surplus Sweeping: Statutory RMD surplus cannot legally enter Roth; swept into Taxable Brokerage
    ```

    ---

    ## 4. Mathematical Sub-Engines & Algorithmic Rules

    ### A. Dynamic Rate of Return (Institutional Glide Path)
    When enabled, portfolio returns transition along an institutional Target-Date Fund trajectory rather than dropping abruptly at retirement.
    *   **Baseline:** 90% Equities / 10% Fixed Income ($r_{\text{pre}}$).
    *   **Terminal Point:** 30% Equities / 70% Fixed Income ($r_{\text{post}}$).
    *   **Adjustment Window:** The step-down begins 25 years prior to retirement, passes the midpoint exactly at retirement age, and stabilizes at the conservative landing point 7 years post-retirement:

    $$\text{ytr} = \text{Retire Age} - \text{Age}$$

    $$r(t) = \begin{cases}      r_{\text{pre}} & \text{if } \text{ytr} \ge 25 \\     r_{\text{post}} & \text{if } \text{ytr} \le -7 \\     r_{\text{post}} + (r_{\text{pre}} - r_{\text{post}}) \cdot \frac{\text{ytr} - (-7)}{32} & \text{otherwise}     \end{cases}$$

    ### B. Dynamic Spending Multiplier (The Spending Smile)
    When enabled, net retirement living needs follow an empirical U-shaped curve:
    *   **Ages $\le$ 65 (Go-Go Phase):** $100\%$ of base target net income.
    *   **Ages 66–75 (Transition to Slow-Go):** Declines by $1.5\%$ per year.
    *   **Ages 76–85 (Slow-Go Trough):** Declines by $0.5\%$ per year, bottoming at $80\%$ at age 85.
    *   **Ages 86–95 (Care Curl):** Increases by $1.0\%$ per year to accommodate assisted living, ending at $90\%$ by age 95.

    ### C. Progressive Tax Engine & Inverse Net-to-Gross Solver
    *   **Federal Tax Brackets:** Based on 2024–2026 Married Filing Jointly (MFJ) brackets applied after the indexed Standard Deduction.
    *   **Inverse Net-to-Gross Solver:** To satisfy an after-tax living target $N$, the engine executes a bounded Brent's root-finding solver (`scipy.optimize.root_scalar`) across the function:

    $$f(G) = G - \text{Tax}_{\text{Fed}}(G, \text{is\_penalized}) - \text{Tax}_{\text{State}}(G) - N = 0$$

    ### D. Statutory Required Minimum Distributions (RMDs)
    *   RMDs initiate at the statutory threshold (default: Age 75, per SECURE 2.0).
    *   Divisors are drawn directly from the IRS Uniform Lifetime Table (e.g., Age 75: 24.6; Age 85: 16.0; Age 95: 8.9).
    """)

# ==========================================
# FINANCIAL STORY TAB
# ==========================================
with tab3:
    st.title("📜 The Financial Story")
    st.markdown("Here is the narrative of your simulated retirement, broken down by distinct life phases.")

    household_text = "your" if is_single else "your household's combined"

    st.header(f"Phase 1: Accumulation (Age {current_age} to {retire_age})")
    if current_age < retire_age:
        st.markdown(
            f"During this period, you continued working and saving. {household_text.capitalize()} Pre-Tax, Post-Tax (Roth), and Taxable Brokerage balances grew, "
            f"and your total portfolio reached **\${portfolio_at_retire:,.0f}** upon retirement at age {retire_age}.")
    else:
        st.markdown("You are already at or past your retirement age. The accumulation phase is complete.")

    if retire_age < penalty_age:
        st.header(f"Phase 2: Penalty Shield (Age {retire_age} to {penalty_age - 1})")
        if tot_penalties > 0:
            st.markdown(
                f"Because you retired early at age {retire_age}, the engine attempted to fund your lifestyle using your Post-Tax (Roth) and Taxable Brokerage accounts to avoid the 10% IRS penalty on Pre-Tax withdrawals. However, those funds were not enough to bridge the gap, and you were forced to incur **\${tot_penalties:,.0f}** in early withdrawal penalties.")
        else:
            st.markdown(
                f"Because you retired early at age {retire_age}, the engine successfully funded your lifestyle using your Post-Tax (Roth) and Brokerage accounts. You successfully avoided all 10% early withdrawal penalties during this period!")

    drawdown_start = max(retire_age, penalty_age)
    if drawdown_start < rmd_start_age:
        st.header(f"Phase 3: Drawdown & Conversions (Age {drawdown_start} to {rmd_start_age - 1})")
        if tot_roth_conv_net > 0:
            st.markdown(
                f"During this window, you had penalty-free access to your Pre-Tax funds. The engine calculated a smooth, levelized withdrawal to last the rest of your life. Because this optimal withdrawal generated *more* cash than your actual living expenses required, the engine successfully captured the surplus and converted **\${tot_roth_conv_net:,.0f}** into your Post-Tax (Roth) bucket, filling up lower tax brackets and building tax-free wealth.")
        else:
            st.markdown(
                f"During this window, you had penalty-free access to your Pre-Tax funds. The engine calculated a withdrawal to fund your living expenses. Based on your spending needs and account balances, no surplus was available to convert to your Post-Tax accounts.")

    st.header(f"Phase 4: Statutory RMDs (Age {rmd_start_age}+)")
    if tot_rmd_overflow_net > 0:
        st.markdown(
            f"Starting at age {rmd_start_age}, the IRS mandated Required Minimum Distributions (RMDs). Because these forced withdrawals exceeded your lifestyle needs, and because RMDs legally cannot be deposited into a Roth account, the engine paid the taxes and swept **\${tot_rmd_overflow_net:,.0f}** of surplus cash proportionally into {household_text} Taxable Brokerage Accounts where it established a new cost basis and continued to grow.")
    else:
        st.markdown(
            f"Starting at age {rmd_start_age}, the IRS mandated Required Minimum Distributions (RMDs). Your lifestyle needs were large enough that all RMDs were consumed by living expenses, resulting in no surplus overflow to {household_text} Taxable Brokerage accounts.")

    st.header("The Conclusion")
    if str(depletion_age) != "Never" and int(depletion_age) <= target_lifespan:
        st.error(
            f"Unfortunately, your portfolio depleted entirely at age **{depletion_age}**. You ran out of money before reaching your target lifespan of {target_lifespan}.")
    else:
        st.success(
            f"Congratulations! You successfully reached your target lifespan of {target_lifespan} with a remaining portfolio balance of **\${end_of_life_balance:,.0f}**.")