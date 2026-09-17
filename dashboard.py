import streamlit as st
import pandas as pd
import scipy.optimize as opt
import numpy_financial as npf
import plotly.express as px
import argparse
import json
import os

# ==========================================
# CONFIGURATION LOADER
# ==========================================
config = {
    "current_age": 40, "retire_age": 65, "target_lifespan": 95,
    "pre_ret_return": 7.0, "post_ret_return": 3.5, "annual_salary_raise": 2.0, "rmd_start_age": 75,
    "p1_salary": 75000, "p1_401k_start": 50000, "p1_401k_cont": 10.0, "p1_401k_match": 5.0, "p1_401k_flat": 0,
    "p1_roth_start": 10000, "p1_roth_cont": 6000,
    "p2_salary": 75000, "p2_401k_start": 50000, "p2_401k_cont": 10.0, "p2_401k_match": 5.0, "p2_401k_flat": 0,
    "p2_roth_start": 10000, "p2_roth_cont": 6000,
    "target_gross_income": 150000, "penalty_age": 60, "penalty_pct": 10.0, "cg_tax_rate": 15.0,
    "use_glide_path": False, "use_smile_model": True
}

parser = argparse.ArgumentParser()
parser.add_argument('--profile', type=str, default=None, help="Path to JSON profile file")
args, _ = parser.parse_known_args()

if args.profile and os.path.exists(args.profile):
    with open(args.profile, 'r') as f:
        custom_config = json.load(f)
        if "target_net_income" in custom_config and "target_gross_income" not in custom_config:
            custom_config["target_gross_income"] = custom_config["target_net_income"]
        config.update(custom_config)


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


def get_gross_for_net(target_net, is_penalized=False, penalty_pct=0.10):
    if target_net <= 0: return 0
    res = opt.root_scalar(
        lambda g: (g - calc_mfj_tax(g, is_penalized, penalty_pct)) - target_net,
        bracket=[target_net, target_net * 2.5]
    )
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


def withdraw_from_brokerage(net_needed, bal, basis, cg_rate, allow_neg):
    if net_needed <= 0: return 0, 0, bal, basis
    available_bal = max(0, bal)
    if available_bal == 0:
        if allow_neg: return net_needed, 0, bal - net_needed, basis
        return 0, 0, bal, basis

    growth_ratio = max(0, (available_bal - basis) / available_bal) if available_bal > 0 else 0
    eff_tax = growth_ratio * cg_rate
    net_available = available_bal * (1 - eff_tax)

    if net_needed <= net_available:
        gross_draw = net_needed / (1 - eff_tax)
        tax_paid = gross_draw - net_needed
        new_bal = bal - gross_draw
        new_basis = basis - (gross_draw * (basis / available_bal)) if available_bal > 0 else basis
        return net_needed, tax_paid, new_bal, new_basis
    else:
        gross_draw = available_bal
        tax_paid = available_bal * eff_tax
        new_bal, new_basis = 0, 0
        net_provided = net_available
        if allow_neg:
            remaining_shortfall = net_needed - net_provided
            new_bal = -remaining_shortfall
            net_provided = net_needed
        return net_provided, tax_paid, new_bal, new_basis


# ==========================================
# DASHBOARD UI SETUP
# ==========================================
st.set_page_config(page_title="Retirement Dashboard", layout="wide")
st.title("Auto-Optimized Retirement Engine")
if args.profile:
    st.caption(f"*(Loaded custom profile: {args.profile})*")

# Create Tabs
tab1, tab2, tab3 = st.tabs(["📊 Retirement Dashboard", "📖 User Manual & Explainer", "📝 The Financial Story"])

with tab1:
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.header("1. Macro Assumptions")
        current_age = st.number_input("Current Age", value=int(config["current_age"]), step=1,
                                      help="Current age of the primary earner for timeline tracking.")
        retire_age = st.number_input("Retirement Age", value=int(config["retire_age"]), step=1,
                                     help="Age when both individuals stop working and start portfolio drawdowns.")
        target_lifespan = st.number_input("Target Lifespan Age", value=int(config["target_lifespan"]), step=1,
                                          help="The maximum age modeled in the simulation.")
        pre_ret_return = st.number_input("Pre-Retirement Return (%)", value=float(config["pre_ret_return"]), step=0.1,
                                         help="Expected portfolio annual growth rate before retirement (e.g., 6-8%).") / 100
        post_ret_return = st.number_input("Post-Retirement Return (%)", value=float(config["post_ret_return"]),
                                          step=0.1,
                                          help="Expected portfolio annual growth rate in retirement (e.g., 3-5%).") / 100
        annual_salary_raise = st.number_input("Annual Salary Raise (%)", value=float(config["annual_salary_raise"]),
                                              step=0.1,
                                              help="Annual wage growth rate applied to current salaries.") / 100
        rmd_start_age = st.number_input("RMD Start Age", value=int(config["rmd_start_age"]), step=1,
                                        help="Age the IRS mandates you begin Required Minimum Distributions (Currently 73 or 75).")

    with col2:
        st.header("2. Person 1 Portfolio")
        p1_salary = st.number_input("P1 Current Salary", value=int(config["p1_salary"]), step=5000,
                                    help="Current gross annual income for Person 1.")
        p1_401k_start = st.number_input("P1 401(k) Starting Bal", value=int(config["p1_401k_start"]), step=10000,
                                        help="Current pre-tax (Traditional) retirement balance for Person 1.")
        p1_401k_cont = st.number_input("P1 401(k) Contrib (%)", value=float(config["p1_401k_cont"]), step=1.0,
                                       help="Percentage of salary contributed to the pre-tax 401(k) by Person 1.") / 100
        p1_401k_match = st.number_input("P1 401(k) Match (%)", value=float(config["p1_401k_match"]), step=1.0,
                                        help="Employer match percentage for Person 1.") / 100
        p1_401k_flat = st.number_input("P1 401(k) Flat/Bonus", value=int(config["p1_401k_flat"]), step=1000,
                                       help="Fixed annual employer contributions (like profit sharing) for Person 1.")
        p1_roth_start = st.number_input("P1 Roth Starting Bal", value=int(config["p1_roth_start"]), step=5000,
                                        help="Current post-tax (Roth) balance for Person 1.")
        p1_roth_cont = st.number_input("P1 Roth Annual Contrib", value=int(config["p1_roth_cont"]), step=500,
                                       help="Annual fixed dollar contribution to a Roth IRA for Person 1.")

    with col3:
        st.header("3. Person 2 Portfolio")
        p2_salary = st.number_input("P2 Current Salary", value=int(config["p2_salary"]), step=5000,
                                    help="Current gross annual income for Person 2.")
        p2_401k_start = st.number_input("P2 401(k) Starting Bal", value=int(config["p2_401k_start"]), step=10000,
                                        help="Current pre-tax (Traditional) retirement balance for Person 2.")
        p2_401k_cont = st.number_input("P2 401(k) Contrib (%)", value=float(config["p2_401k_cont"]), step=1.0,
                                       help="Percentage of salary contributed to the pre-tax 401(k) by Person 2.") / 100
        p2_401k_match = st.number_input("P2 401(k) Match (%)", value=float(config["p2_401k_match"]), step=1.0,
                                        help="Employer match percentage for Person 2.") / 100
        p2_401k_flat = st.number_input("P2 401(k) Flat/Bonus", value=int(config["p2_401k_flat"]), step=1000,
                                       help="Fixed annual employer contributions (like profit sharing) for Person 2.")
        p2_roth_start = st.number_input("P2 Roth Starting Bal", value=int(config["p2_roth_start"]), step=5000,
                                        help="Current post-tax (Roth) balance for Person 2.")
        p2_roth_cont = st.number_input("P2 Roth Annual Contrib", value=int(config["p2_roth_cont"]), step=500,
                                       help="Annual fixed dollar contribution to a Roth IRA for Person 2.")

    with col4:
        st.header("4. Strategy & Settings")
        target_gross_income = st.number_input("Desired Pre-Tax Ret. Income", value=int(config["target_gross_income"]),
                                              step=5000,
                                              help="Target GROSS (pre-tax) income desired each year in retirement. The model calculates the net value of this and uses that net value to fund your lifestyle.")
        penalty_age = st.number_input("Early Withdrawal Penalty Age", value=int(config["penalty_age"]), step=1,
                                      help="Age at which the 10% early withdrawal penalty on pre-tax accounts drops off (typically 59.5, rounded to 60).")
        penalty_pct = st.number_input("Early Withdrawal Penalty (%)", value=float(config["penalty_pct"]), step=1.0,
                                      help="IRS penalty rate for early pre-tax withdrawals (default 10%).") / 100
        cg_tax_rate = st.number_input("Capital Gains Tax Rate (%)", value=float(config["cg_tax_rate"]), step=1.0,
                                      help="Tax rate applied to the growth portion of Brokerage account withdrawals.") / 100

        st.markdown("### Algorithm Toggles")
        use_glide_path = st.checkbox("Use Glide Path for Returns", value=bool(config["use_glide_path"]),
                                     help="Gradually lowers pre-retirement return down to post-retirement return over a 32-year window.")
        use_smile_model = st.checkbox("Use Retirement Spending Smile", value=bool(config["use_smile_model"]),
                                      help="Retirement needs decrease dynamically as you age, modeling go-go, slow-go, and no-go spending phases.")


    # ==========================================
    # SIMULATION ENGINE
    # ==========================================
    def run_simulation(test_gross_income, return_data=False, allow_negative_brokerage=False):
        data = []
        p1_trad_bal, p2_trad_bal = p1_401k_start, p2_401k_start
        p1_roth_bal, p2_roth_bal = p1_roth_start, p2_roth_start
        brokerage_bal = 0
        brokerage_basis = 0
        curr_p1_sal, curr_p2_sal = p1_salary, p2_salary

        tot_penalties = tot_roth_conv_net = tot_rmd_overflow_net = 0
        tot_tax_living = tot_tax_roth = tot_tax_rmd = tot_tax_brokerage = portfolio_at_retire = 0
        depletion_age = None
        ending_bal = 0

        max_sim_age = max(151, target_lifespan + 1)
        base_net_spend = test_gross_income - calc_mfj_tax(test_gross_income, is_penalized=False)

        for age in range(current_age, max_sim_age):
            is_retired = age >= retire_age
            is_penalized = age >= retire_age and age < penalty_age

            trad_bal_tot = p1_trad_bal + p2_trad_bal
            roth_bal_tot = p1_roth_bal + p2_roth_bal

            if return_data and depletion_age is None and (trad_bal_tot + roth_bal_tot + brokerage_bal) < 1:
                depletion_age = age

            if age == retire_age or (age == current_age and current_age >= retire_age):
                portfolio_at_retire = trad_bal_tot + roth_bal_tot + brokerage_bal

            current_return = get_glide_return(age, retire_age, pre_ret_return, post_ret_return) if use_glide_path else (
                pre_ret_return if not is_retired else post_ret_return)

            current_spend = get_smile_spending(age, base_net_spend) if use_smile_model and is_retired else (
                base_net_spend if is_retired else 0)

            gross_trad = 0
            tax = 0

            if not is_retired:
                p1_trad_in = (curr_p1_sal * p1_401k_cont) + (curr_p1_sal * p1_401k_match) + p1_401k_flat
                p2_trad_in = (curr_p2_sal * p2_401k_cont) + (curr_p2_sal * p2_401k_match) + p2_401k_flat

                p1_trad_bal = (p1_trad_bal + p1_trad_in) * (1 + current_return)
                p2_trad_bal = (p2_trad_bal + p2_trad_in) * (1 + current_return)

                p1_roth_bal = (p1_roth_bal + p1_roth_cont) * (1 + current_return)
                p2_roth_bal = (p2_roth_bal + p2_roth_cont) * (1 + current_return)
                brokerage_bal *= (1 + current_return)

                curr_p1_sal *= (1 + annual_salary_raise)
                curr_p2_sal *= (1 + annual_salary_raise)

                status_label = "Working"

            elif is_penalized:
                shortfall = current_spend
                take_roth = min(roth_bal_tot, shortfall)
                r1, r2 = withdraw_proportional(take_roth, p1_roth_bal, p2_roth_bal)
                p1_roth_bal -= r1;
                p2_roth_bal -= r2
                shortfall -= take_roth

                if shortfall > 0:
                    net_brok_drawn, brok_tax, brokerage_bal, brokerage_basis = withdraw_from_brokerage(
                        shortfall, brokerage_bal, brokerage_basis, cg_tax_rate, allow_negative_brokerage)
                    shortfall -= net_brok_drawn
                    tax += brok_tax
                    tot_tax_brokerage += brok_tax

                if shortfall > 0 and trad_bal_tot > 0:
                    gross_forced = get_gross_for_net(shortfall, is_penalized=True, penalty_pct=penalty_pct)
                    take_trad = min(trad_bal_tot, gross_forced)

                    t1, t2 = withdraw_proportional(take_trad, p1_trad_bal, p2_trad_bal)
                    p1_trad_bal -= t1;
                    p2_trad_bal -= t2

                    trad_tax = calc_mfj_tax(take_trad, is_penalized=True, penalty_pct=penalty_pct)
                    gross_trad = take_trad

                    tax += trad_tax
                    tot_penalties += (take_trad * penalty_pct)
                    tot_tax_living += trad_tax

                p1_trad_bal *= (1 + current_return);
                p2_trad_bal *= (1 + current_return)
                p1_roth_bal *= (1 + current_return);
                p2_roth_bal *= (1 + current_return)
                brokerage_bal *= (1 + current_return)
                status_label = "Phase 1 (Penalty)"
            else:
                years_remaining = max(1, target_lifespan - age + 1)
                if age > target_lifespan: years_remaining = max(1, 120 - age + 1)

                pmt = -npf.pmt(current_return, years_remaining, trad_bal_tot) if trad_bal_tot > 0 else 0

                rmd = 0
                if age >= rmd_start_age and trad_bal_tot > 0:
                    rmd = trad_bal_tot / get_rmd_divisor(age)

                gross_need = get_gross_for_net(current_spend, is_penalized=False)
                gross_trad = min(trad_bal_tot, max(rmd, pmt, gross_need))

                trad_tax = calc_mfj_tax(gross_trad, is_penalized=False)
                tax += trad_tax
                net_trad = gross_trad - trad_tax

                t1, t2 = withdraw_proportional(gross_trad, p1_trad_bal, p2_trad_bal)
                p1_trad_bal -= t1;
                p2_trad_bal -= t2

                if net_trad >= current_spend:
                    surplus = net_trad - current_spend
                    ratio_living = current_spend / net_trad if net_trad > 0 else 0
                    ratio_surplus = surplus / net_trad if net_trad > 0 else 0
                    tot_tax_living += trad_tax * ratio_living

                    if age < rmd_start_age:
                        p1_roth_bal += surplus / 2;
                        p2_roth_bal += surplus / 2
                        tot_roth_conv_net += surplus
                        tot_tax_roth += trad_tax * ratio_surplus
                    else:
                        brokerage_bal += surplus
                        brokerage_basis += surplus
                        tot_rmd_overflow_net += surplus
                        tot_tax_rmd += trad_tax * ratio_surplus
                else:
                    tot_tax_living += trad_tax
                    shortfall = current_spend - net_trad
                    take_roth = min(roth_bal_tot, shortfall)
                    r1, r2 = withdraw_proportional(take_roth, p1_roth_bal, p2_roth_bal)
                    p1_roth_bal -= r1;
                    p2_roth_bal -= r2
                    shortfall -= take_roth

                    if shortfall > 0:
                        net_brok_drawn, brok_tax, brokerage_bal, brokerage_basis = withdraw_from_brokerage(
                            shortfall, brokerage_bal, brokerage_basis, cg_tax_rate, allow_negative_brokerage)
                        shortfall -= net_brok_drawn
                        tax += brok_tax
                        tot_tax_brokerage += brok_tax

                p1_trad_bal *= (1 + current_return);
                p2_trad_bal *= (1 + current_return)
                p1_roth_bal *= (1 + current_return);
                p2_roth_bal *= (1 + current_return)
                brokerage_bal *= (1 + current_return)
                status_label = "Phase 2 (Levelized)" if age < rmd_start_age else "Phase 3 (RMDs)"

            if return_data and age <= target_lifespan:
                data.append([
                    age, status_label, f"{current_return * 100:.2f}%", round(current_spend),
                    round(p1_trad_bal), round(p2_trad_bal), round(p1_roth_bal), round(p2_roth_bal),
                    round(brokerage_bal), round(gross_trad), round(tax)
                ])

            if age == target_lifespan:
                ending_bal = p1_trad_bal + p2_trad_bal + p1_roth_bal + p2_roth_bal + brokerage_bal
                if not return_data: return ending_bal

        if depletion_age is None: depletion_age = "Never"

        if return_data:
            return data, ending_bal, portfolio_at_retire, tot_penalties, tot_roth_conv_net, tot_rmd_overflow_net, tot_tax_living, tot_tax_roth, tot_tax_rmd, tot_tax_brokerage, depletion_age
        return ending_bal


    # --- RUN SOLVERS ---
    sim_results = run_simulation(target_gross_income, return_data=True)
    data, _, portfolio_at_retire, tot_penalties, tot_roth_conv_net, tot_rmd_overflow_net, tot_tax_living, tot_tax_roth, tot_tax_rmd, tot_tax_brokerage, depletion_age = sim_results

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
        "Age", "Status", "Effective Return", "Target Net Spend",
        "P1 Trad 401(k)", "P2 Trad 401(k)", "P1 Roth", "P2 Roth",
        "Brokerage Account", "Gross Trad Draw", "Taxes Paid"
    ])

    df['Total_Bal'] = df['P1 Trad 401(k)'] + df['P2 Trad 401(k)'] + df['P1 Roth'] + df['P2 Roth'] + df[
        'Brokerage Account']
    end_of_life_balance = df[df['Age'] == target_lifespan]['Total_Bal'].iloc[0] if not df.empty else 0

    st.markdown("---")
    st.header("5. Lifetime Summary & KPIs")

    # --- Monthly Contributions Section ---
    st.markdown("### Current Monthly Contributions")
    cont_col1, cont_col2, cont_col3 = st.columns(3)

    p1_emp_mo = (p1_salary * p1_401k_cont) / 12
    p1_roth_mo = p1_roth_cont / 12
    p2_emp_mo = (p2_salary * p2_401k_cont) / 12
    p2_roth_mo = p2_roth_cont / 12

    p1_match_mo = (p1_salary * p1_401k_match + p1_401k_flat) / 12
    p2_match_mo = (p2_salary * p2_401k_match + p2_401k_flat) / 12

    total_employee = p1_emp_mo + p1_roth_mo + p2_emp_mo + p2_roth_mo
    total_employer = p1_match_mo + p2_match_mo
    total_saved = total_employee + total_employer

    with cont_col1:
        st.markdown("**Person 1 Monthly Savings:**")
        st.markdown(f"- Employee 401(k): ${p1_emp_mo:,.0f}")
        st.markdown(f"- Employer 401(k): ${p1_match_mo:,.0f}")
        st.markdown(f"- Roth IRA: ${p1_roth_mo:,.0f}")
    with cont_col2:
        st.markdown("**Person 2 Monthly Savings:**")
        st.markdown(f"- Employee 401(k): ${p2_emp_mo:,.0f}")
        st.markdown(f"- Employer 401(k): ${p2_match_mo:,.0f}")
        st.markdown(f"- Roth IRA: ${p2_roth_mo:,.0f}")
    with cont_col3:
        st.markdown("**Total Household Monthly:**")
        st.markdown(f"- **Total Employee (Out of Pocket):** ${total_employee:,.0f}")
        st.markdown(f"- **Total Employer (Company Match):** ${total_employer:,.0f}")
        st.metric("Total Monthly Saved", f"${total_saved:,.0f}", help="Sum of all employee and employer contributions.")

    st.markdown("---")
    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.metric("Age at Portfolio Depletion", str(depletion_age),
                  help="The exact age at which all pre-tax, post-tax, and brokerage balances hit $0.")
        st.metric("Total Early Penalties Paid", f"${tot_penalties:,.0f}",
                  help="Total 10% IRS penalties incurred by forced pre-tax withdrawals before penalty age.")
        st.metric("Portfolio Balance at Retirement", f"${portfolio_at_retire:,.0f}",
                  help="Total combined household portfolio value precisely at the selected retirement age.")
        st.metric(f"Portfolio Balance at Age {target_lifespan}", f"${end_of_life_balance:,.0f}",
                  help="Total combined household portfolio value at your target lifespan age.")

    with kpi2:
        st.metric("Total Tax-Free Roth Conversions", f"${tot_roth_conv_net:,.0f}",
                  help="Net amount of surplus Traditional withdrawals reinvested into Roth IRAs.")
        st.metric("Total RMD Overflows to Brokerage", f"${tot_rmd_overflow_net:,.0f}",
                  help="Net amount of excess mandatory RMD withdrawals swept into the taxable brokerage.")
        st.metric("Taxes: Living Expenses (Income)", f"${tot_tax_living:,.0f}",
                  help="Standard income taxes paid directly to fund baseline living expenses.")

    with kpi3:
        st.metric("Taxes: Roth Conversion", f"${tot_tax_roth:,.0f}",
                  help="Taxes incurred specifically to convert surplus pre-tax money into Roth.")
        st.metric("Taxes: RMD Overflows", f"${tot_tax_rmd:,.0f}",
                  help="Taxes incurred from forced IRS RMDs that exceeded living expense needs.")
        st.metric("Taxes: Brokerage (Cap Gains)", f"${tot_tax_brokerage:,.0f}",
                  help="Capital gains taxes paid when withdrawing the growth portion of the Brokerage account.")

    st.markdown("### Optimization Solvers")
    kpi4, kpi5 = st.columns(2)
    with kpi4:
        st.metric("Max Allowable Pre-Tax Ret. Spend (Die at Zero)", f"${max_income_zero:,.0f}",
                  help="The optimal starting PRE-TAX gross spend that perfectly depletes the portfolio precisely at your target lifespan.")
    with kpi5:
        st.metric("Stable Portfolio Pre-Tax Spend (Preserve Principal)", f"${max_income_stable:,.0f}",
                  help="The optimal starting PRE-TAX gross spend that leaves you with exactly the same portfolio balance at death as you had on the day you retired.")

    st.markdown("---")
    st.subheader("Account Balances Over Time")

    chart_data = df[["Age", "P1 Trad 401(k)", "P2 Trad 401(k)", "P1 Roth", "P2 Roth", "Brokerage Account"]]
    fig = px.area(
        chart_data, x="Age", y=["P1 Trad 401(k)", "P2 Trad 401(k)", "P1 Roth", "P2 Roth", "Brokerage Account"],
        labels={"value": "Account Balance ($)", "variable": "Account Type"}
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("6. Lifetime Projection & Automated Waterfall")
    st.dataframe(df.drop(columns=['Total_Bal']), use_container_width=True)

# ==========================================
# MANUAL & EXPLAINER TAB
# ==========================================
with tab2:
    st.title("📖 User Manual & Architecture Explainer")
    st.markdown("""
    Welcome to the Auto-Optimized Retirement Engine. This tool goes beyond basic compound interest calculators by accurately simulating the U.S. tax code, IRS withdrawal rules, and realistic retirement spending habits.

    ---

    ### 🏦 The Three Accounts (The Tax Buckets)
    The engine categorizes your money into three distinct tax buckets. How money moves between these buckets dictates your tax efficiency.
    1. **Traditional 401(k) / IRA (Pre-Tax):** Money goes in before taxes. It grows tax-deferred. When you withdraw it, it is taxed as standard income. If you withdraw before penalty age (usually 59.5), you pay an additional 10% IRS penalty.
    2. **Roth IRA / 401(k) (Post-Tax):** Money goes in after you've paid taxes. It grows tax-free, and you pay **$0** in taxes when you withdraw it in retirement.
    3. **Brokerage Account (Taxable):** A standard investment account. Money goes in after taxes (establishing a "cost basis"). It grows tax-free, but when you withdraw money, you pay **Capital Gains Tax** only on the *growth* portion.

    ---

    ### 🌊 The Withdrawal Waterfall (How you get paid)
    When you retire, the engine automatically figures out which account to pull from to fund your lifestyle while minimizing taxes. It operates in three distinct phases:

    #### Phase 1: The Penalty Shield (Early Retirement)
    *Applies from your chosen Retirement Age until the Early Withdrawal Penalty Age (e.g., Age 55 to 59).*
    * **Goal:** Avoid the 10% IRS penalty at all costs. 
    * **Action:** The engine will pull $0 from your Traditional 401(k). It will fully fund your lifestyle by draining your **Roth IRA**, then your **Brokerage Account**. It will only touch the 401(k) if you run completely out of other money, at which point it automatically applies the 10% penalty.

    #### Phase 2: Levelized Drawdown & Roth Conversions
    *Applies from Penalty Age until RMD Age (e.g., Age 60 to 74).*
    * **Goal:** Smooth out your taxes so you don't face massive tax bombs later in life.
    * **Action:** The engine calculates a "level" withdrawal amount from your Traditional 401(k) to last the rest of your life. It pays the income tax, and gives you your requested spending money. 
    * **The Secret Sauce:** If the levelized withdrawal results in *more* money than you needed to live on, the engine automatically takes that surplus and **converts it into your Roth IRA**. This aggressively builds tax-free wealth during your low-income retirement years.

    #### Phase 3: Statutory RMDs & Spillovers
    *Applies from RMD Age onward (e.g., Age 75+).*
    * **Goal:** Comply with IRS Required Minimum Distributions (RMDs).
    * **Action:** The IRS forces you to withdraw a percentage of your Traditional 401(k) every year. The engine uses the exact **2024 IRS Uniform Lifetime Table** to calculate this. 
    * **The Spillover:** You cannot put RMD money into a Roth IRA. So, if the IRS forces you to pull out more money than you need to live on, the engine pays the taxes and sweeps the surplus into your **Brokerage Account**, where it establishes a cost basis and continues to grow.

    ---

    ### ⚙️ Algorithm Toggles

    **1. Use Glide Path for Returns**
    * **What it is:** Real retirement funds (like Target Date Funds) don't maintain the same risk profile forever. They transition from aggressive stocks to safe bonds over time.
    * **How it works:** If checked, the engine takes your Pre-Retirement Return (e.g., 7%) and starts smoothly lowering it 25 years before retirement. It crosses the middle point exactly when you retire, and lands safely at your Post-Retirement Return (e.g., 3.5%) exactly 7 years *after* retirement.

    **2. Use Retirement Spending Smile**
    * **What it is:** Research shows retirees do not spend the same amount every year. Spending follows a "Smile" curve: High in early retirement (traveling/hobbies), drops low in middle retirement (slowing down), and rises slightly at the end (healthcare).
    * **How it works:** If checked, your "Desired Retirement Income" is treated as your starting *Go-Go* phase spending. The engine automatically reduces your spending smoothly down to 85% by age 75, bottoms out at 80% by age 85, and then raises it slightly for end-of-life care. If unchecked, it assumes you spend the exact same dollar amount every single year until death.

    ---

    ### 🎯 Optimization Solvers
    At the bottom of the dashboard, you will find two "Solvers". These use complex root-finding algorithms (Brent's Method) to reverse-engineer your perfect retirement number:
    * **Max Allowable Spend (Die at Zero):** The exact gross income you can request such that your last dollar runs out precisely on the day you reach your Target Lifespan Age.
    * **Stable Portfolio (Preserve Principal):** The exact gross income you can pull where your portfolio growth perfectly offsets your spending and taxes. You will die with the exact same net worth you had on the day you retired.
    """)

# ==========================================
# FINANCIAL STORY TAB
# ==========================================
with tab3:
    st.title("📝 The Financial Story")
    st.markdown("Here is the narrative of your simulated retirement, broken down by distinct life phases.")

    # 1. Accumulation Phase
    st.header(f"Phase 1: Accumulation (Age {current_age} to {retire_age})")
    if current_age < retire_age:
        st.markdown(
            f"During this period, you continued working and saving. Your combined household pre-tax balances grew, "
            f"and your total portfolio reached **${portfolio_at_retire:,.0f}** upon retirement at age {retire_age}.")
    else:
        st.markdown("You are already at or past your retirement age. The accumulation phase is complete.")

    # 2. Penalty Shield Phase
    if retire_age < penalty_age:
        st.header(f"Phase 2: Penalty Shield (Age {retire_age} to {penalty_age - 1})")
        if tot_penalties > 0:
            st.markdown(
                f"Because you retired early at age {retire_age}, the engine attempted to fund your lifestyle using your Roth and Brokerage accounts to avoid the 10% IRS penalty on Traditional 401(k) withdrawals. However, those funds were not enough to bridge the gap, and you were forced to incur **${tot_penalties:,.0f}** in early withdrawal penalties.")
        else:
            st.markdown(
                f"Because you retired early at age {retire_age}, the engine successfully funded your lifestyle using your Roth and Brokerage accounts. You successfully avoided all 10% early withdrawal penalties during this period!")

    # 3. Drawdown & Conversions
    drawdown_start = max(retire_age, penalty_age)
    if drawdown_start < rmd_start_age:
        st.header(f"Phase 3: Drawdown & Conversions (Age {drawdown_start} to {rmd_start_age - 1})")
        if tot_roth_conv_net > 0:
            st.markdown(
                f"During this window, you had penalty-free access to your Traditional pre-tax funds. The engine calculated a smooth, levelized withdrawal to last the rest of your life. Because this optimal withdrawal generated *more* cash than your actual living expenses required, the engine successfully captured the surplus and converted **${tot_roth_conv_net:,.0f}** into your Roth IRA, filling up lower tax brackets and building tax-free wealth.")
        else:
            st.markdown(
                f"During this window, you had penalty-free access to your Traditional pre-tax funds. The engine calculated a withdrawal to fund your living expenses. Based on your spending needs and account balances, no surplus was available to convert to your Roth IRA.")

    # 4. Statutory RMDs
    st.header(f"Phase 4: Statutory RMDs (Age {rmd_start_age}+)")
    if tot_rmd_overflow_net > 0:
        st.markdown(
            f"Starting at age {rmd_start_age}, the IRS mandated Required Minimum Distributions (RMDs). Because these forced withdrawals exceeded your lifestyle needs, and because RMDs legally cannot be deposited into a Roth IRA, the engine paid the taxes and swept **${tot_rmd_overflow_net:,.0f}** of surplus cash into your Taxable Brokerage Account where it established a new cost basis and continued to grow.")
    else:
        st.markdown(
            f"Starting at age {rmd_start_age}, the IRS mandated Required Minimum Distributions (RMDs). Your lifestyle needs were large enough that all RMDs were consumed by living expenses, resulting in no surplus overflow to your Brokerage account.")

    # 5. End of Life
    st.header("The Conclusion")
    if str(depletion_age) != "Never" and int(depletion_age) <= target_lifespan:
        st.error(
            f"Unfortunately, your portfolio depleted entirely at age **{depletion_age}**. You ran out of money before reaching your target lifespan of {target_lifespan}.")
    else:
        st.success(
            f"Congratulations! You successfully reached your target lifespan of {target_lifespan} with a remaining portfolio balance of **${end_of_life_balance:,.0f}**.")