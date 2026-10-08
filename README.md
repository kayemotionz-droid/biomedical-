"""
Biomedical simulation: factors influencing drug exposure and effect (PK/PD)
===========================================================================
A one-compartment oral pharmacokinetic (PK) model with an Emax
pharmacodynamic (PD) effect. ILLUSTRATIVE ONLY - parameters are generic and
not for clinical use.

Factors modelled
----------------
Patient:    body weight (allometric scaling), age, sex, serum creatinine
            (-> kidney function via Cockcroft-Gault), liver function
Treatment:  dose, dosing interval, adherence (missed doses)
Context:    food effect on absorption, CYP enzyme inhibitor / inducer
Population: random inter-individual variability (Monte Carlo)

Analyses
--------
1. Scenario comparison  - concentration-time curves under different factors
2. Sensitivity analysis - one-at-a-time effect on exposure (AUC), tornado plot
3. Population simulation - 500 virtual patients, percentile band, outcomes

Run:  python biomedical_pkpd_simulation.py
Needs: numpy, scipy, matplotlib
"""

from dataclasses import dataclass, replace
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

rng = np.random.default_rng(42)


# --------------------------------------------------------------------------
# 1. Parameters
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Drug:
    """Generic oral drug (illustrative values)."""
    cl_nonrenal: float = 3.0     # L/h, hepatic/other clearance at 70 kg
    cl_renal: float = 3.0        # L/h, renal clearance at CrCl = 100 mL/min, 70 kg
    v_pop: float = 50.0          # L, volume of distribution at 70 kg
    ka: float = 1.0              # 1/h, absorption rate
    bioavail: float = 0.9        # fraction absorbed
    emax: float = 100.0          # % max effect
    ec50: float = 2.0            # mg/L
    hill: float = 1.5
    mtc_low: float = 1.0         # mg/L, lower bound of therapeutic window
    mtc_high: float = 8.0        # mg/L, toxicity threshold


@dataclass(frozen=True)
class Patient:
    weight: float = 70.0         # kg
    age: float = 40.0            # years
    female: bool = False
    serum_creatinine: float = 1.0  # mg/dL
    hepatic_function: float = 1.0  # 1 = normal, 0.5 = moderate impairment
    fed: bool = False            # taken with food
    cyp_modifier: float = 1.0    # 0.5 = inhibitor, 2.0 = inducer, 1 = none
    adherence: float = 1.0       # probability each dose is taken


@dataclass(frozen=True)
class Regimen:
    dose_mg: float = 200.0
    interval_h: float = 12.0
    n_doses: int = 14            # 7 days of twice-daily dosing


def creatinine_clearance(p: Patient) -> float:
    """Cockcroft-Gault estimate (mL/min)."""
    crcl = (140 - p.age) * p.weight / (72 * p.serum_creatinine)
    return crcl * (0.85 if p.female else 1.0)


def individual_params(drug: Drug, p: Patient):
    """Map patient factors to individual CL, V, ka, F."""
    scale = p.weight / 70.0
    cl_nr = drug.cl_nonrenal * scale**0.75 * p.hepatic_function * p.cyp_modifier
    cl_r = drug.cl_renal * scale**0.75 * (creatinine_clearance(p) / 100.0)
    cl = cl_nr + cl_r
    v = drug.v_pop * scale
    ka = drug.ka * (0.5 if p.fed else 1.0)       # food slows absorption
    f = drug.bioavail
    return cl, v, ka, f


# --------------------------------------------------------------------------
# 2. Model
# --------------------------------------------------------------------------
def simulate(drug: Drug, p: Patient, reg: Regimen, dt: float = 0.1,
             take_doses=None, tail_h: float = 24.0):
    """Simulate gut/central amounts with discrete dosing events.

    Returns t (h), conc (mg/L), effect (%), taken (bool array of doses).
    """
    cl, v, ka, f = individual_params(drug, p)
    ke = cl / v

    if take_doses is None:
        take_doses = rng.random(reg.n_doses) < p.adherence
    dose_times = np.arange(reg.n_doses) * reg.interval_h
    t_end = dose_times[-1] + reg.interval_h + tail_h
    t_all = np.arange(0, t_end + dt, dt)

    def rhs(_, y):
        gut, cen = y
        return [-ka * gut, ka * gut * f - ke * cen]

    y = np.array([0.0, 0.0])
    conc = np.zeros_like(t_all)
    boundaries = list(dose_times) + [t_end]
    for i in range(len(dose_times)):
        if take_doses[i]:
            y[0] += reg.dose_mg
        t0, t1 = boundaries[i], boundaries[i + 1]
        mask = (t_all >= t0) & (t_all <= t1)
        sol = solve_ivp(rhs, (t0, t1), y, t_eval=t_all[mask],
                        method="LSODA", rtol=1e-8, atol=1e-10)
        conc[mask] = np.maximum(sol.y[1], 0) / v
        y = sol.y[:, -1]

    effect = drug.emax * conc**drug.hill / (drug.ec50**drug.hill + conc**drug.hill)
    return t_all, conc, effect, take_doses


def metrics(drug: Drug, t, conc, reg: Regimen):
    """Exposure and outcome metrics over the final dosing interval + window stats."""
    last_start = (reg.n_doses - 1) * reg.interval_h
    m = (t >= last_start) & (t <= last_start + reg.interval_h)
    auc = np.trapezoid(conc[m], t[m])
    in_window = (conc >= drug.mtc_low) & (conc <= drug.mtc_high)
    dosing_phase = t <= reg.n_doses * reg.interval_h
    return {
        "Cmax": conc[m].max(),
        "Cmin": conc[m].min(),
        "AUC_tau": auc,
        "pct_in_window": 100 * in_window[dosing_phase].mean(),
        "pct_toxic": 100 * (conc[dosing_phase] > drug.mtc_high).mean(),
        "pct_subtherapeutic": 100 * (conc[dosing_phase] < drug.mtc_low).mean(),
    }


# --------------------------------------------------------------------------
# 3. Analyses
# --------------------------------------------------------------------------
def scenario_comparison(drug, reg):
    base = Patient()
    scenarios = {
        "Baseline (70 kg, healthy)": base,
        "Renal impairment (age 80, SCr 2.0)": replace(base, age=80, serum_creatinine=2.0),
        "Liver impairment (50%)": replace(base, hepatic_function=0.5),
        "CYP inhibitor co-medication": replace(base, cyp_modifier=0.5),
        "CYP inducer co-medication": replace(base, cyp_modifier=2.0),
        "Low weight (45 kg)": replace(base, weight=45),
        "Taken with food": replace(base, fed=True),
        "Poor adherence (60%)": replace(base, adherence=0.6),
    }
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    rows = []
    for name, pat in scenarios.items():
        t, c, e, _ = simulate(drug, pat, reg)
        axes[0].plot(t, c, label=name, lw=1.6)
        axes[1].plot(t, e, lw=1.6)
        rows.append((name, metrics(drug, t, c, reg)))
    for ax in axes[:1]:
        ax.axhspan(drug.mtc_low, drug.mtc_high, color="green", alpha=0.08,
                   label="Therapeutic window")
        ax.axhline(drug.mtc_high, color="red", ls="--", lw=0.8)
    axes[0].set_ylabel("Plasma concentration (mg/L)")
    axes[1].set_ylabel("Effect (% of Emax)")
    axes[1].set_xlabel("Time (h)")
    axes[0].set_title("How patient and treatment factors change drug exposure and effect")
    axes[0].legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("1_scenarios.png", dpi=150)

    print("\n=== Scenario comparison (last dosing interval) ===")
    print(f"{'Scenario':<38}{'Cmax':>7}{'Cmin':>7}{'AUC':>8}{'%window':>9}{'%toxic':>8}")
    for name, m in rows:
        print(f"{name:<38}{m['Cmax']:>7.2f}{m['Cmin']:>7.2f}{m['AUC_tau']:>8.1f}"
              f"{m['pct_in_window']:>9.1f}{m['pct_toxic']:>8.1f}")


def sensitivity_analysis(drug, reg):
    """One-at-a-time: vary each factor between a low and high value."""
    base = Patient()
    t, c, _, _ = simulate(drug, base, reg, take_doses=np.ones(reg.n_doses, bool))
    base_auc = metrics(drug, t, c, reg)["AUC_tau"]

    ranges = {
        "Body weight (45-120 kg)":       [dict(weight=45), dict(weight=120)],
        "Age (20-85 y)":                 [dict(age=20), dict(age=85)],
        "Serum creatinine (0.7-2.5)":    [dict(serum_creatinine=0.7), dict(serum_creatinine=2.5)],
        "Liver function (0.4-1.0)":      [dict(hepatic_function=0.4), dict(hepatic_function=1.0)],
        "CYP modifier (0.5-2.0)":        [dict(cyp_modifier=0.5), dict(cyp_modifier=2.0)],
        "Sex (male-female)":             [dict(female=False), dict(female=True)],
    }
    results = []
    for label, (lo, hi) in ranges.items():
        aucs = []
        for kw in (lo, hi):
            t, c, _, _ = simulate(drug, replace(base, **kw), reg,
                                  take_doses=np.ones(reg.n_doses, bool))
            aucs.append(metrics(drug, t, c, reg)["AUC_tau"])
        results.append((label, aucs[0] / base_auc - 1, aucs[1] / base_auc - 1))
    results.sort(key=lambda r: max(abs(r[1]), abs(r[2])))

    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (label, lo, hi) in enumerate(results):
        ax.barh(i, lo * 100, color="#3b82f6", alpha=0.8)
        ax.barh(i, hi * 100, color="#ef4444", alpha=0.8)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels([r[0] for r in results])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Change in steady-state AUC vs baseline (%)")
    ax.set_title("Sensitivity analysis (blue = low value, red = high value)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig("2_sensitivity.png", dpi=150)

    print("\n=== Sensitivity of AUC to each factor ===")
    for label, lo, hi in reversed(results):
        print(f"{label:<32} low: {lo*100:+7.1f}%   high: {hi*100:+7.1f}%")


def population_simulation(drug, reg, n=500):
    """Monte Carlo: virtual patients with realistic covariate spread + random variability."""
    all_c = []
    in_window, toxic = [], []
    for _ in range(n):
        female = rng.random() < 0.5
        pat = Patient(
            weight=float(np.clip(rng.normal(75 if not female else 65, 14), 40, 140)),
            age=float(np.clip(rng.normal(55, 18), 18, 90)),
            female=female,
            serum_creatinine=float(np.clip(rng.lognormal(np.log(1.0), 0.25), 0.5, 3.0)),
            hepatic_function=float(rng.choice([1.0, 0.6], p=[0.9, 0.1])),
            fed=bool(rng.random() < 0.5),
            cyp_modifier=float(rng.choice([1.0, 0.5, 2.0], p=[0.8, 0.12, 0.08])),
            adherence=float(np.clip(rng.beta(8, 2), 0.3, 1.0)),
        )
        # Random inter-individual variability on top of covariates (30% CV on drug params)
        eta = rng.lognormal(0, 0.3)
        d = replace(drug, cl_nonrenal=drug.cl_nonrenal * eta,
                    cl_renal=drug.cl_renal * eta,
                    v_pop=drug.v_pop * rng.lognormal(0, 0.2))
        t, c, _, _ = simulate(d, pat, reg)
        all_c.append(c)
        m = metrics(d, t, c, reg)
        in_window.append(m["pct_in_window"])
        toxic.append(m["pct_toxic"])

    C = np.array(all_c)
    p5, p50, p95 = np.percentile(C, [5, 50, 95], axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].fill_between(t, p5, p95, color="#3b82f6", alpha=0.25, label="5th-95th percentile")
    axes[0].plot(t, p50, color="#1d4ed8", lw=2, label="Median")
    axes[0].axhspan(drug.mtc_low, drug.mtc_high, color="green", alpha=0.08)
    axes[0].axhline(drug.mtc_high, color="red", ls="--", lw=0.8)
    axes[0].set_xlabel("Time (h)")
    axes[0].set_ylabel("Plasma concentration (mg/L)")
    axes[0].set_title(f"Virtual population (n={n})")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].hist(in_window, bins=25, color="#10b981", alpha=0.85)
    axes[1].set_xlabel("% of treatment time inside therapeutic window")
    axes[1].set_ylabel("Patients")
    axes[1].set_title("Variability in treatment quality")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("3_population.png", dpi=150)

    in_window, toxic = np.array(in_window), np.array(toxic)
    print(f"\n=== Population simulation (n={n}) ===")
    print(f"Median time in window:           {np.median(in_window):.1f}%")
    print(f"Patients in window >70% of time: {100*np.mean(in_window > 70):.1f}%")
    print(f"Patients with any toxic exposure: {100*np.mean(toxic > 0):.1f}%")


if __name__ == "__main__":
    matplotlib.rcParams.update({"font.size": 10})
    drug, reg = Drug(), Regimen()
    scenario_comparison(drug, reg)
    sensitivity_analysis(drug, reg)
    population_simulation(drug, reg)
    print("\nSaved: 1_scenarios.png, 2_sensitivity.png, 3_population.png")
    plt.show()# biomedical-