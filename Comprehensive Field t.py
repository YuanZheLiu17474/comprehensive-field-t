"""
Comprehensive Field t: A Cross-Scale Phenomenological Framework
from Quark Masses to Neutron Star Structure

Single-file reproducible code.

Author: Yuanzhe Liu
Repository: https://github.com/YuanZheLiu17474/comprehensive-field-t

Run: python Comprehensive_Field_t.py

Output:
  - Console: full calculation results
  - Files: output_results.txt, data_output.csv, 6 figures (png + pdf)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, least_squares

# ============================================================
# Section 1: Constants
# ============================================================
M_p = 938.27208816
M_n = 939.56542052
u_to_MeV = 931.49410242
m_e = 0.51099895
t_unit = M_p
km_per_Msun = 1.477
MeV_fm3_to_km2 = 1.32e-6
M_Pl = 1.22e19
g_star = 100.0
x_f = 20.0
Omega_DM_h2 = 0.120

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# Section 2: Nuclear Mass Data
# ============================================================
NUCLEI_TRAINING = [
    ("D",2,1,2.0141017781),("T",3,1,3.0160492813),("He3",3,2,3.0160293220),
    ("He4",4,2,4.0026032541),("Li6",6,3,6.0151228874),("Li7",7,3,7.0160034366),
    ("Be9",9,4,9.0121830658),("B10",10,5,10.0129369985),("B11",11,5,11.0093051665),
    ("C12",12,6,12.0000000000),("C13",13,6,13.0033548351),("N14",14,7,14.0030740044),
    ("N15",15,7,15.0001088989),("O16",16,8,15.9949146196),("O17",17,8,16.9991317565),
    ("O18",18,8,17.9991596129),("F19",19,9,18.9984031627),("Ne20",20,10,19.9924401762),
    ("Ne22",22,10,21.9913851140),("Na23",23,11,22.9897692820),("Mg24",24,12,23.9850416971),
    ("Mg26",26,12,25.9825929670),("Al27",27,13,26.9815385300),("Si28",28,14,27.9769265347),
    ("Si30",30,14,29.9737701710),("P31",31,15,30.9737619984),("S32",32,16,31.9720711744),
    ("S34",34,16,33.9678669000),("Cl35",35,17,34.9688526820),("Ar36",36,18,35.9675451060),
    ("Ar40",40,18,39.9623831237),("K39",39,19,38.9637064864),("Ca40",40,20,39.9625908630),
    ("Ca48",48,20,47.9525229000),("Ti48",48,22,47.9479419800),("Cr52",52,24,51.9405062300),
    ("Mn55",55,25,54.9380439100),("Fe54",54,26,53.9396089900),("Fe56",56,26,55.9349363300),
    ("Fe57",57,26,56.9353928400),("Co59",59,27,58.9331942900),("Ni58",58,28,57.9353424100),
    ("Ni60",60,28,59.9307858800),("Ni62",62,28,61.9283448800),("Cu63",63,29,62.9295977200),
    ("Zn64",64,30,63.9291420100),("Zn66",66,30,65.9260338100),("Ga69",69,31,68.9255735000),
    ("Ge74",74,32,73.9211777600),("As75",75,33,74.9215945700),("Se80",80,34,79.9165218000),
    ("Br79",79,35,78.9183376000),("Kr84",84,36,83.9114977300),("Rb85",85,37,84.9117897379),
    ("Sr88",88,38,87.9056125000),("Y89",89,39,88.9058403000),("Zr90",90,40,89.9046977000),
    ("Mo98",98,42,97.9054069000),
]

NUCLEI_BLIND = [
    ("Ag107",107,47,106.9050916000),("Sn120",120,50,119.9022016300),
    ("Xe132",132,54,131.9041550860),("Ba138",138,56,137.9052470000),
    ("Ce140",140,58,139.9054380000),("Nd150",150,60,149.9209000000),
    ("Sm152",152,62,151.9197300000),("Pb208",208,82,207.9766525000),
    ("U238",238,92,238.0507884000),
]

# ============================================================
# Section 3: Nuclear Mass Model
# ============================================================
def nuclear_mass(A, Z, mu):
    return mu * u_to_MeV - Z * m_e * 1e-3 * u_to_MeV

def liquid_drop_dt(A, Z, a, b, c, d, de):
    N = A - Z
    pairing = np.where((Z%2==0)&(N%2==0), de,
              np.where((Z%2!=0)&(N%2!=0), -de, 0.0))
    return a*A - b*A**(2/3) - c*Z*(Z-1)/A**(1/3) - d*(N-Z)**2/A + pairing

def mass_model(A, Z, params):
    a, b, c, d, de = params
    dt = liquid_drop_dt(A, Z, a, b, c, d, de)
    return (Z*(M_p/t_unit) + (A-Z)*(M_n/t_unit) - dt) * t_unit

# ============================================================
# Section 4: Neutron Star
# ============================================================
def EOS_pheno(n_B, n_0=0.16, K_0=240.0, M_N=939.0, n_soft=0.30):
    if n_B <= n_soft:
        K_eff = K_0
    else:
        x = (n_B - n_soft) / n_soft
        K_eff = K_0 * (1 - 0.5 * x / (1 + x))
    x = n_B / n_0
    E_A = -16 + (K_eff/18) * (x - 1)**2
    eps = (M_N + E_A) * n_B
    P = n_B**2 * (K_eff / (9 * n_0)) * (x - 1)
    return eps, P

def eps_of_P(P):
    if P <= 0:
        return 0.0
    try:
        n_B = brentq(lambda n: EOS_pheno(n)[1] - P, 0.001, 3.0)
        return EOS_pheno(n_B)[0]
    except ValueError:
        return 4000.0

def tov_rhs(r, y):
    m, P = y
    if P <= 0 or r <= 0:
        return [0.0, 0.0]
    eps = eps_of_P(P)
    if eps <= 0:
        return [0.0, 0.0]
    eps_g = eps * MeV_fm3_to_km2
    P_g = P * MeV_fm3_to_km2
    rs = 2.0 * m
    if r <= rs * 1.001:
        return [0.0, 0.0]
    factor = 1.0 - rs / r
    dm = 4 * np.pi * r**2 * eps_g
    dP_g = -(eps_g + P_g) * (m + 4*np.pi*r**3*P_g) / (r**2 * factor)
    return [dm, dP_g / MeV_fm3_to_km2]

def solve_star(eps_c):
    try:
        n_c = brentq(lambda n: EOS_pheno(n)[0] - eps_c, 0.001, 3.0)
    except ValueError:
        return None, None
    _, P_c = EOS_pheno(n_c)
    if P_c <= 0:
        return None, None

    def event_P_zero(r, y):
        return y[1]
    event_P_zero.terminal = True
    event_P_zero.direction = -1

    sol = solve_ivp(tov_rhs, (1e-3, 25.0), [0.0, P_c],
                    events=[event_P_zero], rtol=1e-8, atol=1e-10, max_step=0.02)
    if len(sol.t) == 0:
        return None, None
    return sol.y[0, -1] / km_per_Msun, sol.t[-1]

# ============================================================
# Section 5: Electroweak Sector
# ============================================================
def V_CW(t, mu=246.0, kappa=0.13, g_N=0.1):
    if t <= 0:
        return 0.0
    M2_Phi = kappa * t**2
    V_Phi = (4 / (64 * np.pi**2)) * M2_Phi**2 * (np.log(M2_Phi / mu**2) - 1.5)
    M2_N = g_N**2 * t**2
    V_N = (-12 / (64 * np.pi**2)) * M2_N**2 * (np.log(M2_N / mu**2) - 1.5)
    return V_Phi + V_N

def V_tree(t, lambda_t=1.0, v_t=246.0):
    return lambda_t / 4 * (t**2 - v_t**2)**2

def V_eff(t, mu=246.0, lambda_t=1.0, v_t=246.0, kappa=0.13, g_N=0.1):
    return V_tree(t, lambda_t, v_t) + V_CW(t, mu, kappa, g_N)

def beta_system(ln_mu, y, N_c=3, N_f=3, lambda_H=0.13, g_s=1.22, g=0.653, gp=0.358):
    lam_t, kappa, g_N = y
    b0 = 11 - 2 * N_f / 3
    gs2 = g_s**2 / (1 + b0 * g_s**2 / (16 * np.pi**2) * (ln_mu - np.log(246.0)))
    gs2 = max(gs2, 0.01)
    bl = (3*lam_t**2 + 2*kappa**2 - 8*N_c*g_N**4) / (16*np.pi**2)
    bk = kappa * (6*lam_t + 6*lambda_H + 4*kappa - 4*N_c*g_N**2) / (16*np.pi**2)
    bg = g_N * (4.5*g_N**2 - 8*gs2 - 2.25*g**2 - 1.25*gp**2) / (16*np.pi**2)
    return [bl, bk, bg]

def run_rg(lam0=0.13, kap0=0.13, gN0=0.1):
    return solve_ivp(beta_system, (np.log(246.0), np.log(1e18)),
                     [lam0, kap0, gN0], dense_output=True, rtol=1e-10, atol=1e-12)

# ============================================================
# Section 6: Dark Matter
# ============================================================
def kappa_higgs_portal(Lambda_d, y_chi=0.01, v_EW=246.0):
    return y_chi * Lambda_d / v_EW**2

def sigma_SI(m_t, kappa, m_h=125.25, f_N=0.3, m_N=0.939):
    lambda_hs = 2 * kappa
    numerator = lambda_hs**2 * f_N**2 * m_N**4
    denominator = 4 * np.pi * m_h**4 * m_t**2 * (m_t + m_N)**2
    return numerator / denominator * 3.894e-28

# ============================================================
# Section 7: Figure Generation
# ============================================================
def make_all_figures(popt, output_dir):
    plt.rcParams['font.family'] = 'serif'

    # --- Figure 1: Binding energy curve ---
    A_range = np.arange(2, 240)
    E_per_A = []
    for A in A_range:
        Z = int(round(A / 2.05))
        N = A - Z
        if Z < 1 or N < 1:
            continue
        E_per_A.append(-liquid_drop_dt(A, Z, *popt) * M_p / A)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(A_range[:len(E_per_A)], E_per_A, 'b-', lw=2, label='Comprehensive field model')
    ax.axvline(56, color='r', ls='--', alpha=0.7, label='Fe-56')
    ax.axhline(-8.8, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel('Mass number A')
    ax.set_ylabel('Binding energy per nucleon E/A (MeV)')
    ax.set_title('Figure 1: Nuclear binding energy curve')
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig1_binding_energy.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig1_binding_energy.pdf'))
    plt.close()

    # --- Figure 2: Mass-radius ---
    M_list, R_list = [], []
    for eps_c in np.linspace(200, 1500, 40):
        M, R = solve_star(eps_c)
        if M and 0.5 < M < 3.0 and 5 < R < 20:
            M_list.append(M); R_list.append(R)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(R_list, M_list, 'b-o', ms=4, lw=1.5, label='Comprehensive field EOS')
    ax.axhline(2.1, color='r', ls='--', alpha=0.7, label='Observed maximum 2.1 M_sun')
    ax.axvspan(11, 13, alpha=0.15, color='green', label='NICER constraint R_1.4')
    ax.set_xlabel('Radius R (km)')
    ax.set_ylabel('Mass M (M_sun)')
    ax.set_title('Figure 2: Neutron star mass-radius relation')
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig2_mass_radius.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig2_mass_radius.pdf'))
    plt.close()

    # --- Figure 3: Tidal deformability ---
    M_list, Lam_list = [], []
    for eps_c in np.linspace(200, 1500, 40):
        M, R = solve_star(eps_c)
        if M and 0.5 < M < 3.0 and 5 < R < 20:
            C = M * km_per_Msun / R
            Lam_list.append(0.05 / C**5); M_list.append(M)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(M_list, Lam_list, 'g-o', ms=4, lw=1.5, label='Model')
    ax.axhspan(70, 800, alpha=0.15, color='blue', label='GW170817 constraint')
    ax.set_xlabel('Mass M (M_sun)')
    ax.set_ylabel('Tidal deformability Lambda')
    ax.set_title('Figure 3: Tidal deformability')
    ax.legend(); ax.grid(alpha=0.3, which='both'); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig3_tidal.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig3_tidal.pdf'))
    plt.close()

    # --- Figure 4: RG running ---
    sol = run_rg()
    mu = np.exp(sol.t)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(mu, sol.y[0], 'b-', lw=2, label='lambda_t')
    ax.semilogx(mu, sol.y[1], 'r-', lw=2, label='kappa')
    ax.semilogx(mu, sol.y[2], 'g-', lw=2, label='g_N')
    ax.set_xlabel('Energy scale mu (GeV)')
    ax.set_ylabel('Coupling constant')
    ax.set_title('Figure 4: Renormalization group running')
    ax.legend(); ax.grid(alpha=0.3, which='both'); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig4_rg_running.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig4_rg_running.pdf'))
    plt.close()

    # --- Figure 5: Dark matter parameter space ---
    def relic(m_t, k):
        sv = k**4 / (16 * np.pi * m_t**2)
        return 1.07e9 / M_Pl * x_f / np.sqrt(g_star) / sv

    m_t_range = np.linspace(100, 3000, 50)
    kappa_curve = []
    for m_t in m_t_range:
        try:
            k = brentq(lambda k: relic(m_t, k) - Omega_DM_h2, 0.001, 3.0)
            kappa_curve.append(k)
        except ValueError:
            kappa_curve.append(np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(m_t_range, kappa_curve, 'purple', lw=2, label='Omega h^2 = 0.12')
    ax.fill_between(m_t_range, kappa_curve, 3.0, alpha=0.15, color='red', label='Omega h^2 < 0.12')
    ax.fill_between(m_t_range, 0, kappa_curve, alpha=0.15, color='blue', label='Omega h^2 > 0.12')
    ax.set_xlabel('Dark matter mass m_t (GeV)')
    ax.set_ylabel('Coupling kappa')
    ax.set_title('Figure 5: Dark matter parameter space')
    ax.set_xlim(100, 3000); ax.set_ylim(0, 1.0)
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig5_dark_matter.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig5_dark_matter.pdf'))
    plt.close()

    # --- Figure 6: Error vs A ---
    train_A = np.array([n[1] for n in NUCLEI_TRAINING])
    train_err = np.array([mass_model(n[1], n[2], popt) - nuclear_mass(n[1], n[2], n[3])
                          for n in NUCLEI_TRAINING])
    test_A = np.array([n[1] for n in NUCLEI_BLIND])
    test_err = np.array([mass_model(n[1], n[2], popt) - nuclear_mass(n[1], n[2], n[3])
                         for n in NUCLEI_BLIND])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(train_A, train_err, c='blue', alpha=0.6, s=30, label='Training (A<=100)')
    ax.scatter(test_A, test_err, c='red', alpha=0.8, s=50, marker='s', label='Blind test (A>100)')
    ax.axhline(0, color='k', lw=0.5)
    ax.axvline(100, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('Mass number A')
    ax.set_ylabel('Mass error (MeV)')
    ax.set_title('Figure 6: Model error vs mass number')
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fig6_errors.png'), dpi=300)
    plt.savefig(os.path.join(output_dir, 'fig6_errors.pdf'))
    plt.close()

    print("  [Figures] 6 figures saved (png + pdf)")


# ============================================================
# Section 8: Main Execution
# ============================================================
def main():
    print("=" * 70)
    print("Comprehensive Field t: Full Reproducible Calculation")
    print("=" * 70)

    # ---------- Step 1: Nuclear mass fit ----------
    print("\n[Step 1] Fitting nuclear masses (A <= 100)...")
    A_tr = np.array([n[1] for n in NUCLEI_TRAINING], dtype=float)
    Z_tr = np.array([n[2] for n in NUCLEI_TRAINING], dtype=float)
    M_tr = np.array([nuclear_mass(n[1], n[2], n[3]) for n in NUCLEI_TRAINING])

    def residuals(p):
        return mass_model(A_tr, Z_tr, p) - M_tr

    result = least_squares(
        residuals,
        x0=[0.0153, 0.0158, 0.0006, 0.0216, 0.0030],
        bounds=([0.005, 0.005, 0.0001, 0.005, 0.0001],
                [0.050, 0.050, 0.0050, 0.100, 0.0100])
    )
    popt = result.x

    print(f"  Fitted parameters:")
    print(f"    a     = {popt[0]:.6f}")
    print(f"    b     = {popt[1]:.6f}")
    print(f"    c     = {popt[2]:.6f}")
    print(f"    d     = {popt[3]:.6f}")
    print(f"    delta = {popt[4]:.6f}")

    # Errors
    train_errs = np.array([mass_model(n[1], n[2], popt) - nuclear_mass(n[1], n[2], n[3])
                           for n in NUCLEI_TRAINING])
    test_errs = np.array([mass_model(n[1], n[2], popt) - nuclear_mass(n[1], n[2], n[3])
                          for n in NUCLEI_BLIND])
    train_rms = np.sqrt(np.mean(train_errs**2))
    test_rms = np.sqrt(np.mean(test_errs**2))
    print(f"  Training RMS error (A<=100): {train_rms:.3f} MeV")
    print(f"  Blind RMS error (A>100):     {test_rms:.3f} MeV")

    # ---------- Step 2: Neutron star ----------
    print("\n[Step 2] Solving neutron star TOV equations...")
    M_max = 0.0
    R_at_Mmax = 0.0
    R_14 = 0.0
    for eps_c in np.linspace(200, 1500, 40):
        M, R = solve_star(eps_c)
        if M and 0.5 < M < 3.0 and 5 < R < 20:
            if M > M_max:
                M_max = M
                R_at_Mmax = R
            if abs(M - 1.4) < 0.05 and R_14 == 0.0:
                R_14 = R

    print(f"  Maximum mass:  M_max = {M_max:.3f} M_sun")
    print(f"  Radius at M_max: R = {R_at_Mmax:.2f} km")
    print(f"  Radius at 1.4 M_sun: R_1.4 = {R_14:.2f} km")

    # ---------- Step 3: Electroweak ----------
    print("\n[Step 3] Computing electroweak effective potential...")
    t_scan = np.linspace(100, 500, 200)
    V_scan = np.array([V_eff(t) for t in t_scan])
    t_min = t_scan[np.argmin(V_scan)]
    print(f"  Vacuum expectation value: t_min = {t_min:.2f} GeV")
    print(f"  (Experimental: v_EW = 246 GeV)")

    # ---------- Step 4: Dark matter ----------
    print("\n[Step 4] Computing dark matter observables...")
    Lambda_d = 250.0
    m_t = 4 * Lambda_d
    kappa = kappa_higgs_portal(Lambda_d)
    sigma = sigma_SI(m_t, kappa)
    print(f"  Composite t-particle mass: m_t = {m_t} GeV")
    print(f"  t-Higgs coupling: kappa = {kappa:.3e}")
    print(f"  Direct detection cross section: sigma_SI = {sigma:.3e} cm^2")

    # ---------- Step 5: Save results ----------
    print("\n[Step 5] Saving results to files...")
    with open(os.path.join(OUTPUT_DIR, 'output_results.txt'), 'w') as f:
        f.write("Comprehensive Field t: Calculation Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("[Nuclear Mass Fit]\n")
        f.write(f"  a     = {popt[0]:.6f}\n")
        f.write(f"  b     = {popt[1]:.6f}\n")
        f.write(f"  c     = {popt[2]:.6f}\n")
        f.write(f"  d     = {popt[3]:.6f}\n")
        f.write(f"  delta = {popt[4]:.6f}\n")
        f.write(f"  Training RMS: {train_rms:.3f} MeV\n")
        f.write(f"  Blind RMS:    {test_rms:.3f} MeV\n\n")
        f.write("[Neutron Star]\n")
        f.write(f"  M_max = {M_max:.3f} M_sun\n")
        f.write(f"  R_at_Mmax = {R_at_Mmax:.2f} km\n")
        f.write(f"  R_1.4 = {R_14:.2f} km\n\n")
        f.write("[Electroweak]\n")
        f.write(f"  t_min = {t_min:.2f} GeV\n\n")
        f.write("[Dark Matter]\n")
        f.write(f"  m_t = {m_t} GeV\n")
        f.write(f"  kappa = {kappa:.3e}\n")
        f.write(f"  sigma_SI = {sigma:.3e} cm^2\n")
    print(f"  Results saved: output_results.txt")

    # Save data as CSV
    with open(os.path.join(OUTPUT_DIR, 'data_output.csv'), 'w') as f:
        f.write("name,A,Z,atomic_mass_u,dataset,predicted_mass_MeV,error_MeV\n")
        for name, A, Z, mu in NUCLEI_TRAINING:
            Mt = nuclear_mass(A, Z, mu)
            Mp = mass_model(A, Z, popt)
            f.write(f"{name},{A},{Z},{mu},training,{Mp:.4f},{Mp-Mt:.4f}\n")
        for name, A, Z, mu in NUCLEI_BLIND:
            Mt = nuclear_mass(A, Z, mu)
            Mp = mass_model(A, Z, popt)
            f.write(f"{name},{A},{Z},{mu},blind,{Mp:.4f},{Mp-Mt:.4f}\n")
    print(f"  Data saved: data_output.csv")

    # ---------- Step 6: Generate figures ----------
    print("\n[Step 6] Generating figures...")
    make_all_figures(popt, OUTPUT_DIR)

    # ---------- Summary ----------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Nuclear fit parameters: {popt}")
    print(f"  Neutron star maximum mass: {M_max:.3f} M_sun")
    print(f"  Dark matter: m_t = {m_t} GeV, sigma_SI = {sigma:.3e} cm^2")
    print(f"  All 6 figures saved.")
    print("=" * 70)


if __name__ == "__main__":
    main()