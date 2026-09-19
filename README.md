# Comprehensive Field t: A Cross-Scale Phenomenological Framework

Code repository for the paper:

**The Comprehensive Field t: A Cross-Scale Phenomenological Framework from Quark Masses to Neutron Star Structure**

Yuanzhe Liu
Yali High School, Changsha, China

---

## Overview

This repository contains the complete, reproducible code for the paper. The framework describes:

- Nuclear binding energies for A = 2 to 100
- Neutron star mass–radius relation
- Electroweak symmetry breaking
- Dark matter relic density

All calculations are performed in a single, self-contained Python script.

---

## Requirements

- Python 3.10 or higher
- numpy >= 1.24
- scipy >= 1.10
- matplotlib >= 3.7

Install dependencies:

```
pip install -r requirements.txt
```

---

## Usage

Run the full calculation:

```
python Comprehensive_Field_t.py
```

The script will automatically:

1. Fit nuclear masses (A ≤ 100) using the liquid drop model
2. Solve neutron star TOV equations with the phenomenological EOS
3. Compute the electroweak Coleman-Weinberg effective potential
4. Compute dark matter observables (composite t-particle, direct detection)
5. Save results to `output_results.txt` and `data_output.csv`
6. Generate 6 figures in both PNG and PDF format

All output files are saved in the same directory as the script, regardless of where it is run from.

---

## Expected Output

The script prints a summary to the console:

```
======================================================================
Comprehensive Field t: Full Reproducible Calculation
======================================================================

[Step 1] Fitting nuclear masses (A <= 100)...
  Fitted parameters:
    a     = 0.015272
    b     = 0.015793
    c     = 0.000580
    d     = 0.021634
    delta = 0.002963
  Training RMS error (A<=100): 1.951 MeV
  Blind RMS error (A>100):     24.742 MeV

[Step 2] Solving neutron star TOV equations...
  Maximum mass:  M_max = 2.209 M_sun
  Radius at M_max: R = 10.00 km
  Radius at 1.4 M_sun: R_1.4 = 10.75 km

[Step 3] Computing electroweak effective potential...
  Vacuum expectation value: t_min = 246.73 GeV
  (Experimental: v_EW = 246 GeV)

[Step 4] Computing dark matter observables...
  Composite t-particle mass: m_t = 1000.0 GeV
  t-Higgs coupling: kappa = 4.131e-05
  Direct detection cross section: sigma_SI = 6.003e-59 cm^2

[Step 5] Saving results to files...
  Results saved: output_results.txt
  Data saved: data_output.csv

[Step 6] Generating figures...
  [Figures] 6 figures saved (png + pdf)

======================================================================
SUMMARY
======================================================================
  Nuclear fit parameters: [0.01527201 0.01579337 0.0005797  0.02163442 0.00296338]
  Neutron star maximum mass: 2.209 M_sun
  Dark matter: m_t = 1000.0 GeV, sigma_SI = 6.003e-59 cm^2
  All 6 figures saved.
======================================================================
```

---

## Output Files

After running the script, the following files are generated in the same directory:

| File | Description |
|:---|:---|
| `output_results.txt` | All numerical results in text format |
| `data_output.csv` | Predicted mass and error for each nucleus |
| `fig1_binding_energy.png/pdf` | Nuclear binding energy curve (E/A vs A) |
| `fig2_mass_radius.png/pdf` | Neutron star mass–radius relation |
| `fig3_tidal.png/pdf` | Tidal deformability Λ vs M |
| `fig4_rg_running.png/pdf` | Renormalization group running of couplings |
| `fig5_dark_matter.png/pdf` | Dark matter parameter space (m_t, κ) |
| `fig6_errors.png/pdf` | Mass prediction error vs mass number A |

---

## File Structure

```
.
├── Comprehensive_Field_t.py   # Main script (all calculations)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── output_results.txt         # Generated: numerical results
├── data_output.csv            # Generated: per-nucleus data
├── fig1_binding_energy.png    # Generated: figure 1
├── fig2_mass_radius.png       # Generated: figure 2
├── fig3_tidal.png             # Generated: figure 3
├── fig4_rg_running.png        # Generated: figure 4
├── fig5_dark_matter.png       # Generated: figure 5
└── fig6_errors.png            # Generated: figure 6
```

---

## Reproducibility

To verify the results presented in the paper:

1. Clone or download this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run: `python Comprehensive_Field_t.py`
4. Compare the console output with the "Expected Output" section above
5. Compare the generated figures with those in the paper

The code is fully self-contained and does not require any external data files. All nuclear mass data (from AME2020) is embedded directly in the script.

---

## Physics Summary

The framework is based on a single scalar field t (the "comprehensive field") whose local potential wells (t-holes) govern the effective mass of particles across all scales. Key results:

- Nuclear masses: 1.95 MeV RMS for A ≤ 100
- Neutron star: M_max = 2.21 M_sun, R_1.4 = 10.75 km
- Electroweak: VEV = 246.73 GeV dynamically generated
- Dark matter: composite t-particle with m_t = 1000 GeV

---

## License

MIT License

---

## Contact

For questions or comments, please open an issue on GitHub.