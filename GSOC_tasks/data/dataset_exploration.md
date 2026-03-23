# Quark-Gluon Jet Classification Dataset Exploration

This document provides a comprehensive overview of the `quark-gluon_data-set_n139306.hdf5` dataset, which is a standard benchmark in High Energy Physics (HEP) for machine learning, specifically for the task of **Quark-Gluon Discrimination**.

---

## 1. Overview
- **Dataset Name**: Quark-Gluon Jet Images (derived from CMS Open Data)
- **Total Samples**: 139,306
- **File Format**: HDF5 (`.hdf5`)
- **Primary Task**: Binary Classification (Quark vs. Gluon)
- **Original Source**: Based on the work of Komiske, Metodiev, and Thaler (MIT).

---

## 2. Dataset Structure (HDF5 Keys)

The HDF5 file contains four primary datasets:

| Key | Shape | Data Type | Description |
| :--- | :--- | :--- | :--- |
| **`X_jets`** | `(139306, 125, 125, 3)` | `float32` | **Jet Images**. 125x125 pixel resolution with 3 color channels. |
| **`y`** | `(139306,)` | `float32` | **Labels**. Ground truth classification (0 = Gluon, 1 = Quark). |
| **`m0`** | `(139306,)` | `float32` | **Mass**. The invariant mass of the jet. |
| **`pt`** | `(139306,)` | `float32` | **Transverse Momentum**. The $p_T$ of the consolidated jet. |

---

## 3. Feature Definitions

### `X_jets`: The Jet Image
The image is a 2D histogram of the energy deposited by particles in the detector, centered and rotated relative to the jet axis. The 3 channels represent different types of particles detected:

1. **Channel 0 (Charged Particles)**: The summed $p_T$ of all charged particles (tracks) in that pixel.
2. **Channel 1 (Neutral Hadrons)**: The summed $p_T$ of all neutral hadrons in that pixel.
3. **Channel 2 (Photons)**: The summed $p_T$ of all photons in that pixel.

### `y`: The Labels
Standard convention for this dataset:
- **`0.0` (Gluon)**: A jet initiated by a gluon.
- **`1.0` (Quark)**: A jet initiated by a light quark (up, down, or strange).

---

## 4. Physical Significance

### What are Quarks and Gluons?
- **Quarks**: Fundamental fermions that carry a "color charge" of 3. They are the building blocks of hadrons like protons and neutrons.
- **Gluons**: Fundamental gauge bosons that carry a "color charge" of 8. They mediate the strong force (QCD).

### What is a "Jet"?
In high-energy collisions (like those at the LHC), quarks and gluons are produced at high energies. Because of **color confinement**, they cannot exist freely. Instead, they undergo "fragmentation" and "hadronization," producing a narrow cone of particles—this is called a **Jet**.

### Why Classify Them?
Distinguishing between jets initiated by quarks and those initiated by gluons is a critical task in HEP:
- **Gluons** radiate more because they have a higher color charge ($C_A = 3$) compared to **Quarks** ($C_F = 4/3$).
- This causes Gluon jets to be **broader**, have **more particles** (higher multiplicity), and a **different energy distribution** within the jet cone compared to Quark jets.
- Success in this task helps physicists identify specific decay processes, such as identifying if a Higgs boson decayed into a pair of bottom quarks or if a signal is just "background" gluon noise.

---

## 5. Machine Learning Context

This dataset is frequently used to train:
- **Convolutional Neural Networks (CNNs)**: Using the `X_jets` images.
- **Graph Neural Networks (GNNs)**: Often by converting the images back into point clouds or using them directly as grid-graphs.
- **Autoencoders**: For anomaly detection in jet substructure.

### Citation & References
If you use this dataset in research, it is typically attributed to:
> *Patrick T. Komiske, Eric M. Metodiev, and Jesse Thaler, "Jet Images — Deep Learning Edition," JHEP 1701 (2017) 110, [arXiv:1612.01551](https://arxiv.org/abs/1612.01551).*
