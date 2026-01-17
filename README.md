# Neuro-Symbolic Generation of Hindustani Classical Music

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Status: Research Draft](https://img.shields.io/badge/Status-Research_Draft-orange)]()

**A hierarchical framework for modeling long-form structure (*vistaar*) in data-scarce musical domains.**

This repository contains the code and methodology for my research project: **"A Hierarchical Framework for Modeling Long-Form Structure in Hindustani Classical Music Using Emphasis and Functional Tagging."**

## 🔭 The Problem: Structure in the "Wild"

Hindustani Classical Music (HCM) relies on *vistaar*—an improvised, long-form unfolding of a melodic framework (Raga). Most current generative models (like LSTMs or Transformers) struggle here because:

1. **Data Scarcity:** There are no massive symbolic datasets for specific Ragas.
2. **Long-Horizon Drift:** Purely statistical models "forget" the grammar over long durations.
3. **Black Box Limitations:** End-to-end models are hard to constrain explicitly.

## 💡 The Approach: Guided Stochasticity

This project explores a **Neuro-Symbolic** architecture (combining rules with probability) to decouple "local creativity" from "global structure."

The system consists of three main components:

* **Emphasis Curves:** A continuous signal that dynamically re-weights transition probabilities to mimic the acoustic density of a real performance.
* **Functional Tagging:** A high-level planning system that assigns roles to phrases (e.g., `Introduce`, `Explore`, `Transition`) to ensure the melody moves forward logically.
* **Hybrid Architecture:** Merging the exploratory breadth of Markov chains with the structural rigor of functional tags.

## 📊 Current Results

I conducted an ablative study comparing this framework against standard baselines.

| Model                       | Structural Fidelity (Lower is Better) | Tag Coverage (Higher is Better) |
| :-------------------------- | :------------------------------------ | :------------------------------ |
| **Baseline (Markov)** | High Error (Poor Structure)           | N/A                             |
| **Emphasis Only**     | **Low Error**                   | 70%                             |
| **Hybrid (Proposed)** | **Low Error**                  | **100%**                  |

> **Note:** I am currently validating these objective metrics with a listening study involving expert Hindustani musicians.

## 🚀 Usage

### Prerequisites

* Python 3.9+
* Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```

### Generating Music

The main generation script has constants to change active layers (for ablative testing) and other parameters.

```bash
# Generate a variation in Raga Amritvarshini
python generate_fsm.py
```

## 📂 Repository Structure

* `raga_data_*/`: JSON definitions of Raga grammar and constraints.
* `training/`: Scripts to extract Transition Probability Matrices (TPMs) from the source transcription.
* `model_data_*/`: Trained TPMs, emphasis splines, and tag tables used for generation.
* `samples/`: Generated audio examples (MP3/MIDI).

## 📜 Citation

This work is currently a draft under review. If you find this code useful for your own research or learning, you can cite the draft as:

``Agrawal, P. (2026). A Hierarchical Framework for Modeling Long-Form Structure in Hindustani Classical Music. (Draft).``

## 🙏 Acknowledgments

* Mahesh Kale School of Music (MKSM): For permission to use the didactic recordings for the training corpus.
* Musical experts who participated in the listening study
* MIR experts who provided valuable critiques, insights, and next steps

*This is a research project in progress.*
