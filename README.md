# Web-Based Cheminformatics Platform for De Novo Molecular Design

A functional, interactive web application built with **Python** and **Streamlit** designed to automate molecular properties analysis, predict thermodynamic parameters via QSPR, and perform fragment-based *de novo* molecular generation.

🚀 **Live Demo:** [ВСТАВЬ СЮДА ССЫЛКУ НА СВОЙ STREAMLIT, КОГДА ЗАПУСТИШЬ]

---

## 🔬 Core Features

- **Automated Structural Retrieval:** Connects to the **PubChem REST API** to instantly fetch molecular structures using trivial names (e.g., *Caffeine*, *Aspirin*) and converts them into standardized SMILES strings.
- **3D Conformation Generation:** Computes 3D coordinates using **RDKit** embedding and energy minimization algorithms, providing an interactive, rotatable 3D model in the browser.
- **Explainable AI (XAI) Analytics:** Computes fundamental descriptors (Molecular Weight, Lipophilicity $\log P$, and Topological Polar Surface Area - $TPSA$) and visualizes atomic contributions using Crippen visualization.
- **Predictive QSPR Modeling:** Estimates critical thermodynamic properties, including Melting Point ($T_m$), Boiling Point ($T_b$), and Aqueous Solubility ($\log S$) based on molecular graph vectorization.
- **Fragment-Based De Novo Design:** Generates novel structural derivatives within user-defined target property windows (such as specific molecular mass ranges or Target Lipophilicity) via a radical substitution mutation algorithm.

---

## 🛠️ Tech Stack & Libraries

- **Frontend / UI:** [Streamlit](https://streamlit.io) — for reactive, web-oriented interface development.
- **Cheminformatics Engine:** [RDKit](https://rdkit.org) — for parsing SMILES, 3D conformation generation, and molecular descriptor calculations.
- **Data Integration:** [PubChemPy](https://readthedocs.io) — for semantic chemical search and database querying.
- **3D Rendering:** `py3Dmol` & `stmol` — for interactive WebGL molecular visualization.
- **Image Processing:** `Pillow (PIL)` — for 2D molecular structure rendering and mapping.

---

## 📦 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com
   cd cheminformatics-de-novo-platform
   ```

2. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---
*Developed as an undergraduate research project in Computational Chemistry & Informatics.*
