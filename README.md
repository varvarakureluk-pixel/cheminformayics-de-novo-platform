# Web-Based Cheminformatics & QSPR Platform for De Novo Molecular Design

A functional interactive web application developed with **Python** and **Streamlit** for molecular structure analysis, prediction of physicochemical properties using QSPR models, molecular visualization, and fragment-based *de novo* generation of structural analogues.

🚀 **Live Demo:** [Launch Cheminformatics Platform](https://cheminformatic.streamlit.app/)

---

## 🔬 Core Features

### Molecular Input and Structure Retrieval

The application accepts either a **SMILES string** or the name of a chemical compound. Molecular names are resolved using **PubChem**, while several common compounds can also be recognized locally.

If an entered name does not correspond exactly to a known compound, the application can suggest similar names instead of automatically selecting a potentially unrelated molecular structure.

### Molecular Descriptor Calculation

Molecular structures are processed using **RDKit**. The application calculates physicochemical and structural descriptors including molecular weight, TPSA, hydrogen-bond donors and acceptors, ring-related descriptors, and other molecular characteristics.

Morgan fingerprints (**ECFP4**) are additionally used as molecular representations for the machine-learning models.

### QSPR Property Prediction

Three independent regression models are used to predict:

**Lipophilicity (LogP)**
**Aqueous solubility (LogS)**
**Melting point (Tm)**

The models are based on a combination of RDKit molecular descriptors and Morgan fingerprints and were trained using the **Extra Trees regression** algorithm.

The LogS model was trained using experimental aqueous solubility data from the **ESOL dataset**.

The LogP model was trained using experimental and adjusted lipophilicity measurements from the **SangsterLogP dataset**.

The melting-point model was trained using the highly curated **Jean-Claude Bradley Double Plus Good Melting Point Dataset**.

---

## 📊 Model Validation

Model performance was evaluated using both conventional random train/test splitting and **scaffold-based splitting**.

Scaffold validation provides a more demanding estimate of model performance because molecules in the test set contain structural frameworks that differ from those represented in the training data.

### LogS Model

Random split:

**R² = 0.874**

Scaffold split:

**R² = 0.867**

For scaffold validation, the mean absolute error was approximately **0.54 log units**, with an RMSE of approximately **0.70 log units**.

### LogP Model

The final deployed LogP model uses **150 Extra Trees estimators**.

Random split:

**R² = 0.827**
**MAE = 0.501**
**RMSE = 0.700**

Scaffold split:

**R² = 0.651**
**MAE = 0.695**
**RMSE = 0.939**

Reducing the model from 400 to 150 trees substantially decreased the model file size while producing only a minimal change in validation performance.

### Melting Point Model

Random split:

**R² = 0.821**
**MAE = 30.334 °C**
**RMSE = 40.979 °C**

Scaffold split:

**R² = 0.784**
**MAE = 35.247 °C**
**RMSE = 49.650 °C**

Melting point is particularly difficult to predict from two-dimensional molecular structure because crystal packing, polymorphism, intermolecular interactions, and other solid-state effects are not completely represented by conventional molecular descriptors and fingerprints.

---

## 🧬 Molecular Visualization

Three-dimensional molecular conformations are generated using **RDKit** embedding and geometry optimization and displayed interactively in the browser using **py3Dmol**.

The application also generates a two-dimensional lipophilicity contribution map using **RDKit Crippen atomic contributions and Similarity Maps**.

This visualization indicates which regions of the molecular structure contribute positively or negatively to lipophilicity according to the Crippen approach.

The Crippen contribution map is an interpretable molecular visualization and is independent of the experimental LogP machine-learning model.

---

## 🧪 Fragment-Based De Novo Design

The platform includes a simple *de novo* molecular design module for generating structural analogues of an input compound.

Candidate structures are produced by attaching predefined molecular fragments to the parent structure. Generated molecules are chemically validated, duplicate structures are removed, and candidates are filtered according to user-defined physicochemical constraints such as molecular weight and lipophilicity.

The remaining candidate molecules can then be evaluated using the trained QSPR models.

This functionality provides a simple workflow for exploring how structural modifications may influence predicted molecular properties.

---

## 🛠️ Technologies

The application is developed primarily in **Python**.

**Streamlit** is used to build the interactive web interface.

**RDKit** provides molecular parsing, descriptor calculation, molecular fingerprints, molecular drawing, conformer generation, and cheminformatics functionality.

**scikit-learn** is used for the machine-learning regression models.

**PubChemPy** provides access to PubChem for compound-name resolution.

**py3Dmol** provides interactive WebGL-based three-dimensional molecular visualization.

**pandas** and **NumPy** are used for data processing and numerical calculations.

**joblib** is used for storing and loading the trained machine-learning models.

---

## 📦 Running the Application Locally

Clone the repository:

```bash
git clone https://github.com/varvarakureluluk-pixel/cheminformayics-de-novo-platform.git
cd cheminformayics-de-novo-platform
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app3_ml_v4.py
```

---

## ⚠️ Limitations

The platform is intended as a **research and educational cheminformatics tool** and should not be considered a replacement for experimental measurements.

Prediction accuracy depends on how well a target compound is represented within the chemical space of the corresponding training dataset.

Compounds that differ substantially from structures encountered during model training may have considerably higher prediction uncertainty.

The reported R², MAE, and RMSE values therefore describe overall validation performance and should not be interpreted as guaranteed accuracy for every individual molecule.

---

## 🎓 Project Context

This project was developed as part of undergraduate research in **cheminformatics, QSPR modeling, molecular property prediction, and computational molecular design**.
