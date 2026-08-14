import io
import random
from difflib import SequenceMatcher

import joblib
import pandas as pd
import pubchempy as pcp
import py3Dmol
import streamlit as st
import streamlit.components.v1 as components

from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps

from ml_features import make_single_feature_frame


# -----------------------------
# Конфигурация
# -----------------------------

st.set_page_config(
    page_title="Cheminformatics & QSPR Web Platform",
    page_icon="🧪",
    layout="wide",
)

BUILDING_BLOCKS = ("C", "O", "N", "F", "Cl", "CCO", "C(=O)O")
DESCRIPTOR_FUNCS = dict(Descriptors._descList)


# -----------------------------
# Загрузка ML-моделей
# -----------------------------

@st.cache_resource(show_spinner=False)
def load_models():
    result = {
        "logs_model": None,
        "logs_features": [],
        "logp_model": None,
        "logp_features": [],
        "tm_model": None,
        "tm_features": [],
        "logs_error": None,
        "logp_error": None,
        "tm_error": None,
    }

    model_files = {
        "logs": ("logs_model_v2.pkl", "solubility_model.pkl"),
        "logp": ("logp_model_v2.pkl", "logp_model.pkl"),
        "tm": ("tm_model_v2.pkl", None),
    }

    for key, (primary, fallback) in model_files.items():
        loaded = False
        last_error = None

        for filename in (primary, fallback):
            if not filename:
                continue

            try:
                data = joblib.load(filename)
                result[f"{key}_model"] = data["model"]
                result[f"{key}_features"] = list(data["feature_names"])
                loaded = True
                break
            except Exception as exc:
                last_error = str(exc)

        if not loaded:
            result[f"{key}_error"] = last_error or "model not found"

    return result


MODELS = load_models()

LOGS_MODEL = MODELS["logs_model"]
LOGS_FEATURES = MODELS["logs_features"]

LOGP_MODEL = MODELS["logp_model"]
LOGP_FEATURES = MODELS["logp_features"]

TM_MODEL = MODELS["tm_model"]
TM_FEATURES = MODELS["tm_features"]

HAS_LOGS_MODEL = LOGS_MODEL is not None
HAS_LOGP_MODEL = LOGP_MODEL is not None
HAS_TM_MODEL = TM_MODEL is not None


# -----------------------------
# Утилиты
# -----------------------------

def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def descriptor_value(mol, feature_name):
    # Прямое совпадение с RDKit-дескриптором
    func = DESCRIPTOR_FUNCS.get(feature_name)
    if func is not None:
        try:
            return float(func(mol))
        except Exception:
            return 0.0

    # Частые альтернативные названия признаков
    aliases = {
        "mw": Descriptors.MolWt,
        "molwt": Descriptors.MolWt,
        "molecularweight": Descriptors.MolWt,
        "logp": Descriptors.MolLogP,
        "mollogp": Descriptors.MolLogP,
        "tpsa": Descriptors.TPSA,
        "hbd": Descriptors.NumHDonors,
        "numhdonors": Descriptors.NumHDonors,
        "hba": Descriptors.NumHAcceptors,
        "numhacceptors": Descriptors.NumHAcceptors,
    }

    func = aliases.get(str(feature_name).replace("_", "").lower())
    if func is not None:
        try:
            return float(func(mol))
        except Exception:
            return 0.0

    return 0.0


def build_feature_frame(mol, feature_names):
    # Считаются только признаки, реально используемые моделью
    values = {
        feature: descriptor_value(mol, feature)
        for feature in feature_names
    }
    return pd.DataFrame([values], columns=feature_names)


# -----------------------------
# Расчёт свойств
# -----------------------------

@st.cache_data(show_spinner=False, max_entries=1024)
def calculate_scientific_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol_wt = Descriptors.MolWt(mol)

    # LogP
    if HAS_LOGP_MODEL and LOGP_FEATURES:
        if LOGP_FEATURES[0].startswith(("desc_", "fp_")):
            df_logp = make_single_feature_frame(
                smiles,
                task="logp",
                feature_names=LOGP_FEATURES,
            )
        else:
            df_logp = build_feature_frame(mol, LOGP_FEATURES)

        computed_logp = float(LOGP_MODEL.predict(df_logp)[0])
    else:
        computed_logp = float(Descriptors.MolLogP(mol))

    # LogS
    if HAS_LOGS_MODEL and LOGS_FEATURES:
        if LOGS_FEATURES[0].startswith(("desc_", "fp_")):
            df_logs = make_single_feature_frame(
                smiles,
                task="logs",
                feature_names=LOGS_FEATURES,
            )
        else:
            df_logs = build_feature_frame(mol, LOGS_FEATURES)

        computed_logs = float(LOGS_MODEL.predict(df_logs)[0])
    else:
        computed_logs = 0.5 - computed_logp - 0.01 * (120.0 - 25.0)

    # Tm
    if HAS_TM_MODEL and TM_FEATURES:
        df_tm = make_single_feature_frame(
            smiles,
            task="tm",
            feature_names=TM_FEATURES,
        )
        computed_tm = float(TM_MODEL.predict(df_tm)[0])
    else:
        # Резервная оценка до обучения Tm-модели
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)

        if aromatic_rings == 0 and mol_wt < 100:
            oh_pattern = Chem.MolFromSmarts("[OX2H]")
            num_oh = len(mol.GetSubstructMatches(oh_pattern))
            base_tm = -60.0 + mol_wt * 0.6

            if num_oh > 0:
                base_tm -= 75.0
        else:
            ring_contribution = 45.0

            if aromatic_rings == 1 and hbd <= 1 and mol_wt < 120:
                ring_contribution = 10.0

            base_tm = (
                15.0
                + mol_wt * 0.45
                + aromatic_rings * ring_contribution
                + hbd * 25.0
                + tpsa * 0.3
            )

        computed_tm = max(-185.0, min(base_tm, 450.0))

    return computed_logs, computed_tm, computed_logp


# -----------------------------
# Поиск структуры
# -----------------------------

# Частые соединения доступны без PubChem
LOCAL_COMPOUNDS = {
    "water": ("Water", "O"),
    "вода": ("Water", "O"),

    "ethane": ("Ethane", "CC"),
    "этан": ("Ethane", "CC"),

    "ethanol": ("Ethanol", "CCO"),
    "этанол": ("Ethanol", "CCO"),
    "ethyl alcohol": ("Ethanol", "CCO"),
    "этиловый спирт": ("Ethanol", "CCO"),

    "ethene": ("Ethene", "C=C"),
    "ethylene": ("Ethene", "C=C"),
    "этен": ("Ethene", "C=C"),
    "этилен": ("Ethene", "C=C"),

    "methane": ("Methane", "C"),
    "метан": ("Methane", "C"),

    "methanol": ("Methanol", "CO"),
    "метанол": ("Methanol", "CO"),

    "benzene": ("Benzene", "c1ccccc1"),
    "бензол": ("Benzene", "c1ccccc1"),

    "phenol": ("Phenol", "Oc1ccccc1"),
    "фенол": ("Phenol", "Oc1ccccc1"),

    "caffeine": (
        "Caffeine",
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    ),
    "кофеин": (
        "Caffeine",
        "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    ),

    "aspirin": (
        "Aspirin",
        "CC(=O)Oc1ccccc1C(=O)O",
    ),
    "acetylsalicylic acid": (
        "Aspirin",
        "CC(=O)Oc1ccccc1C(=O)O",
    ),
    "ацетилсалициловая кислота": (
        "Aspirin",
        "CC(=O)Oc1ccccc1C(=O)O",
    ),

    "acetone": ("Acetone", "CC(=O)C"),
    "ацетон": ("Acetone", "CC(=O)C"),

    "toluene": ("Toluene", "Cc1ccccc1"),
    "толуол": ("Toluene", "Cc1ccccc1"),

    "naphthalene": ("Naphthalene", "c1ccc2ccccc2c1"),
    "нафталин": ("Naphthalene", "c1ccc2ccccc2c1"),

    "benzoic acid": ("Benzoic acid", "O=C(O)c1ccccc1"),
    "бензойная кислота": ("Benzoic acid", "O=C(O)c1ccccc1"),

    "propanol": ("1-Propanol", "CCCO"),
    "1-propanol": ("1-Propanol", "CCCO"),
    "пропанол": ("1-Propanol", "CCCO"),

    "isopropanol": ("2-Propanol", "CC(O)C"),
    "2-propanol": ("2-Propanol", "CC(O)C"),
    "isopropyl alcohol": ("2-Propanol", "CC(O)C"),
    "изопропанол": ("2-Propanol", "CC(O)C"),
}


def normalize_chemical_name(value):
    return (
        str(value)
        .strip()
        .lower()
        .replace("–", "-")
        .replace("—", "-")
    )


def compact_name(value):
    return (
        normalize_chemical_name(value)
        .replace(" ", "")
        .replace("-", "")
        .replace(",", "")
    )


def name_similarity(a, b):
    return SequenceMatcher(
        None,
        compact_name(a),
        compact_name(b),
    ).ratio()


def local_name_suggestions(query, limit=5):
    # Группируем синонимы одной молекулы
    unique = {}

    for alias, (display_name, smiles) in LOCAL_COMPOUNDS.items():
        score = name_similarity(query, alias)

        key = (display_name, smiles)

        if key not in unique or score > unique[key]["score"]:
            unique[key] = {
                "name": display_name,
                "smiles": smiles,
                "score": score,
                "source": "local",
            }

    suggestions = sorted(
        unique.values(),
        key=lambda item: item["score"],
        reverse=True,
    )

    # Не показываем совсем непохожие варианты
    return [
        item
        for item in suggestions[:limit]
        if item["score"] >= 0.55
    ]


def pubchem_candidate(query):
    try:
        compounds = pcp.get_compounds(query, "name")

        if not compounds:
            return None, None

        compound = compounds[0]

        found_smiles = (
            getattr(compound, "isomeric_smiles", None)
            or getattr(compound, "canonical_smiles", None)
            or getattr(compound, "connectivity_smiles", None)
        )

        if not found_smiles:
            return None, None

        mol = Chem.MolFromSmiles(found_smiles)
        if mol is None:
            return None, None

        names = []

        iupac_name = getattr(compound, "iupac_name", None)
        if iupac_name:
            names.append(iupac_name)

        try:
            synonym_data = pcp.get_synonyms(
                compound.cid,
                namespace="cid",
            )

            if synonym_data:
                names.extend(
                    synonym_data[0].get("Synonym", [])[:50]
                )
        except Exception:
            pass

        if not names:
            names = [query]

        # Выбираем синоним, наиболее похожий на ввод пользователя
        best_name = max(
            names,
            key=lambda name: name_similarity(query, name),
        )
        best_score = name_similarity(query, best_name)

        candidate = {
            "name": best_name,
            "smiles": Chem.MolToSmiles(mol, canonical=True),
            "score": best_score,
            "source": "pubchem",
        }

        return candidate, None

    except Exception as exc:
        return None, str(exc)


@st.cache_data(ttl=3600, show_spinner=False, max_entries=512)
def resolve_input_to_smiles(query_string):
    query = query_string.strip()

    if not query:
        return {
            "status": "empty",
            "smiles": None,
            "label": None,
            "suggestions": [],
            "error": None,
        }

    # 1. Прямой SMILES
    mol = Chem.MolFromSmiles(query)

    if mol is not None:
        return {
            "status": "resolved",
            "smiles": Chem.MolToSmiles(mol, canonical=True),
            "label": None,
            "source": "smiles",
            "suggestions": [],
            "error": None,
        }

    normalized = normalize_chemical_name(query)

    # 2. Точное локальное название
    if normalized in LOCAL_COMPOUNDS:
        display_name, local_smiles = LOCAL_COMPOUNDS[normalized]
        mol = Chem.MolFromSmiles(local_smiles)

        return {
            "status": "resolved",
            "smiles": Chem.MolToSmiles(mol, canonical=True),
            "label": display_name,
            "source": "local",
            "suggestions": [],
            "error": None,
        }

    # 3. Получаем возможный результат PubChem
    candidate, pubchem_error = pubchem_candidate(query)

    # Если PubChem действительно нашёл практически точное имя
    if candidate and candidate["score"] >= 0.94:
        return {
            "status": "resolved",
            "smiles": candidate["smiles"],
            "label": candidate["name"],
            "source": "pubchem",
            "suggestions": [],
            "error": None,
        }

    # 4. Формируем варианты для исправления опечатки
    suggestions = local_name_suggestions(query)

    if candidate and candidate["score"] >= 0.55:
        duplicate = any(
            item["smiles"] == candidate["smiles"]
            for item in suggestions
        )

        if not duplicate:
            suggestions.append(candidate)

    suggestions = sorted(
        suggestions,
        key=lambda item: item["score"],
        reverse=True,
    )[:5]

    if suggestions:
        return {
            "status": "suggest",
            "smiles": None,
            "label": None,
            "source": None,
            "suggestions": suggestions,
            "error": pubchem_error,
        }

    return {
        "status": "not_found",
        "smiles": None,
        "label": None,
        "source": None,
        "suggestions": [],
        "error": pubchem_error,
    }


# -----------------------------
# 3D-конформация
# -----------------------------

@st.cache_data(show_spinner=False, max_entries=256)
def generate_3d_pdb(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol_3d = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    params.useRandomCoords = False

    result = AllChem.EmbedMolecule(mol_3d, params)
    if result != 0:
        return None

    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_3d):
            AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=150)
        else:
            AllChem.UFFOptimizeMolecule(mol_3d, maxIters=150)
    except Exception:
        pass

    return Chem.MolToPDBBlock(mol_3d)


# -----------------------------
# XAI-карта
# -----------------------------

@st.cache_data(show_spinner=False, max_entries=256)
def get_xai_map_png(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        mol_2d = Chem.Mol(mol)
        mol_2d.RemoveAllConformers()
        AllChem.Compute2DCoords(mol_2d)

        contribs = rdMolDescriptors._CalcCrippenContribs(mol_2d)
        logp_contribs = [float(item[0]) for item in contribs]

        # Та же SimilarityMaps-карта, вывод через Cairo
        drawer = rdMolDraw2D.MolDraw2DCairo(400, 300)

        SimilarityMaps.GetSimilarityMapFromWeights(
            mol_2d,
            logp_contribs,
            draw2d=drawer,
            contourLines=6,
            step=0.08,
            alpha=0.45,
        )

        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    except Exception:
        return None


# -----------------------------
# De Novo
# -----------------------------

def generate_analogues(
    parent_smiles,
    max_mw,
    min_logp,
    max_logp,
    num_attempts=30,
):
    parent_mol = Chem.MolFromSmiles(parent_smiles)
    if parent_mol is None or parent_mol.GetNumAtoms() == 0:
        return []

    generated = {}
    parent_atom_count = parent_mol.GetNumAtoms()

    for _ in range(num_attempts):
        try:
            atom_idx = random.randrange(parent_atom_count)
            group = random.choice(BUILDING_BLOCKS)
            frag = Chem.MolFromSmiles(group)

            if frag is None:
                continue

            combined = Chem.CombineMols(parent_mol, frag)
            rw_mol = Chem.RWMol(combined)

            rw_mol.AddBond(
                atom_idx,
                parent_atom_count,
                Chem.rdchem.BondType.SINGLE,
            )

            new_mol = rw_mol.GetMol()
            Chem.SanitizeMol(new_mol)

            new_mw = float(Descriptors.MolWt(new_mol))
            if new_mw > max_mw:
                continue

            new_smiles = Chem.MolToSmiles(new_mol, canonical=True)
            if new_smiles in generated:
                continue

            properties = calculate_scientific_properties(new_smiles)
            if properties is None:
                continue

            pred_logs, pred_tm, pred_logp = properties

            # Ограничения применяются к прогнозируемому LogP
            if not (min_logp <= pred_logp <= max_logp):
                continue

            generated[new_smiles] = {
                "smiles": new_smiles,
                "mw": new_mw,
                "logp": pred_logp,
                "logs": pred_logs,
                "tm": pred_tm,
            }

        except Exception:
            continue

    return list(generated.values())


# -----------------------------
# Локализация
# -----------------------------

lang = st.sidebar.radio("🌐 Language / Язык", ["English", "Русский"])

t = {
    "title": "🧪 Cheminformatics & QSPR Platform",
    "desc": (
        "Web platform for molecular analysis, QSPR modeling, and de novo molecular design"
        if lang == "English"
        else "Веб-платформа для анализа молекул, QSPR моделирования и de novo молекулярного дизайна"
    ),
    "input_header": "📥 Input Data" if lang == "English" else "📥 Ввод исходных данных",
    "input_label": "Enter SMILES or Name (En):" if lang == "English" else "Введите SMILES или название (En):",
    "denovo_header": "🎯 De Novo Design Constraints" if lang == "English" else "🎯 Ограничения De Novo дизайна",
    "max_mw": "Max Mol. Wt." if lang == "English" else "Максимальная Mol.Wt",
    "min_logp": "Min LogP" if lang == "English" else "Мин. LogP",
    "max_logp": "Max LogP" if lang == "English" else "Макс. LogP",
    "found_pubchem": "Found in PubChem:" if lang == "English" else "Найдено в PubChem:",
    "direct_smiles": "Direct SMILES entered" if lang == "English" else "Введен прямой SMILES",
    "err_not_found": (
        "Failed to recognize structure or find compound in PubChem."
        if lang == "English"
        else "Не удалось распознать структуру или найти вещество в PubChem."
    ),
    "sec1": "1. Basic Physico-Chemical Descriptors" if lang == "English" else "1. Базовые физико-химические дескрипторы",
    "sec2": (
        "2. 3D Conformation & Explainable AI (LogP Atom Contributions)"
        if lang == "English"
        else "2. 3D-Конформация и Explainable AI (Вклады атомов в LogP)"
    ),
    "mol_struct": "3D Molecular Structure" if lang == "English" else "3D Молекулярная структура",
    "mol_struct_desc": (
        "MMFF94 force field generation and interactive rendering"
        if lang == "English"
        else "Генерация силовым полем MMFF94 и интерактивный рендеринг"
    ),
    "xai_map": "XAI: LogP Atom Contribution Map" if lang == "English" else "XAI: Карта вкладов атомов в LogP",
    "xai_desc": (
        "Crippen method: 🔴 Red = increases lipophilicity, 🔵 Blue = decreases"
        if lang == "English"
        else "Метод Криппена: 🔴 Красный = повышает липофильность, 🔵 Синий = понижает"
    ),
    "sec3": (
        "3. Physico-Chemical Property Modeling (GSE & Group Contribution)"
        if lang == "English"
        else "3. Моделирование физико-химических свойств (GSE и групповые вклады)"
    ),
    "sec3_desc": (
        "Prediction of aqueous solubility (LogS) and melting point (Tm)."
        if lang == "English"
        else "Прогнозирование водной растворимости (LogS) и температуры плавления (Tm)."
    ),
    "solubility": "Solubility Prediction (LogS)" if lang == "English" else "Прогноз растворимости (LogS)",
    "melting": "Melting Point Prediction (Tm)" if lang == "English" else "Прогноз темп. плавления (Tm)",
    "high_sol": "High solubility in water" if lang == "English" else "Высокая растворимость в воде",
    "mod_sol": "Moderate solubility" if lang == "English" else "Умеренная растворимость",
    "low_sol": (
        "Low solubility (hydrophobic compound)"
        if lang == "English"
        else "Низкая растворимость (гидрофобное соединение)"
    ),
    "sec4": "4. De Novo Molecular Generation & Filtering" if lang == "English" else "4. De Novo Генерация и фильтрация молекул",
    "sec4_desc": (
        "Generation of new molecular structures with predefined property constraints."
        if lang == "English"
        else "Генерация новых молекулярных структур с заданными ограничениями по свойствам."
    ),
    "btn_generate": "🚀 Generate New Modifications" if lang == "English" else "🚀 Сгенерировать новые модификации",
    "spinner_msg": (
        "Evolutionary generation algorithm running..."
        if lang == "English"
        else "Алгоритм эволюционной генерации работает..."
    ),
    "gen_fail": (
        "Failed to generate molecules matching the constraints. Try expanding ranges in sidebar."
        if lang == "English"
        else "Не удалось сгенерировать молекулы под заданные ограничения. Попробуйте расширить диапазоны."
    ),
    "gen_success": "Generated {} unique valid analogues!" if lang == "English" else "Сгенерировано {} уникальных валидных аналогов!",
    "variant": "Variant #{}" if lang == "English" else "Вариант #{}",
    "footer": (
        "Developed as part of research work in cheminformatics | 2026"
        if lang == "English"
        else "Разработано в рамках научно-исследовательской работы по хемоинформатике | 2026"
    ),
}


# -----------------------------
# Интерфейс
# -----------------------------

st.title(t["title"])
st.caption(t["desc"])

if MODELS["logs_error"]:
    st.sidebar.warning("⚠️ solubility_model.pkl не загружен")

if MODELS["logp_error"]:
    st.sidebar.warning("⚠️ LogP model не загружена")

if MODELS["tm_error"]:
    st.sidebar.info(
        "ℹ️ Tm model не загружена: используется резервная эмпирическая оценка"
    )

st.sidebar.header(t["input_header"])

user_input = st.sidebar.text_input(
    t["input_label"],
    value="Aspirin",
    help="for example: Aspirin, Caffeine, Ethanol or SMILES CC(=O)OC1=CC=CC=C1C(=O)O",
)

resolution = resolve_input_to_smiles(user_input)

# Сбрасываем старое подтверждение при изменении ввода
if st.session_state.get("last_raw_query") != user_input:
    st.session_state["last_raw_query"] = user_input
    st.session_state.pop("confirmed_suggestion", None)

smiles_code = None
resolved_label = None
resolve_source = None

if resolution["status"] == "resolved":
    smiles_code = resolution["smiles"]
    resolved_label = resolution["label"]
    resolve_source = resolution.get("source")

elif resolution["status"] == "suggest":
    suggestions = resolution["suggestions"]

    if lang == "English":
        st.sidebar.warning(
            "Exact compound name was not found. Did you mean:"
        )
    else:
        st.sidebar.warning(
            "Точное название не найдено. Возможно, вы имели в виду:"
        )

    option_labels = []

    for item in suggestions:
        percent = int(round(item["score"] * 100))
        option_labels.append(
            f"{item['name']}  ·  {percent}%"
        )

    selected_index = st.sidebar.selectbox(
        "Suggested compound"
        if lang == "English"
        else "Предполагаемое соединение",
        options=range(len(suggestions)),
        format_func=lambda idx: option_labels[idx],
        key="suggested_compound_select",
    )

    selected = suggestions[selected_index]

    st.sidebar.caption(
        f"SMILES: {selected['smiles']}"
    )

    use_suggestion = st.sidebar.button(
        "✓ Use this compound"
        if lang == "English"
        else "✓ Использовать это соединение",
        use_container_width=True,
        key="confirm_compound_suggestion",
    )

    if use_suggestion:
        st.session_state["confirmed_suggestion"] = {
            "raw_query": user_input,
            "name": selected["name"],
            "smiles": selected["smiles"],
            "source": selected["source"],
        }

    confirmed = st.session_state.get("confirmed_suggestion")

    if (
        confirmed
        and confirmed.get("raw_query") == user_input
    ):
        smiles_code = confirmed["smiles"]
        resolved_label = confirmed["name"]
        resolve_source = "suggestion"
    else:
        st.info(
            "Select the intended compound in the sidebar to continue."
            if lang == "English"
            else "Выберите предполагаемое соединение в боковой панели, чтобы продолжить."
        )
        st.stop()

else:
    st.error(f"❌ {t['err_not_found']}")

    if resolution.get("error"):
        st.caption(
            "PubChem lookup is currently unavailable."
            if lang == "English"
            else "Сервис PubChem в данный момент недоступен."
        )

    st.stop()

current_mol = Chem.MolFromSmiles(smiles_code)

if current_mol is None:
    st.error(
        "Resolved structure is invalid."
        if lang == "English"
        else "Полученная структура некорректна."
    )
    st.stop()

if resolve_source == "pubchem":
    status_msg = f"{t['found_pubchem']} {resolved_label}"
elif resolve_source == "local":
    status_msg = (
        f"Resolved locally: {resolved_label}"
        if lang == "English"
        else f"Распознано локально: {resolved_label}"
    )
elif resolve_source == "suggestion":
    status_msg = (
        f"Selected suggestion: {resolved_label}"
        if lang == "English"
        else f"Выбрано предполагаемое соединение: {resolved_label}"
    )
else:
    status_msg = t["direct_smiles"]

st.sidebar.success(f"✓ {status_msg}")
st.sidebar.code(smiles_code, language="text")


# -----------------------------
# Базовые дескрипторы
# -----------------------------

properties = calculate_scientific_properties(smiles_code)

if properties is None:
    st.error(t["err_not_found"])
    st.stop()

pred_logs, pred_tm, pred_logp = properties

mw = float(Descriptors.MolWt(current_mol))
tpsa = float(Descriptors.TPSA(current_mol))
hbd = int(Descriptors.NumHDonors(current_mol))
hba = int(Descriptors.NumHAcceptors(current_mol))

st.header(t["sec1"])

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Mol. Weight", f"{mw:.2f} g/mol")
col2.metric("LogP (AI model)", f"{pred_logp:.2f}")
col3.metric("TPSA", f"{tpsa:.2f} Å²")
col4.metric("H-Donors", str(hbd))
col5.metric("H-Acceptors", str(hba))

st.markdown("---")


# -----------------------------
# 3D и XAI
# -----------------------------

st.header(t["sec2"])
col_left, col_right = st.columns(2)

with col_left:
    st.subheader(t["mol_struct"])
    st.caption(t["mol_struct_desc"])

    # 3D запускается только по запросу
    show_3d = st.toggle(
        "Show interactive 3D" if lang == "English" else "Показать интерактивную 3D-модель",
        value=False,
        key=f"show_3d_{smiles_code}",
    )

    if show_3d:
        pdb_block = generate_3d_pdb(smiles_code)

        if pdb_block:
            viewer = py3Dmol.view(width=400, height=350)
            viewer.addModel(pdb_block, "pdb")
            viewer.setStyle({"stick": {}, "sphere": {"scale": 0.25}})
            viewer.zoomTo()

            components.html(
                viewer._make_html(),
                height=370,
                scrolling=False,
            )
        else:
            st.info(
                "3D structure is unavailable."
                if lang == "English"
                else "3D-структура недоступна."
            )
    else:
        # Лёгкий preview без WebGL
        preview = Draw.MolToImage(current_mol, size=(360, 300))
        st.image(preview, use_container_width=False)

with col_right:
    st.subheader(t["xai_map"])
    st.write(t["xai_desc"])

    xai_png = get_xai_map_png(smiles_code)

    if xai_png:
        st.image(
            xai_png,
            use_container_width=False,
        )
    else:
        st.info(
            "Can not generate 2D contribution map."
            if lang == "English"
            else "Не удалось сгенерировать 2D карту вкладов."
        )

st.markdown("---")


# -----------------------------
# QSPR
# -----------------------------

st.header(t["sec3"])
st.write(t["sec3_desc"])

col_qspr1, col_qspr2 = st.columns(2)

with col_qspr1:
    st.metric(t["solubility"], f"{pred_logs:.2f} mol/L")

    if pred_logs > -2:
        st.info(f"💧 {t['high_sol']}")
    elif pred_logs > -4:
        st.warning(f"⚠️ {t['mod_sol']}")
    else:
        st.error(f"🚫 {t['low_sol']}")

with col_qspr2:
    st.metric(t["melting"], f"{pred_tm:.1f} °C")
    if HAS_TM_MODEL:
        st.caption(
            "Predicted by the experimental-data ML model"
            if lang == "English"
            else "Прогноз ML-моделью, обученной на экспериментальных данных"
        )
    else:
        st.caption(
            "Temporary descriptor-based empirical estimate"
            if lang == "English"
            else "Временная эмпирическая оценка на основе дескрипторов"
        )

st.markdown("---")


# -----------------------------
# De Novo
# -----------------------------

st.header(t["sec4"])
st.write(t["sec4_desc"])

st.sidebar.markdown("---")
st.sidebar.header(t["denovo_header"])

default_max_mw = int(min(600, max(100, round(mw + 50))))
default_min_logp = float(min(5.0, max(-2.0, pred_logp - 1.0)))
default_max_logp = float(min(7.0, max(-2.0, pred_logp + 1.5)))

# Форма предотвращает rerun на каждом движении слайдера
with st.sidebar.form("denovo_constraints"):
    target_max_mw = st.slider(
        t["max_mw"],
        min_value=100,
        max_value=600,
        value=default_max_mw,
    )

    target_logp_min = st.slider(
        t["min_logp"],
        min_value=-2.0,
        max_value=5.0,
        value=default_min_logp,
        step=0.1,
    )

    target_logp_max = st.slider(
        t["max_logp"],
        min_value=-2.0,
        max_value=7.0,
        value=default_max_logp,
        step=0.1,
    )

    generate_clicked = st.form_submit_button(t["btn_generate"], use_container_width=True)

if "generated_analogs" not in st.session_state:
    st.session_state.generated_analogs = []

if "generated_for_smiles" not in st.session_state:
    st.session_state.generated_for_smiles = None

if generate_clicked:
    if target_logp_min > target_logp_max:
        st.warning(
            "Min LogP must be lower than Max LogP."
            if lang == "English"
            else "Мин. LogP должен быть меньше Макс. LogP."
        )
    else:
        with st.spinner(t["spinner_msg"]):
            st.session_state.generated_analogs = generate_analogues(
                parent_smiles=smiles_code,
                max_mw=target_max_mw,
                min_logp=target_logp_min,
                max_logp=target_logp_max,
                num_attempts=30,
            )
            st.session_state.generated_for_smiles = smiles_code

generated_analogs = (
    st.session_state.generated_analogs
    if st.session_state.generated_for_smiles == smiles_code
    else []
)

if generate_clicked or generated_analogs:
    if not generated_analogs:
        st.warning(t["gen_fail"])
    else:
        st.success(t["gen_success"].format(len(generated_analogs)))

        shown = generated_analogs[:4]
        cols = st.columns(len(shown))

        for i, item in enumerate(shown):
            with cols[i]:
                mol = Chem.MolFromSmiles(item["smiles"])

                if mol is not None:
                    img = Draw.MolToImage(mol, size=(220, 220))
                    st.image(img, use_container_width=True)

                st.caption(f"**{t['variant'].format(i + 1)}**")
                st.text(f"MW: {item['mw']:.1f}")
                st.text(f"LogP: {item['logp']:.2f}")
                st.text(f"Pred. LogS: {item['logs']:.2f}")
                st.text(f"Pred. Tm: {item['tm']:.1f} °C")
                st.code(item["smiles"], language="text")


# -----------------------------
# Футер
# -----------------------------

st.markdown("---")
st.caption(t["footer"])
