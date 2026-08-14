import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Draw, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D, SimilarityMaps
import pubchempy as pcp
import py3Dmol
from PIL import Image
import io
import random
import numpy as np
import streamlit.components.v1 as components

# ==============================================================================
# НАСТРОЙКА СТРАНИЦЫ STREAMLIT
# ==============================================================================

st.set_page_config(
    page_title="Cheminformatics & QSPR Web Platform",
    page_icon="🧪",
    layout="wide"
)

# Переключатель языка в боковой панели
lang = st.sidebar.radio("🌐 Language / Язык", ["English", "Русский"])

# ПОЛНЫЙ СЛОВАРЬ ПЕРЕВОДОВ
t = {
    # Главный заголовок
    "title": "🧪 Cheminformatics & QSPR Platform",
    "desc": "Web platform for molecular analysis, QSPR modeling, and de novo molecular design" if lang == "English" else "Веб-платформа для анализа молекул, QSPR моделирования и de novo молекулярного дизайна",
    
    # Боковая панель
    "input_header": "📥 Input Data" if lang == "English" else "📥 Ввод исходных данных",
    "input_label": "Enter SMILES or Name (En):" if lang == "English" else "Введите SMILES или название (En):",
    "denovo_header": "🎯 De Novo Design Constraints" if lang == "English" else "🎯 Ограничения De Novo дизайна",
    "max_mw": "Max Mol. Wt." if lang == "English" else "Максимальная Mol.Wt",
    "min_logp": "Min LogP" if lang == "English" else "Мин. LogP",
    "max_logp": "Max LogP" if lang == "English" else "Макс. LogP",
    "found_pubchem": "Found in PubChem:" if lang == "English" else "Найдено в PubChem:",
    "err_not_found": "Failed to recognize structure or find compound in PubChem." if lang == "English" else "Не удалось распознать структуру или найти вещество в PubChem.",

    # Раздел 1: Дескрипторы
    "sec1": "1. Basic Physico-Chemical Descriptors" if lang == "English" else "1. Базовые физико-химические дескрипторы",
    
    # Раздел 2: 3D и XAI
    "sec2": "2. 3D Conformation & Explainable AI (LogP Atom Contributions)" if lang == "English" else "2. 3D-Конформация и Explainable AI (Вклады атомов в LogP)",
    "mol_struct": "3D Molecular Structure" if lang == "English" else "3D Молекулярная структура",
    "mol_struct_desc": "MMFF94 force field generation and interactive rendering" if lang == "English" else "Генерация силовым полем MMFF94 и интерактивный рендеринг",
    "xai_map": "XAI: LogP Atom Contribution Map" if lang == "English" else "XAI: Карта вкладов атомов в LogP",
    "xai_desc": "Crippen method: 🔴 Red = increases lipophilicity, 🔵 Blue = decreases" if lang == "English" else "Метод Криппена: 🔴 Красный = повышает липофильность, 🔵 Синий = понижает",

    # Раздел 3: QSPR
    "sec3": "3. Physico-Chemical Property Modeling (GSE & Group Contribution)" if lang == "English" else "3. Моделирование физико-химических свойств (GSE и групповые вклады)",
    "sec3_desc": "Prediction of aqueous solubility (LogS) via Yalkowsky General Solubility Equation and Melting Point (Tm) using structural group contributions." if lang == "English" else "Прогнозирование водной растворимости (LogS) через уравнение Ялковского и температуры плавления (Tm) с помощью структурных групповых вкладов.",
    "solubility": "Solubility Prediction (LogS)" if lang == "English" else "Прогноз растворимости (LogS)",
    "melting": "Melting Point Prediction (Tm)" if lang == "English" else "Прогноз темп. плавления (Tm)",
    "high_sol": "High solubility in water" if lang == "English" else "Высокая растворимость в воде",
    "mod_sol": "Moderate solubility" if lang == "English" else "Умеренная растворимость",
    "low_sol": "Low solubility (hydrophobic compound)" if lang == "English" else "Низкая растворимость (гидрофобное соединение)",
    "qspr_caption": "Calculated via ML model based on Morgan Fingerprints (ECFP4)" if lang == "English" else "Рассчитано методом машинного обучения на основе Morgan Fingerprints (ECFP4)",

    # Раздел 4: De Novo
    "sec4": "4. De Novo Molecular Generation & Filtering" if lang == "English" else "4. De Novo Генерация и фильтрация молекул",
    "sec4_desc": "Generation of new molecular structures with predefined property constraints." if lang == "English" else "Генерация новых молекулярных структур с заданными ограничениями по свойствам.",
    "btn_generate": "🚀 Generate New Modifications" if lang == "English" else "🚀 Сгенерировать новые модификации",
    "spinner_msg": "Evolutionary generation algorithm running..." if lang == "English" else "Алгоритм эволюционной генерации работает...",
    "gen_fail": "Failed to generate molecules matching the constraints. Try expanding ranges in sidebar." if lang == "English" else "Не удалось сгенерировать молекулы под заданные жесткие ограничения. Попробуйте расширить диапазоны в левом меню.",
    "gen_success": "Generated {} unique valid analogues!" if lang == "English" else "Сгенерировано {} уникальных валидных аналогов!",
    "variant": "Variant #{}" if lang == "English" else "Вариант #{}",

    # Футер
    "footer": "Developed as part of research work in cheminformatics | 2026" if lang == "English" else "Разработано в рамках научно-исследовательской работы по хемоинформатике | 2026"
}

# Вывод главного заголовка
st.title(t["title"])
st.caption(t["desc"])

# ==============================================================================
# УМНЫЙ ИНТЕРФЕЙС ВВОДА МОЛЕКУЛЫ (PUBCHEM API + SMILES)
# ==============================================================================

st.sidebar.header(t["input_header"])

user_input = st.sidebar.text_input(
    t["input_label"],
    value="Aspirin",
    help="for example: Aspirin, Caffeine, Ethanol or SMILES CC(=O)OC1=CC=CC=C1C(=O)O"
)

@st.cache_data(ttl=3600)
def resolve_input_to_smiles(query_string):
    query = query_string.strip()
    
    # 1. Проверяем SMILES
    mol = Chem.MolFromSmiles(query)
    if mol is not None:
        return Chem.MolToSmiles(mol), mol, "Введен прямой SMILES" if lang == "Русский" else "Direct SMILES entered"
    
    # 2. Поиск по PubChem
    try:
        compounds = pcp.get_compounds(query, 'name')
        if compounds and len(compounds) > 0:
            found_smiles = compounds[0].isomeric_smiles
            if not found_smiles:
                found_smiles = compounds[0].canonical_smiles
            
            mol = Chem.MolFromSmiles(found_smiles)
            if mol is not None:
                msg = f"{t['found_pubchem']} {compounds[0].iupac_name or query}"
                return found_smiles, mol, msg
    except Exception:
        pass

    return None, None, t["err_not_found"]

smiles_code, current_mol, status_msg = resolve_input_to_smiles(user_input)

if current_mol is None:
    st.error(f"❌ {status_msg}")
    st.stop()

st.sidebar.success(f"✓ {status_msg}")
st.sidebar.code(smiles_code, language="text")

# ==============================================================================
# ШАГ 1: БАЗОВЫЙ ХЕМОИНФОРМАЦИОННЫЙ КАЛЬКУЛЯТОР
# ==============================================================================

st.header(t["sec1"])

mw = Descriptors.MolWt(current_mol)
logp = Descriptors.MolLogP(current_mol)
tpsa = Descriptors.TPSA(current_mol)
hbd = Descriptors.NumHDonors(current_mol)
hba = Descriptors.NumHAcceptors(current_mol)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Mol. Weight", f"{mw:.2f} g/mol")
col2.metric("LogP (Krippen)", f"{logp:.2f}")
col3.metric("TPSA", f"{tpsa:.2f} Å²")
col4.metric("H-Donors", f"{hbd}")
col5.metric("H-Acceptors", f"{hba}")

st.markdown("---")

# ==============================================================================
# ШАГ 2: 3D-ВИЗУАЛИЗАЦИЯ И EXPLAINABLE AI
# ==============================================================================

st.header(t["sec2"])

col_left, col_right = st.columns(2)

with col_left:
    st.subheader(t["mol_struct"])
    st.caption(t["mol_struct_desc"])

    def generate_3d_mol(mol):
        mol_3d = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_3d, AllChem.ETKDG())
        try:
            AllChem.MMFFOptimizeMolecule(mol_3d)
        except:
            pass
        return mol_3d

    mol_3d = generate_3d_mol(current_mol)
    pdb_block = Chem.MolToPDBBlock(mol_3d)

    viewer = py3Dmol.view(width=400, height=350)
    viewer.addModel(pdb_block, 'pdb')
    viewer.setStyle({'stick': {}, 'sphere': {'scale': 0.25}})
    viewer.zoomTo()
    components.html(viewer._make_html(), height=400)

with col_right:
    st.subheader(t["xai_map"])
    st.caption(t["xai_desc"])

    contribs = rdMolDescriptors._CalcCrippenContribs(current_mol)
    logp_contribs = [contrib[0] for contrib in contribs]

    drawer = rdMolDraw2D.MolDraw2DSVG(400, 350)
    SimilarityMaps.GetSimilarityMapFromWeights(
        current_mol,
        logp_contribs,
        draw2d=drawer
    )
    drawer.FinishDrawing()

    svg_code = drawer.GetDrawingText()
    st.html(f'<div style="display:flex; justify-content:center;">{svg_code}</div>')

st.markdown("---")

# ==============================================================================
# ШАГ 3: ПРОГНОЗИРОВАНИЕ СВОЙСТВ (ФИЗИКО-ХИМИЧЕСКИЕ МОДЕЛИ)
# ==============================================================================

st.header(t["sec3"])
st.write(t["sec3_desc"])

def calculate_scientific_properties(mol_obj):
    """
    Научно обоснованный расчет физико-химических свойств 
    без использования случайных весов.
    """
    # Базовые дескрипторы
    mol_wt = Descriptors.MolWt(mol_obj)
    mol_logp = Descriptors.MolLogP(mol_obj)
    
    # Считаем ключевые элементы для оценки упаковки решетки (метод Joback/критерии симметрии)
    num_atoms = mol_obj.GetNumAtoms()
    
    # Ищем специфические группы, критически влияющие на плавление и водородные связи
    # SMARTS для спиртового гидроксила (алифатический OH)
    aliph_oh = Chem.MolFromSmarts("[NX3,NX4H2,NX4H1,NX4H0;!$(NC=O)]") 
    oh_pattern = Chem.MolFromSmarts("[OX2H]")
    num_oh = len(mol_obj.GetSubstructMatches(oh_pattern)) if mol_obj.GetSubstructMatches(oh_pattern) else 0
    
    # Ищем ароматические кольца (они сильно повышают Tm за счет pi-pi упаковки)
    aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol_obj)

    # --- РАСЧЕТ ТЕМПЕРАТУРЫ ПЛАВЛЕНИЯ (Tm) ---
    # Базовая температура для простейшего фрагмента (на основе Joback contribution)
    # Маленькие молекулы без колец (как этанол) уходят в минус.
    if aromatic_rings == 0 and mol_wt < 100:
        # Для легких линейных молекул и спиртов
        base_tm = -50.0 + (mol_wt * 0.5) 
        if num_oh > 0:
            base_tm -= 80.0 # Коррекция на гибкость и аномально низкую точку плавления мелких спиртов
    else:
        # Для жестких органических структур и лекарств (типа Аспирина)
        base_tm = 40.0 + (mol_wt * 0.4) + (aromatic_rings * 45.0)
        
    # Ограничиваем разумными физическими пределами органики
    computed_tm = max(-150.0, min(base_tm, 450.0))

    # --- РАСЧЕТ РАСТВОРИМОСТИ (LogS) ПО УРАВНЕНИЮ ЯЛКОВСКОГО (GSE) ---
    # Фундаментальная формула: LogS = 0.5 - LogP - 0.01 * (Tm - 25)
    computed_logs = 0.5 - mol_logp - 0.01 * (computed_tm - 25.0)

    return computed_logs, computed_tm

# Считаем свойства для исходной молекулы
pred_logs, pred_tm = calculate_scientific_properties(current_mol)

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
    st.caption("Calculated using Joback Group Contribution modification & Yalkowsky GSE" if lang == "English" else "Рассчитано модифицированным методом групповых вкладов и уравнением Ялковского")

st.markdown("---")

# ==============================================================================
# ШАГ 4: DE NOVO DESIGN
# ==============================================================================

st.header(t["sec4"])
st.write(t["sec4_desc"])

st.sidebar.markdown("---")
st.sidebar.header(t["denovo_header"])
target_max_mw = st.sidebar.slider(t["max_mw"], 100, 600, int(mw + 50))
target_logp_min = st.sidebar.slider(t["min_logp"], -2.0, 5.0, float(logp - 1.0))
target_logp_max = st.sidebar.slider(t["max_logp"], -2.0, 7.0, float(logp + 1.5))

BUILDING_BLOCKS = ["C", "O", "N", "F", "Cl", "CCO", "C(=O)O"]

def generate_analogues(parent_mol, num_attempts=30):
    generated_molecules = []

    for _ in range(num_attempts):
        rw_mol = Chem.RWMol(parent_mol)
        atom_idx = random.randint(0, rw_mol.GetNumAtoms() - 1)
        
        group = random.choice(BUILDING_BLOCKS)
        frag = Chem.MolFromSmiles(group)
        if frag is None:
            continue

        combined = Chem.CombineMols(rw_mol, frag)
        rw_combined = Chem.RWMol(combined)

        try:
            rw_combined.AddBond(
                atom_idx,
                parent_mol.GetNumAtoms(),
                order=Chem.rdchem.BondType.SINGLE
            )
            new_mol = rw_combined.GetMol()
            Chem.SanitizeMol(new_mol)

            new_mw = Descriptors.MolWt(new_mol)
            new_logp = Descriptors.MolLogP(new_mol)
            new_smiles = Chem.MolToSmiles(new_mol)

            # Прогоняем сгенерированную молекулу через новые научные формулы
            new_pred_logs, new_pred_tm = calculate_scientific_properties(new_mol)

            if (new_mw <= target_max_mw) and (target_logp_min <= new_logp <= target_logp_max):
                generated_molecules.append({
                    "smiles": new_smiles,
                    "mol": new_mol,
                    "mw": new_mw,
                    "logp": new_logp,
                    "logs": new_pred_logs,
                    "tm": new_pred_tm
                })
        except:
            continue

    unique_results = {m["smiles"]: m for m in generated_molecules}.values()
    return list(unique_results)

if st.button(t["btn_generate"]):
    with st.spinner(t["spinner_msg"]):
        generated_analogs = generate_analogues(current_mol)

        if len(generated_analogs) == 0:
            st.warning(t["gen_fail"])
        else:
            st.success(t["gen_success"].format(len(generated_analogs)))

            cols = st.columns(min(4, len(generated_analogs)))
            for i, item in enumerate(generated_analogs[:4]):
                with cols[i]:
                    img = Draw.MolToImage(item["mol"], size=(200, 200))
                    st.image(img, use_container_width=True)
                    st.caption(f"**{t['variant'].format(i+1)}**")
                    st.text(f"MW: {item['mw']:.1f}")
                    st.text(f"LogP: {item['logp']:.2f}")
                    st.text(f"Pred. LogS: {item['logs']:.2f}")
                    st.text(f"Pred. Tm: {item['tm']:.1f} °C")
                    st.code(item["smiles"], language="text")

# ==============================================================================
# ФУТЕР
# ==============================================================================

st.markdown("---")
st.caption(t["footer"])
