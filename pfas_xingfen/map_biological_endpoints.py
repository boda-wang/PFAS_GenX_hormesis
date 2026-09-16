"""Map curated effect-measurement labels to biological endpoint categories."""

import pandas as pd

input_file = '../pfas_xingfen/curated_dataset.xlsx'
df = pd.read_excel(input_file)

# Effect-measurement labels are mapped to the 11 biological endpoint categories.
endpoint_mapping = {
    'crh': 'Endocrine_Thyroid', 'tsh β': 'Endocrine_Thyroid', 'tshβ': 'Endocrine_Thyroid', 'nis': 'Endocrine_Thyroid', 'tg': 'Endocrine_Thyroid',
    'ttr': 'Endocrine_Thyroid', 'tr α': 'Endocrine_Thyroid', 'trα': 'Endocrine_Thyroid', 'tr β': 'Endocrine_Thyroid', 'trβ': 'Endocrine_Thyroid',
    'dio1': 'Endocrine_Thyroid', 'dio2': 'Endocrine_Thyroid', 'T4': 'Endocrine_Thyroid', 'T3': 'Endocrine_Thyroid', 'Female T4': 'Endocrine_Thyroid',
    'Male T4': 'Endocrine_Thyroid', 'Female T3': 'Endocrine_Thyroid', 'Male T3': 'Endocrine_Thyroid',
    'TSH level': 'Endocrine_Thyroid', 'T4 level': 'Endocrine_Thyroid', 'NIS': 'Endocrine_Thyroid',
    'Ratio of glycosylated NIS': 'Endocrine_Thyroid', 'Iodide uptake': 'Endocrine_Thyroid',

    'Male VTG1': 'Endocrine_Reproductive', 'Male VTG3': 'Endocrine_Reproductive',
    'Female VTG1': 'Endocrine_Reproductive', 'Female VTG3': 'Endocrine_Reproductive', 'bmp15': 'Endocrine_Reproductive',
    'cyp19a1a': 'Endocrine_Reproductive', 'cyp19ala': 'Endocrine_Reproductive', 'foxl2': 'Endocrine_Reproductive',
    'sox9b': 'Endocrine_Reproductive', 'vtg1': 'Endocrine_Reproductive', 'esr1': 'Endocrine_Reproductive',
    'esr2b': 'Endocrine_Reproductive', 'wnt4a': 'Endocrine_Reproductive', 'dmrt1': 'Endocrine_Reproductive',
    'amh': 'Endocrine_Reproductive', 'sox9a': 'Endocrine_Reproductive', 'E2': 'Endocrine_Reproductive',
    'E2 level': 'Endocrine_Reproductive', 'T level': 'Endocrine_Reproductive', 'T/E2': 'Endocrine_Reproductive',
    'VTG': 'Endocrine_Reproductive', 'T': 'Endocrine_Reproductive',

    'AchE activity': 'Behavior_Neuro', 'toal swimming distance': 'Behavior_Neuro',
    'exercise duration(s)': 'Behavior_Neuro', 'average locomotion speed(mm/s)': 'Behavior_Neuro',
    'average locomotion speed(mm/s）': 'Behavior_Neuro',
    'Total distance (mm)': 'Behavior_Neuro', 'Total distance（mm)': 'Behavior_Neuro',
    'Dopamine content': 'Behavior_Neuro', 'Swirl-escape rate': 'Behavior_Neuro', 'Distance moved': 'Behavior_Neuro', 'embryo mov': 'Behavior_Neuro',
    'Average number of visits': 'Behavior_Neuro', 'Average Time/ visit': 'Behavior_Neuro',
    'Time spent in Dark zone': 'Behavior_Neuro', 'Impaired equilibrium': 'Behavior_Neuro',
    'F1代鱼Swirl-escsape rate': 'Behavior_Neuro', 'F2代鱼Swirl-escsape rate': 'Behavior_Neuro',
    'Speed': 'Behavior_Neuro', 'Distance': 'Behavior_Neuro', 'total distance': 'Behavior_Neuro',
    'slc6a13': 'Behavior_Neuro', 'slc6a1b': 'Behavior_Neuro', 'gabrg2': 'Behavior_Neuro', 'glsa': 'Behavior_Neuro',
    'grin1b': 'Behavior_Neuro',

    'SOD activity': 'Oxidative_Stress', 'Gpx activity': 'Oxidative_Stress', 'CAT activity': 'Oxidative_Stress',
    'ROS Intensity': 'Oxidative_Stress', 'ROS fluorescence Intensity': 'Oxidative_Stress',
    'T-SOD content': 'Oxidative_Stress', 'MDA content': 'Oxidative_Stress', 'CAT content': 'Oxidative_Stress',
    '8-OHdG content': 'Oxidative_Stress', 'ROS': 'Oxidative_Stress',

    'Responnse(%of control)Interleu': 'Immunotoxicity', 'Responnse(％of control)Interleukin-1β': 'Immunotoxicity',
    'IFN mRNA': 'Immunotoxicity', 'IL-1 β mRNA': 'Immunotoxicity', 'IL-1β mRNA': 'Immunotoxicity',
    'IL-4 mRNA': 'Immunotoxicity', 'IL-21 mRNA': 'Immunotoxicity', 'BAFF mRNA': 'Immunotoxicity',
    'IgD mRNA': 'Immunotoxicity', 'lgD mRNA': 'Immunotoxicity',
    'IgM mRNA': 'Immunotoxicity', 'lgM mRNA': 'Immunotoxicity',
    'IgZ mRNA': 'Immunotoxicity', 'lgZ mRNA': 'Immunotoxicity',
    'P65 transcription factor(relA)': 'Immunotoxicity', 'myd88 mRNA': 'Immunotoxicity', 'TLR2 mRNA': 'Immunotoxicity',
    'TL-1 β': 'Immunotoxicity', 'TL-1β': 'Immunotoxicity',
    'TNaF- α content': 'Immunotoxicity', 'TNaF-α content': 'Immunotoxicity',
    'IL-1 β gene': 'Immunotoxicity', 'IL-1βgene': 'Immunotoxicity',

    'Heart rate': 'Cardiovascular_Angiogenesis', 'Pericardial area': 'Cardiovascular_Angiogenesis',
    'SV-BA distance': 'Cardiovascular_Angiogenesis', 'amhc': 'Cardiovascular_Angiogenesis',
    'vmhc': 'Cardiovascular_Angiogenesis', 'notch1b': 'Cardiovascular_Angiogenesis',
    'bmp4': 'Cardiovascular_Angiogenesis', 'has2': 'Cardiovascular_Angiogenesis',
    'tbx5a': 'Cardiovascular_Angiogenesis', 'nkx2.5': 'Cardiovascular_Angiogenesis',
    'gata4': 'Cardiovascular_Angiogenesis', 'spaw': 'Cardiovascular_Angiogenesis',
    'vegfaa': 'Cardiovascular_Angiogenesis', 'vegfab': 'Cardiovascular_Angiogenesis',
    'vegfc': 'Cardiovascular_Angiogenesis', 'vegfd': 'Cardiovascular_Angiogenesis',
    'flt1': 'Cardiovascular_Angiogenesis', 'flt4': 'Cardiovascular_Angiogenesis', 'kdrl': 'Cardiovascular_Angiogenesis',

    'whole body length(μM)': 'Morphological_Development', 'head length': 'Morphological_Development',
    'Head Width': 'Morphological_Development', 'Spine curvature': 'Morphological_Development',
    'Swim bladder': 'Morphological_Development', 'nodal1': 'Morphological_Development',
    'nodal2': 'Morphological_Development', 'lefty2': 'Morphological_Development', 'pitx2': 'Morphological_Development',
    'prrxla': 'Morphological_Development', 'total abnormality': 'Morphological_Development',
    'hemorrhage': 'Morphological_Development', 'delayed development': 'Morphological_Development',
    'cell height': 'Morphological_Development', 'Yollk sac area': 'Morphological_Development',
    'eye area': 'Morphological_Development', 'Swim bladder area': 'Morphological_Development',
    'malformation(uninflated swim b': 'Morphological_Development', 'malformation(uninflated swim bladder)': 'Morphological_Development',
    'malformation(yollk sac oedemas)': 'Morphological_Development', 'malformation rate(%)': 'Morphological_Development',

    'Erythrocyte fluorescence inten': 'Hematotoxicity', 'Erythrocyte fluorescence intensity': 'Hematotoxicity',
    'hbael siginal': 'Hematotoxicity', 'gata2': 'Hematotoxicity',
    'alas2': 'Hematotoxicity', 'ho-1': 'Hematotoxicity', 'hbae1': 'Hematotoxicity', 'hbael': 'Hematotoxicity', 'scf4': 'Hematotoxicity',

    'p53': 'Apoptosis_Proliferation', 'bax': 'Apoptosis_Proliferation', 'bcl-2': 'Apoptosis_Proliferation',
    'p53 gene': 'Apoptosis_Proliferation', 'relative cell proliferation': 'Apoptosis_Proliferation',
    'relative Luciferase proliferat': 'Apoptosis_Proliferation', 'relative Luciferase proliferation': 'Apoptosis_Proliferation',
    'relative luciferase activity': 'Apoptosis_Proliferation',

    'ATP content': 'Metabolism_Energy', 'ppar- α': 'Metabolism_Energy', 'ppar-α': 'Metabolism_Energy', 'ucp-2': 'Metabolism_Energy',
    'ugt1ab': 'Metabolism_Energy',

    'Survival': 'General_Fitness', 'Female condition factor(CF)': 'General_Fitness',
    'Male condition factor(CF)': 'General_Fitness', 'lethal effects(%)': 'General_Fitness',
    'Survival rate': 'General_Fitness'
}

df['Effect Measurement'] = df['Effect Measurement'].astype(str).str.strip()
df['Bio_Big_Class'] = df['Effect Measurement'].map(endpoint_mapping)

missing_classes = df[df['Bio_Big_Class'].isna()]['Effect Measurement'].unique()
if len(missing_classes) > 0:
    print("【仍然有警告】以下小类没有匹配到大类：")
    print(missing_classes)
else:
    print("【完美】所有小类均已成功归入大类！")

output_file = '../pfas_xingfen/dataset_with_biological_categories.xlsx'
df.to_excel(output_file, index=False)

print(f"\n✅ 数据已成功添加大类，并保存至文件: {output_file}")
