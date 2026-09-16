"""Create the modelling dataset from endpoint-classified curated records."""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import warnings
import joblib
import json

warnings.filterwarnings('ignore')

col_target = 'Hormesis效应判定'
col_big_class = 'Bio_Big_Class'
col_small_class = 'Effect Measurement'

input_file = '../pfas_xingfen/dataset_with_biological_categories.xlsx'
print(f"1. 正在读取原始数据: {input_file}")
df = pd.read_excel(input_file)

# Canonical SMILES strings used for RDKit descriptor calculation.
pfas_smiles_dict = {
    'PFOA': 'C(=O)(C(C(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)(F)F)O',
    'PFOS': 'C(C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F',
    'PFHxA': 'C(=O)(C(C(C(C(C(F)(F)F)(F)F)(F)F)(F)F)(F)F)O',
    'PFBA': 'OC(=O)C(F)(F)C(F)(F)C(F)(F)F',
    'PFBS': 'C(C(C(F)(F)S(=O)(=O)O)(F)F)(C(F)(F)F)(F)F',
    'PFHxS': 'C(C(C(C(F)(F)S(=O)(=O)O)(F)F)(F)F)(C(C(F)(F)F)(F)F)(F)F',
    'GenX': 'C(=O)(C(C(F)(F)F)(OC(C(C(F)(F)F)(F)F)(F)F)F)[O-]',
    'HFPO-DA': 'C(=O)(C(C(F)(F)F)(OC(C(C(F)(F)F)(F)F)(F)F)F)[O-]',
    'HFPO-TA': 'C(O)(=O)C(F)(OC(F)(F)C(F)(OC(F)(F)C(F)(F)C(F)(F)F)C(F)(F)F)C(F)(F)F',
    'HFPO-TeA': 'C(=O)(C(C(F)(F)F)(OC(C(C(F)(F)F)(F)F)(F)F)F)O',
    'PFOSA': 'C(C(C(C(C(F)(F)S(=O)(=O)N)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F',
    'PFECHS': 'C1(C(C(C(C(C1(F)F)(F)F)(F)S(=O)(=O)O)(F)F)(F)F)(C(C(F)(F)F)(F)F)F',
    'PFO3OA': 'C(=O)(C(OC(OC(OC(F)(F)F)(F)F)(F)F)(F)F)O'
}
df['Chemical Name'] = df['Chemical Name'].astype(str).str.strip()
df['SMILES'] = df['Chemical Name'].map(pfas_smiles_dict)

print("2. 正在计算 RDKit 核心分子描述符...")
def calc_rdkit_descriptors(smiles):
    if pd.isna(smiles): return {}
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return {}
        return Descriptors.CalcMolDescriptors(mol)
    except:
        return {}

features_df = pd.DataFrame(df['SMILES'].apply(calc_rdkit_descriptors).tolist())
final_df = pd.concat([df, features_df], axis=1)

columns_to_drop = ['DOI号', 'Author and year', 'Control', 'Chemical Name', 'SMILES',
                   'Conc 1（μg/L）', 'Conc 2（μg/L）', 'Conc 3（μg/L）', 'Conc 4（μg/L）', 'Conc 5（μg/L）',
                   '终点值1', '终点值2', '终点值3', '终点值4', '终点值5', '终点值6']
cols_actually_dropped = [c for c in columns_to_drop if c in final_df.columns]
final_df = final_df.drop(columns=cols_actually_dropped)

print("\n--- 进入机器学习特征预处理阶段 ---")

print("3. 正在转换预测标签为数字...")
final_df[col_target] = final_df[col_target].map({'Y': 1, 'N': 0, '是': 1, '否': 0, 1: 1, 0: 0})
y = final_df[col_target].astype(int)
X = final_df.drop(columns=[col_target])

print("4. 正在对大类和小类进行数值编码...")

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [col_big_class, col_small_class]
X[num_cols] = X[num_cols].fillna(X[num_cols].median())
X[cat_cols] = X[cat_cols].fillna(X[cat_cols].mode().iloc[0])

# Smoothed leave-one-out encoding for Effect Measurement (m = 2.0).
print("   - 正在执行平滑留一法计算 (m=2.0)...")
global_mean = y.mean()
category_sum = y.groupby(X[col_small_class]).transform('sum')
category_count = y.groupby(X[col_small_class]).transform('count')

m = 2.0
loo_smooth = (category_sum - y + m * global_mean) / (category_count - 1 + m)

X[col_small_class + '_LOO'] = loo_smooth.astype(float)
X = X.drop(columns=[col_small_class])

X = pd.get_dummies(X, columns=[col_big_class], drop_first=False, dtype=int, prefix='', prefix_sep='')
print("5. 正在清理低方差特征...")
rdkit_cols = [c for c in num_cols if c in X.columns]

# Retain RDKit descriptors with variance greater than 0.01.
selector = VarianceThreshold(threshold=0.01)
selector.fit(X[rdkit_cols])
kept_rdkit_cols = X[rdkit_cols].columns[selector.get_support()]
X_rdkit = X[kept_rdkit_cols]

other_cols = [c for c in X.columns if c not in rdkit_cols]
X_clean = pd.concat([X_rdkit, X[other_cols]], axis=1)

print("6. 执行数值标准化...")
scaler = StandardScaler()
X_clean[kept_rdkit_cols] = scaler.fit_transform(X_clean[kept_rdkit_cols])

X_clean = X_clean.astype(float)
X_clean['Target_Hormesis'] = y.values.astype(int)

output_file = '../pfas_xingfen/modeling_dataset.xlsx'
X_clean.to_excel(output_file, index=False)

print(f"\n✅ 处理完成！")
print(f"✅ TRUE/FALSE 已全部转为 0/1")
print(f"✅ 共线性筛选已移除，保留了更多特征")
print(f"✅ 文件已保存至: {output_file}")

print("7. 正在保存问答系统所需的预处理上下文 (Scaler, Mappings, Feature Lists)...")

joblib.dump(scaler, '../pfas_xingfen/main_model_results/scaler.pkl')

# Save preprocessing artifacts for future external prediction.
loo_mapping = {}
for idx, row in df.iterrows():
    small_cls = row[col_small_class]
    loo_val = X.loc[idx, col_small_class + '_LOO']
    loo_mapping[small_cls] = loo_val

loo_mapping['UNKNOWN_DEFAULT'] = global_mean

with open('../pfas_xingfen/one-hot/loo_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(loo_mapping, f, ensure_ascii=False, indent=4)

final_features = X_clean.drop(columns=['Target_Hormesis']).columns.tolist()
with open('../pfas_xingfen/one-hot/final_features.json', 'w', encoding='utf-8') as f:
    json.dump(final_features, f, ensure_ascii=False, indent=4)

with open('../pfas_xingfen/one-hot/kept_rdkit_cols.json', 'w', encoding='utf-8') as f:
    json.dump(list(kept_rdkit_cols), f, ensure_ascii=False, indent=4)

print("✅ 所有预测依赖文件已保存完毕！未来的问答系统可以直接调用它们。")
