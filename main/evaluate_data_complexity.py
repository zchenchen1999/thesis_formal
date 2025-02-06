# %%
import pandas as pd
import ast
import numpy as np
import math
import matplotlib.pyplot as plt

import scipy.stats as stats
# %%
def str_to_list(df, columns):
    for c in columns:
        df[c] = df[c].apply(lambda x: ast.literal_eval(x))
    return df
def get_response(result, data):
    if (data == "webmd"):
        choice = ['A', 'B', 'C', 'D']
    else:
        choice = ['A', 'B']
    try:
        answer = result.get('response', None)
        if (answer not in choice):
            if (data == "webmd"):
                return "D"
            else:
                return "B"
        return result.get('response', None)
    except:
        if (data == "webmd"):
            return "D"
        else:
            return "B"
# %%
# 定義要分析的 path type
path_type = "default" # consecutive, multi_hop, default
data = "bc5cdr" # webmd, bc5cdr

# 讀取兩個 CSV
if (data == "webmd"):
    preprocess_data = pd.read_csv('/home/zchenchen1999/thesis_formal/main/preprocessed_data/WebMD/WebMD_annotated_v2_exploded_reasoning_train.csv')  # 包含 'drugs', 'symptoms', 'sents'
    preprocess_data_subset = preprocess_data.drop_duplicates(subset=['drugs', 'symptoms', 'sents'])

    result_data = pd.read_csv('/home/zchenchen1999/thesis_formal/main/result/webmd/replace_pronoun/no_reasoning/11_28_train_r_ds_Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf.csv')  # 包含 'drugs', 'symptoms', 'sents', 'text', 'all_drug', 'all_symptom'
    result_data = result_data.drop_duplicates(subset=['drugs', 'symptoms', 'sents'])
else:
    preprocess_data = pd.read_csv('/home/zchenchen1999/thesis_formal/main/preprocessed_data/BC5CDR/BC5CDR_preprocess_train_smallset.csv')  # 包含 'drugs', 'symptoms', 'sents'
    preprocess_data_subset = preprocess_data.drop_duplicates(subset=['drugs', 'symptoms', 'sents'])

    result_data = pd.read_csv('/home/zchenchen1999/thesis_formal/main/result/bc5cdr/replace_pronoun/no_reasoning/11_28_train_r_ds_Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf.csv')  # 包含 'drugs', 'symptoms', 'sents', 'text', 'all_drug', 'all_symptom'
    result_data = result_data.drop_duplicates(subset=['drugs', 'symptoms', 'sents'])

webmd_mapping = {
    'A': 'has_positive_reaction',
    'B': 'has_negative_reaction',
    'C': 'has_adverse_drug_reaction',
    'D': 'none_of_above'
}
bc5cdr_mapping = {
    'A': 'has_adverse_drug_reaction',
    'B': 'none_of_above'
}

webmd_label = ["has_positive_reaction", "has_negative_reaction", "has_adverse_drug_reaction", "none_of_above"]
bc5cdr_label = ["has_adverse_drug_reaction", "none_of_above"]

if (data == "webmd"):
    mapping_dict = webmd_mapping
    label_list = webmd_label
else:
    mapping_dict = bc5cdr_mapping
    label_list = bc5cdr_label

# %%
######################
# 1. 合併＋前處理
######################

# 合併 csv

# 只選擇要合併的特定欄位
if ('sents_replace_pronoun' in result_data.columns):
    preprocess_data_subset = preprocess_data[['drugs', 'symptoms', 'sents', 
        'text', 
        'text_replace_pronoun', 
        'all_drug', 'all_symptom']]  # 指定想要的欄位
else:
    preprocess_data_subset = preprocess_data[['drugs', 'symptoms', 'sents', 
        'text', 
        'text_replace_pronoun', 
        'sents_replace_pronoun', 
        'all_drug', 'all_symptom']]  # 指定想要的欄位

# 使用 merge 進行水平合併，根據 'drugs', 'symptoms', 'sents' 進行匹配
merged_df = pd.merge(result_data, preprocess_data_subset, on=['drugs', 'symptoms', 'sents'], how='inner')
# merged_df = result_data

# 轉換型態
merged_df = str_to_list(merged_df, ['sents','sents_replace_pronoun','drugs','symptoms',
                      'original_result','consecutive_result',
                      'multi_hop_result','default_result',
                      'all_drug', 'all_symptom'])

# 只取某個 path type 有的資料
merged_df = merged_df[merged_df[f'{path_type}_result'] != {}]

# 分配 answer
merged_df[f'{path_type}_answer'] = merged_df[f'{path_type}_result'].apply(lambda x: mapping_dict[get_response(x,data)])
merged_df[f'original_answer'] = merged_df[f'original_result'].apply(lambda x: mapping_dict[get_response(x,data)])

# # 檢查結果
merged_df

# %%
######################
# 2. 計算文本複雜程度
######################
# 計算文本的複雜程度

# 1. 計算出每筆資料的文本長度（計算 text 長度）
# merged_df['text_length'] = merged_df['text'].apply(len)
merged_df['text_length'] = merged_df['text_replace_pronoun'].apply(len)

# 2. 計算 entity pair 之間的句子間隔數（使用 drugs/symptoms的第 0 項去 sents 搜尋相隔幾句話）
def calculate_sentence_gap(row):
    try:
        # 解析 'sents' 欄位
        sents = row['sents_replace_pronoun']
        drug_0 = row['drugs'][0]
        symptom_0 = row['symptoms'][0]
        
        # 1️⃣ 找出 drug_0 在 sents 中的所有索引
        drug_indices = [i for i, sent in enumerate(sents) if drug_0.lower() in sent.lower()]
        
        # 2️⃣ 找出 symptom_0 在 sents 中的所有索引
        symptom_indices = [i for i, sent in enumerate(sents) if symptom_0.lower() in sent.lower()]
        
        # 如果 drug 和 symptom 都有出現，計算最小距離
        if drug_indices and symptom_indices:
            # 3️⃣ 計算所有 drug 和 symptom 之間的跨句距離
            min_distance = min(abs(d - s) for d in drug_indices for s in symptom_indices)
            return min_distance
        else:
            return None  # 如果 drug 或 symptom 沒出現，回傳 None
    except Exception as e:
        return None

merged_df['sentence_gap'] = merged_df.apply(calculate_sentence_gap, axis=1)

# 3. 計算文本的 entity 組合數（all_drug 數量 * all_symptom 數量）
def calculate_entity_combinations(row):
    try:
        all_drug_count = len(row['all_drug'])
        all_symptom_count = len(row['all_symptom'])
        return all_drug_count * all_symptom_count
    except Exception as e:
        return None

merged_df['entity_combinations'] = merged_df.apply(calculate_entity_combinations, axis=1)

merged_df['drug_len'] = merged_df.apply(lambda x: len(x['all_drug']), axis=1)
merged_df['symptom_len'] = merged_df.apply(lambda x: len(x['all_symptom']), axis=1)
merged_df.head()

# %%
#############################################################################
# 4. 計算 original 正確數量、consecutive 正確數量、original 與 consecutive 皆正確數量
#############################################################################

# 計算 original 正確的數量（original_result 與 ground_truth 相同）
merged_df['original_correct'] = merged_df['original_answer'] == merged_df['ground_truth']

# 計算 consecutive 正確的數量（consecutive_result 與 ground_truth 相同）
merged_df[f'{path_type}_correct'] = merged_df[f'{path_type}_answer'] == merged_df['ground_truth']

# 計算 original 和 consecutive 皆正確的數量
merged_df['both_correct'] = merged_df['original_correct'] & merged_df[f'{path_type}_correct']

# 計算 original 正確的數量
original_correct_count = merged_df['original_correct'].sum()

# 計算 consecutive 正確的數量
path_correct_count = merged_df[f'{path_type}_correct'].sum()

# 計算 original 和 consecutive 皆正確的數量
both_correct_count = merged_df['both_correct'].sum()

# 1. 計算 original 正確中有多少比例 consecutive 也正確
if original_correct_count > 0:
    ratio_path_among_original = both_correct_count / original_correct_count
else:
    ratio_path_among_original = None

# 2. 計算 consecutive 正確中有多少比例 original 也正確
if path_correct_count > 0:
    ratio_original_among_path = both_correct_count / path_correct_count
else:
    ratio_original_among_path = None

# 顯示計算結果
calculation_results = {
    'original_correct_count': original_correct_count,
    f'{path_type}_correct_count': path_correct_count,
    'both_correct_count': both_correct_count,
    'ratio_path_among_original': ratio_path_among_original,
    'ratio_original_among_path': ratio_original_among_path
}

calculation_results
# merged_df.head().to_csv('test.csv', index=False)

# %%
######################
# 5. 混淆矩陣
######################

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 定義類別標籤
labels = label_list

# 計算 Original 與 Ground Truth 的混淆矩陣
cm_original = confusion_matrix(merged_df["ground_truth"], merged_df["original_answer"], labels=labels)

# 計算 path 與 Ground Truth 的混淆矩陣
cm_path = confusion_matrix(merged_df["ground_truth"], merged_df[f"{path_type}_answer"], labels=labels)

# 繪製 Original 混淆矩陣
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.heatmap(cm_original, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted (Original)")
plt.ylabel("Ground Truth")
plt.title("Confusion Matrix: Original")

# 繪製 path 混淆矩陣
plt.subplot(1, 2, 2)
sns.heatmap(cm_path, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted (path)")
plt.ylabel("Ground Truth")
plt.title(f"Confusion Matrix: Add {path_type} path")

plt.tight_layout()
plt.show()

# %%
################################################################################################
# 4. 分別過濾出 (original 與 consecutive 皆正確) 與 (只有 consecutive 正確) 的資料
#  1. 計算文本長度統計（例如：平均....）
#  2. 計算entity pair 之間的句子間隔數統計（例如：平均....）
#  3. 計算文本的 entity 組合數（drug 數量 * symptom數量）統計（例如：平均....）
################################################################################################

only_original_correct_df = merged_df[merged_df['original_correct'] & ~merged_df[f'{path_type}_correct']]

only_original_correct_stats = {
    'text_length': only_original_correct_df['text_length'].describe(),
    'sentence_gap': only_original_correct_df['sentence_gap'].describe(),
    'entity_combinations': only_original_correct_df['entity_combinations'].describe()
}

print("\n\n只有 original 正確")
print(only_original_correct_stats)

only_path_correct_df = merged_df[~merged_df['original_correct'] & merged_df[f'{path_type}_correct']]

only_path_correct_stats = {
    'text_length': only_path_correct_df['text_length'].describe(),
    'sentence_gap': only_path_correct_df['sentence_gap'].describe(),
    'entity_combinations': only_path_correct_df['entity_combinations'].describe()
}
print(f"\n\n只有 {path_type} 正確")
print(only_path_correct_stats)


both_correct_df = merged_df[merged_df['both_correct']]

both_correct_stats = {
    'text_length': both_correct_df['text_length'].describe(),
    'sentence_gap': both_correct_df['sentence_gap'].describe(),
    'entity_combinations': both_correct_df['entity_combinations'].describe()
}
print(f"\n\n兩者都正確")
print(both_correct_stats)

# both_correct_df

original_correct_df = merged_df[merged_df['original_correct']]
path_correct_df = merged_df[merged_df[f'{path_type}_correct']]

both_correct_df = original_correct_df
# only_path_correct_df = path_correct_df

# %%
#################
# Text length
#################

both_correct_lengths = both_correct_df["text_length"].to_list()
path_correct_only_lengths = only_path_correct_df["text_length"]

# 繪製密度圖
sns.kdeplot(both_correct_lengths, label="Original Correct", fill=True)
sns.kdeplot(path_correct_only_lengths, label="Path Correct Only", fill=True)

# 設定 x 軸標籤
plt.xlabel("Text Length") 

# 進行 Levene 檢定
levene_stat, levene_p_value = stats.levene(both_correct_lengths, path_correct_only_lengths)

# 判斷是否使用 Welch's t-test
if levene_p_value < 0.05:
    # 變異數不相等，使用 Welch's t-test
    t_stat, p_value = stats.ttest_ind(path_correct_only_lengths, both_correct_lengths, equal_var=False)
    test_type = "Welch's t-test (unequal variances)"
else:
    # 變異數相等，使用標準 t-test
    t_stat, p_value = stats.ttest_ind(path_correct_only_lengths, both_correct_lengths, equal_var=True)
    test_type = "Standard t-test (equal variances)"

# 在圖框內的 Legend 下方標註 t-test 結果
plt.text(
    x=0.756, y=0.75,  # 相對座標，調整到 Legend 下方
    s=f"t-stat: {t_stat:.2f}\np-value: {p_value:.4f}",
    transform=plt.gca().transAxes,  # 使用相對座標
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
)

# 加標題和圖例
plt.title("Density Plot of Text Length")
plt.legend()
plt.show()

# 顯示結果
levene_stat, levene_p_value, test_type, t_stat, p_value
# %%
#################
# Sentence gap
#################
both_correct_sentence_gap = both_correct_df["sentence_gap"].to_list()
path_correct_only_sentence_gap = only_path_correct_df["sentence_gap"]

# 繪製密度圖
sns.kdeplot(both_correct_sentence_gap, label="Original Correct", fill=True)
sns.kdeplot(path_correct_only_sentence_gap, label="Path Correct Only", fill=True)

# 設定 x 軸標籤
plt.xlabel("Entity Pair Distance (sentences)") 

# 進行 Levene 檢定
levene_stat, levene_p_value = stats.levene(both_correct_sentence_gap, path_correct_only_sentence_gap)

# 判斷是否使用 Welch's t-test
if levene_p_value < 0.05:
    # 變異數不相等，使用 Welch's t-test
    t_stat, p_value = stats.ttest_ind(path_correct_only_sentence_gap, both_correct_sentence_gap, equal_var=False)
    test_type = "Welch's t-test (unequal variances)"
else:
    # 變異數相等，使用標準 t-test
    t_stat, p_value = stats.ttest_ind(path_correct_only_sentence_gap, both_correct_sentence_gap, equal_var=True)
    test_type = "Standard t-test (equal variances)"

# 在圖框內的 Legend 下方標註 t-test 結果
plt.text(
    x=0.756, y=0.75,  # 相對座標，調整到 Legend 下方
    s=f"t-stat: {t_stat:.2f}\np-value: {p_value:.4f}",
    transform=plt.gca().transAxes,  # 使用相對座標
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
)

# 加標題和圖例
plt.title("Density Plot of Entity Pair Distance")
plt.legend()
plt.show()

# 顯示結果
levene_stat, levene_p_value, test_type, t_stat, p_value
# %%
####################
# Entity Combination
####################
both_correct_entity_combinations = both_correct_df["entity_combinations"].to_list()
path_correct_only_entity_combinations = only_path_correct_df["entity_combinations"]

# 繪製密度圖
sns.kdeplot(both_correct_entity_combinations, label="Original Correct", fill=True)
sns.kdeplot(path_correct_only_entity_combinations, label="Path Correct Only", fill=True)

# 設定 x 軸標籤
plt.xlabel("Number of Entity Combinations (in document)") 

# 進行 Levene 檢定
levene_stat, levene_p_value = stats.levene(both_correct_entity_combinations, path_correct_only_entity_combinations)

# 判斷是否使用 Welch's t-test
if levene_p_value < 0.05:
    # 變異數不相等，使用 Welch's t-test
    t_stat, p_value = stats.ttest_ind(path_correct_only_entity_combinations, both_correct_entity_combinations, equal_var=False)
    test_type = "Welch's t-test (unequal variances)"
else:
    # 變異數相等，使用標準 t-test
    t_stat, p_value = stats.ttest_ind(path_correct_only_entity_combinations, both_correct_entity_combinations, equal_var=True)
    test_type = "Standard t-test (equal variances)"

# 在圖框內的 Legend 下方標註 t-test 結果
plt.text(
    x=0.756, y=0.75,  # 相對座標，調整到 Legend 下方
    s=f"t-stat: {t_stat:.2f}\np-value: {p_value:.4f}",
    transform=plt.gca().transAxes,  # 使用相對座標
    fontsize=10,
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black')
)

# 加標題和圖例
plt.title("Density Plot of Entity Combinations")
plt.legend()
plt.show()

# 顯示結果
levene_stat, levene_p_value, test_type, t_stat, p_value

# %%
