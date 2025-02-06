# %%
import difflib
import pandas as pd
import ast
import json
import re
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import classification_report
from collections import Counter
from bstrap import bootstrap, boostrapping_CI
from statsmodels.stats.contingency_tables import mcnemar

import utils
# %%

# 定义映射函数
def map_relation(text):
    # 定义标签和关键词的对应关系
    mapping = {
        'has_adverse_drug_reaction': ['has_adverse_drug_reaction'],
        'has_positive_reaction': ['has_positive_reaction'],
        'has_negative_reaction': ['has_negative_reaction'],
        'none_of_above': ['none_of_above']
    }
    
    # 去掉文本中的特殊符号
    # clean_text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    
    # 遍历每个标签，检查关键词是否在文本中
    for label, keywords in mapping.items():
        for keyword in keywords:
            if keyword in text:
                return label
    
    print(text)
    # 如果没有匹配到，返回 'none_of_above'
    return 'none_of_above'

def json_to_list(json_string):
    list_ressult = []
    data = json.loads(json_string)
    for key, value in data.items():
        # 分割 key 獲取藥物名稱和症狀
        head, tail = key.split('_', 1)

        # mapping 到 label
        value = map_relation(value)

        # 創建新的列表並添加到結果中
        if (value != "none_of_above"):
            list_ressult.append([head, value, tail])
    return list_ressult

# %%
def preprocess(df, column_name_list):
    for n in column_name_list:
        df[n] = df[n].apply(utils.json_to_dict)
    return df

def get_probability_for_response(logprobs_dict, target_relation):
    try:
        for entry in logprobs_dict['top_logprobs']:
            if entry['token'].strip() == target_relation:
                return np.exp(entry['logprob'])
    except:
        print(logprobs_dict)
    return 0.0

# 定義一個函數來查找並相加指定 token 的 logprob
# 計算單個類別在不同 path 結果的加權平均
# paths_weights: 多個 tuple (path 結果, 權重)
def get_probability_for_relation(*paths_weights, target_relation="A"):
    probability = 0.0
    for item in paths_weights:
        probability += get_probability_for_response(item[0], target_relation) * item[1]
    return probability

# 計算多個類別的綜合結果  ==> 與 threshold 比較
# paths_weights: 多個 tuple (path 結果, 權重)
def get_combine_result(*paths_weights, parm_a, parm_b):
    mapping = {
        'A': 'has_positive_reaction',
        'B': 'has_negative_reaction',
        'C': 'has_adverse_drug_reaction',
        'D': 'none_of_above'
    }

    A_probability = get_probability_for_relation(*paths_weights, target_relation="A")
    B_probability = get_probability_for_relation(*paths_weights, target_relation="B")
    C_probability = get_probability_for_relation(*paths_weights, target_relation="C")
    D_probability = get_probability_for_relation(*paths_weights, target_relation="D")

    classes = ["A","B","C","D"]
    probabilities = [A_probability, B_probability, C_probability, D_probability]
    max_probability = np.max(probabilities)
    max_index = np.argmax(probabilities)

    # print(f"A: {A_probability}, B: {B_probability}, C: {C_probability}, D: {D_probability}")
    # print(f"最大機率值是: {max_probability}")
    # print(f"最大機率值的索引是: {classes[max_index]}")

    if (classes[max_index] == "D"):
        # print("答案是 D")
        return mapping["D"]
    # 不是 non_of_above
    else:
        # max_probability 越高代表越確定是那個類別， tau 越小，即分配到 none_of_above 機會比較小
        tau = parm_a - parm_b * np.log(max_probability) 
        # tau = parm_a - parm_b * (max_probability ** 2)
        # if (abs(max_probability - D_probability) > parm_a): # 新增
        if (max_probability - D_probability < tau) : # 新增
            # print("答案是 D")
            return mapping["D"]
        else:
            # print(f"答案是 {classes[max_index]}")
            return mapping[classes[max_index]]

    # if (classes[max_index] == "D"):
    #     # print("答案是 D")
    #     return mapping["D"]
    # # 不是 non_of_above
    # else:
    #     if (abs(max_probability - D_probability) < parm_a):
    #         return mapping["D"]
    #     else:
    #         return mapping[classes[max_index]]

# 單一 LLM response 內容
def get_response_result(target_dict):
    try:
        result = target_dict['response'].strip()
    except:
        # print(target_dict)
        pass
    mapping = {
        'A': 'has_positive_reaction',
        'B': 'has_negative_reaction',
        'C': 'has_adverse_drug_reaction',
        'D': 'none_of_above'
    }
    try:
        # 取第一個字（通常是 relation 類別）
        return mapping[result[0]]
    except:
        print(result[0])
        return 'none_of_above'

# 單一 LLM response 內容（取最高機率的類別）
def get_highest_pro_result(target_dict, data):
    if (len(target_dict) < 1):
        return "-"
    
    if (data == "webmd"):
        mapping = {
            'A': 'has_positive_reaction',
            'B': 'has_negative_reaction',
            'C': 'has_adverse_drug_reaction',
            'D': 'none_of_above'
        }
    else:
        mapping = {
            'A': 'has_adverse_drug_reaction',
            'B': 'none_of_above'
        }
    relation_labels = ['A', 'B', 'C', 'D']
    try:
        for entry in target_dict['top_logprobs']:
            if entry['token'].strip() in relation_labels:
                return mapping[entry['token'].strip()]
    except:
        print(target_dict)
        return 'none_of_above'
    return 

# 單一 open LLM response 內容（直接取 response key）
def get_open_llm_result(target_dict, data):
    if (len(target_dict) < 1):
        return "-"
    
    if (data == "webmd"):
        mapping = {
            'A': 'has_positive_reaction',
            'B': 'has_negative_reaction',
            'C': 'has_adverse_drug_reaction',
            'D': 'none_of_above'
        }
    else:
        mapping = {
            'A': 'has_adverse_drug_reaction',
            'B': 'none_of_above'
        }
    relation_labels = ['A', 'B', 'C', 'D']
    try:
        return mapping[target_dict['response']]
    except:
        print(target_dict)
        return 'none_of_above'


# %%
def get_self_consistency_result(*results):
    responses_with_prob = []
    
    for i, result in enumerate(results):
        value = get_response_result(result)
        prob = get_probability_for_response(result, result['response'].strip())
        responses_with_prob.append((value, prob))

    counter = Counter([response for response, _ in responses_with_prob])
    most_common_answers = counter.most_common()
    print(most_common_answers)
    max_count = most_common_answers[0][1]

    # 找到所有出現次數最多的答案
    candidates = [response for response, count in most_common_answers if count == max_count]
    # 找出這些候選答案中置信度最高的那個
    best_answer = None
    best_probability = 0.0
    
    for response, probability in responses_with_prob:
        if response in candidates and probability >= best_probability:
            best_answer = response
            best_probability = probability
    return best_answer

# 定義一個函數來決定 priority_result 的值
def determine_priority(row):
    if row['consecutive_result'] != '{}':
        return row['consecutive_result']
    elif row['multi_hop_result'] != '{}':
        return row['multi_hop_result']
    elif row['default_result'] != '{}':
        return row['default_result']
    else:
        return '{}'

def load_path_data(file_path, evaluate_paths=[], subset=True):
    df = pd.read_csv(file_path)
    # # 讀取第一個 CSV 檔案
    # csv1 = pd.read_csv('/home/zchenchen1999/thesis_formal/main/result/webmd/replace_pronoun/no_reasoning/shots/1_31_train_r_ds_Meta-Llama-3.1-8B-Instruct-Doctor.Q4_K_M.gguf_1shots.csv')
    # # csv1 = pd.read_csv('/home/zchenchen1999/thesis_formal/main/result/bc5cdr/replace_pronoun/no_reasoning/shots/1_26_train_r_ds_Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf_1shots.csv')

    # # 讀取第二個 CSV 檔案
    # csv2 = pd.read_csv('/home/zchenchen1999/thesis_formal/main/result/webmd/replace_pronoun/no_reasoning/1_31_train_r_ds_Meta-Llama-3.1-8B-Instruct-Doctor.Q4_K_M.gguf_with_intersection_union.csv')
    # # csv2 = pd.read_csv('/home/zchenchen1999/thesis_formal/main/result/bc5cdr/replace_pronoun/no_reasoning/1_31_train_r_ds_Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf_with_intersection_union.csv')

    # # 按照 'sents', 'drugs', 'symptoms' 作為基準合併
    # df = pd.merge(
    #     csv1,
    #     csv2[['Unnamed: 0','sents','sents_replace_pronoun', 'drugs', 'symptoms', 'intersection_result', 'union_result']],
    #     on=['Unnamed: 0','sents','sents_replace_pronoun', 'drugs', 'symptoms'],
    #     how='left'  # 使用左連接以保留第一個 CSV 中的所有資料
    # )

    # 取得優先級結果
    df['priority_result'] = df.apply(determine_priority, axis=1)

    if (subset):
        df = df[df[evaluate_paths[0]] != '{}'] # 過濾掉沒結果的row
    else:
        pass
    df = preprocess(df, evaluate_paths+['original_result'] if ('original_result' not in evaluate_paths) else evaluate_paths)

    print(evaluate_paths[0])
    print(df.shape)
    return df

def get_predictions(df, path="consecutive", dataset_type="webmd", llm_type="gpt", subset=True):
    if (llm_type == "gpt"):
        df['predict_original'] = df.apply(lambda x: get_highest_pro_result(x['original_result'], dataset_type), axis=1)
        df['predict_compare'] = df.apply(lambda x: get_highest_pro_result(x['original_result'], dataset_type) if x[path] == {} else get_highest_pro_result(x[path], dataset_type), axis=1)
    else:
        df['predict_original'] = df.apply(lambda x: get_open_llm_result(x['original_result'], dataset_type), axis=1)
        df['predict_compare'] = df.apply(lambda x: get_open_llm_result(x['original_result'], dataset_type) if x[path] == {} else get_open_llm_result(x[path], dataset_type), axis=1)
    return df



# df = pd.read_csv('/home/zchenchen1999/MindMap-DDI/data/Web_MD/result/self_consistency/all_train_.csv')
# df = preprocess(df, ['result_1', 'result_2', 'result_3'])
# # a = get_self_consistency_result(df.loc[2, 'result_1'], df.loc[2, 'result_2'], df.loc[2, 'result_3'])
# df['predict'] = df.apply(lambda x: get_self_consistency_result(x['result_1'], x['result_2'], x['result_3']), axis=1)
# ground_truth =  df['ground_truth'].to_list()
# predict = df['predict'].to_list()

# # 明確指定 labels 和 target_names 的順序
# labels = ['none_of_above', 'has_adverse_drug_reaction', 'has_positive_reaction', 'has_negative_reaction']
# target_names = ['none_of_above', 'has_adverse_drug_reaction', 'has_positive_reaction', 'has_negative_reaction']
# report = classification_report(ground_truth, predict, labels=labels, target_names=target_names, output_dict=True)
# report
# %%
def print_classification_report(file_path, dataset_type, llm_type, evaluate_paths):
    #################################################
    #          衡量- LLM evidence 比較 - 總表         #
    #################################################
    print("""
###########
#   總表   #
###########
""")
    df = load_path_data(file_path=file_path, evaluate_paths=evaluate_paths, subset=False)
    df = get_predictions(df, path=evaluate_paths[0], dataset_type=dataset_type, llm_type=llm_type, subset=False)

    gt = df['ground_truth'].to_list()
    compare = df['predict_compare'].to_list()

    report = classification_report(gt, compare, output_dict=True)
    print(json.dumps(report, indent=4))

    #################################################
    #        衡量- LLM evidence 比較 - 各自比較        #
    #################################################
    print("""
###########
# 各自比較  #
###########
""")

    df = load_path_data(file_path=file_path, evaluate_paths=evaluate_paths, subset=True)
    df = get_predictions(df, path=evaluate_paths[0], dataset_type=dataset_type, llm_type=llm_type, subset=False)

    gt = df['ground_truth'].to_list()
    original = df['predict_original'].to_list()
    compare = df['predict_compare'].to_list()

    report_original = classification_report(gt, original, output_dict=True)
    report_compare = classification_report(gt, compare, output_dict=True)


    print("original")
    print(json.dumps(report_original, indent=4))
    print("\n=====================\n")
    print(evaluate_paths[0])
    print(json.dumps(report_compare, indent=4))

# ===========參數設置============
file_path = '/home/zchenchen1999/thesis_formal/main/result/webmd/replace_pronoun/no_reasoning/11_28_train_r_ds_Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf.csv'
# webmd, bc5cdr
dataset_type = "webmd"
# gpt, openllm
llm_type = "openllm"
# 想要評估的欄位  [original_result, consecutive_result, multi_hop_result, default_result, intersection_result, union_result, priority_result, umls_result
evaluate_paths = ['priority_result']
# ==============================
print_classification_report(file_path=file_path, dataset_type=dataset_type, llm_type=llm_type, evaluate_paths=evaluate_paths)
# %%
############################
# boostrap
############################
def print_bootstrap(file_path, dataset_type="webmd", llm_type="gpt", evaluate_paths=['consecutive_result'], subset=True, nbr_runs=100):
    ############################
    # 1. implement metric
    ############################
    def compute_f1(df):
        ground_truth = df['ground_truth'].to_list()
        predict = df['predictions'].to_list()
        report = classification_report(ground_truth, predict, output_dict=True)
        return report['macro avg']['f1-score']
    def compute_precision(df):
        ground_truth = df['ground_truth'].to_list()
        predict = df['predictions'].to_list()
        report = classification_report(ground_truth, predict, output_dict=True)
        return report['macro avg']['precision']
    def compute_recall(df):
        ground_truth = df['ground_truth'].to_list()
        predict = df['predictions'].to_list()
        report = classification_report(ground_truth, predict, output_dict=True)
        return report['macro avg']['recall']
    def compute_accuracy(df):
        ground_truth = df['ground_truth'].to_list()
        predict = df['predictions'].to_list()
        report = classification_report(ground_truth, predict, output_dict=True)
        return report['accuracy']
    metric = compute_recall
    
    ############################
    # 2. load data
    ############################
    df = load_path_data(file_path=file_path, evaluate_paths=evaluate_paths, subset=subset)
    df = get_predictions(df, path=evaluate_paths[0], dataset_type=dataset_type, llm_type=llm_type, subset=subset)

    print(df.shape)
    ############################
    # 3. reformat data to a single pandas dataframe per method with standardized column names
    ############################
    data_method1 = df[["ground_truth", "predict_original"]]
    data_method1 = data_method1.rename(columns={"predict_original": "predictions"})
    data_method2 = df[["ground_truth", "predict_compare"]]
    data_method2 = data_method2.rename(columns={"predict_compare": "predictions"})

    ############################
     # 4. compare method 1 and 2 (same code as example 1)
    ############################
    stats_method1, stats_method2, p_value = bootstrap(metric, data_method1, data_method2, nbr_runs=nbr_runs, verbose=False)
    print(stats_method1)
    print(stats_method2)
    print(p_value)

    # # compute CI and mean for each method separately (same code as example 1)
    # stats_method1 = boostrapping_CI(metric, data_method1, nbr_runs=nbr_runs, verbose=False)
    # stats_method2 = boostrapping_CI(metric, data_method2, nbr_runs=nbr_runs, verbose=False)
    # print(stats_method1)
    # print(stats_method2)

# %%
def print_mcNemars(file_path, dataset_type="webmd", llm_type="gpt", evaluate_paths=['consecutive_result'], subset=True):

    df = load_path_data(file_path=file_path, evaluate_paths=evaluate_paths, subset=subset)
    df = get_predictions(df, path=evaluate_paths[0], dataset_type=dataset_type, llm_type=llm_type, subset=subset)

    # 假設 ground_truth、original_result、consecutive_result 是 NumPy 陣列
    ground_truth = df['ground_truth'].to_list()          # Ground truth labels
    original = df['predict_original'].to_list()       # Predictions from original 
    compare = df['predict_compare'].to_list()    # Predictions from consecutive 

    # 初始化 a, b, c, d
    a = b = c = d = 0

    # 計算 a, b, c, d
    for gt, ori, com in zip(ground_truth, original, compare):
        if ori == gt and com == gt:
            a += 1  # 兩個模型都正確
        elif ori == gt and com != gt:
            b += 1  # original 正確, compare 錯誤
        elif ori != gt and com == gt:
            c += 1  # original 錯誤, compare 正確
        else:
            d += 1  # 兩個模型都錯誤

    # 建立 Contingency Table
    contingency_table = [[a, b],
                        [c, d]]
    print(contingency_table)
    # calculate mcnemar test
    result = mcnemar(contingency_table, exact=True)
    # summarize the finding
    print('statistic=%.3f, p-value=%.3f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
        print('Same proportions of errors (fail to reject H0)')
    else:
        print('Different proportions of errors (reject H0)')

# ===========參數設置============
file_path = '/home/zchenchen1999/thesis_formal/main/result/bc5cdr/replace_pronoun/no_reasoning/11_28_train_r_ds_Meta-Llama-3.1-8B-Instruct-Doctor.Q4_K_M.gguf.csv'
# webmd, bc5cdr
dataset_type = "bc5cdr"
# gpt, openllm
llm_type = "openllm"
# 想要評估的欄位  [original_result, consecutive_result, multi_hop_result, default_result, umls_result, priority_result
evaluate_paths = ["priority_result"]
# 總表或各自比較表 True, False
subset = False
# ==============================
print_mcNemars(file_path=file_path, dataset_type=dataset_type, 
               llm_type=llm_type, evaluate_paths=evaluate_paths, 
               subset=subset)
# %%
# def main():
#     for path in ['original_result', 'consecutive_result', 'multi_hop_result', 'default_result']:
#         ############################
#         # WebMD
#         ############################
#         # ===========參數設置============
#         file_path = '/home/zchenchen1999/thesis_formal/main/result/webmd/replace_pronoun/no_reasoning/11_28_train_r_ds_Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf.csv'
#         # webmd, bc5cdr
#         dataset_type = "webmd"
#         # gpt, openllm
#         llm_type = "openllm"
#         # 想要評估的欄位  [original_result, consecutive_result, multi_hop_result, default_result, umls_result
#         evaluate_paths = [path]
#         # 總表或各自比較表 True, False
#         subset = True
#         # 算幾次指標 100, 1000, 10000
#         nbr_runs = 10000
#         # ==============================
#         print(f"""
#         資料集：{dataset_type}
#         LLM: {llm_type}
#         檔案：{file_path}
#         各自比較：{subset}""")
#         print_bootstrap(file_path=file_path, dataset_type=dataset_type, 
#                         llm_type=llm_type, evaluate_paths=evaluate_paths, 
#                         subset=subset, nbr_runs=nbr_runs)

#         # ===========參數設置============
#         file_path = '/home/zchenchen1999/thesis_formal/main/result/webmd/replace_pronoun/no_reasoning/11_28_train_r_ds_Meta-Llama-3.1-8B-Instruct-Doctor.Q4_K_M.gguf.csv'
#         # webmd, bc5cdr
#         dataset_type = "webmd"
#         # gpt, openllm
#         llm_type = "openllm"
#         # 想要評估的欄位  [original_result, consecutive_result, multi_hop_result, default_result, umls_result
#         evaluate_paths = [path]
#         # ==============================
#         print(f"""
#         資料集：{dataset_type}
#         LLM: {llm_type}
#         檔案：{file_path}
#         各自比較：{subset}""")
#         print_bootstrap(file_path=file_path, dataset_type=dataset_type, 
#                         llm_type=llm_type, evaluate_paths=evaluate_paths, 
#                         subset=subset, nbr_runs=nbr_runs)

#         # ===========參數設置============
#         file_path = '/home/zchenchen1999/thesis_formal/main/result/webmd/replace_pronoun/no_reasoning/11_15_train_r_dsm_gpt-3.5-turbo.csv'
#         # webmd, bc5cdr
#         dataset_type = "webmd"
#         # gpt, openllm
#         llm_type = "gpt"
#         # 想要評估的欄位  [original_result, consecutive_result, multi_hop_result, default_result, umls_result
#         evaluate_paths = [path]
#         # ==============================
#         print(f"""
#         資料集：{dataset_type}
#         LLM: {llm_type}
#         檔案：{file_path}
#         各自比較：{subset}""")
#         print_bootstrap(file_path=file_path, dataset_type=dataset_type, 
#                         llm_type=llm_type, evaluate_paths=evaluate_paths, 
#                         subset=subset, nbr_runs=nbr_runs)

#         # ===========參數設置============
#         file_path = '/home/zchenchen1999/thesis_formal/main/result/webmd/replace_pronoun/no_reasoning/11_15_train_r_dsm_gpt-4o-mini.csv'
#         # webmd, bc5cdr
#         dataset_type = "webmd"
#         # gpt, openllm
#         llm_type = "gpt"
#         # 想要評估的欄位  [original_result, consecutive_result, multi_hop_result, default_result, umls_result
#         evaluate_paths = [path]
#         # ==============================
#         print(f"""
#         資料集：{dataset_type}
#         LLM: {llm_type}
#         檔案：{file_path}
#         各自比較：{subset}""")
#         print_bootstrap(file_path=file_path, dataset_type=dataset_type, 
#                         llm_type=llm_type, evaluate_paths=evaluate_paths, 
#                         subset=subset, nbr_runs=nbr_runs)

#         ############################
#         # BC5CDR
#         ############################
#         # ===========參數設置============
#         file_path = '/home/zchenchen1999/thesis_formal/main/result/bc5cdr/replace_pronoun/no_reasoning/11_28_train_r_ds_Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf.csv'
#         # webmd, bc5cdr
#         dataset_type = "bc5cdr"
#         # gpt, openllm
#         llm_type = "openllm"
#         # 想要評估的欄位  [original_result, consecutive_result, multi_hop_result, default_result, umls_result
#         evaluate_paths = [path]
#         # ==============================
#         print(f"""
#         資料集：{dataset_type}
#         LLM: {llm_type}
#         檔案：{file_path}
#         各自比較：{subset}""")
#         print_bootstrap(file_path=file_path, dataset_type=dataset_type, 
#                         llm_type=llm_type, evaluate_paths=evaluate_paths, 
#                         subset=subset, nbr_runs=nbr_runs)

#         # ===========參數設置============
#         file_path = '/home/zchenchen1999/thesis_formal/main/result/bc5cdr/replace_pronoun/no_reasoning/11_28_train_r_ds_Meta-Llama-3.1-8B-Instruct-Doctor.Q4_K_M.gguf.csv'
#         # webmd, bc5cdr
#         dataset_type = "bc5cdr"
#         # gpt, openllm
#         llm_type = "openllm"
#         # 想要評估的欄位  [original_result, consecutive_result, multi_hop_result, default_result, umls_result
#         evaluate_paths = [path]
#         # ==============================
#         print(f"""
#         資料集：{dataset_type}
#         LLM: {llm_type}
#         檔案：{file_path}
#         各自比較：{subset}""")
#         print_bootstrap(file_path=file_path, dataset_type=dataset_type, 
#                         llm_type=llm_type, evaluate_paths=evaluate_paths, 
#                         subset=subset, nbr_runs=nbr_runs)

#         # ===========參數設置============
#         file_path = '/home/zchenchen1999/thesis_formal/main/result/bc5cdr/replace_pronoun/no_reasoning/11_28_train_r_ds_gpt-3.5-turbo.csv'
#         # webmd, bc5cdr
#         dataset_type = "bc5cdr"
#         # gpt, openllm
#         llm_type = "gpt"
#         # 想要評估的欄位  [original_result, consecutive_result, multi_hop_result, default_result, umls_result
#         evaluate_paths = [path]
#         # ==============================
#         print(f"""
#         資料集：{dataset_type}
#         LLM: {llm_type}
#         檔案：{file_path}
#         各自比較：{subset}""")
#         print_bootstrap(file_path=file_path, dataset_type=dataset_type, 
#                         llm_type=llm_type, evaluate_paths=evaluate_paths, 
#                         subset=subset, nbr_runs=nbr_runs)

#         # ===========參數設置============
#         file_path = '/home/zchenchen1999/thesis_formal/main/result/bc5cdr/replace_pronoun/no_reasoning/11_28_train_r_ds_gpt-4o-mini.csv'
#         # webmd, bc5cdr
#         dataset_type = "bc5cdr"
#         # gpt, openllm
#         llm_type = "gpt"
#         # 想要評估的欄位  [original_result, consecutive_result, multi_hop_result, default_result, umls_result
#         evaluate_paths = [path]
#         # ==============================
#         print(f"""
#         資料集：{dataset_type}
#         LLM: {llm_type}
#         檔案：{file_path}
#         各自比較：{subset}""")
#         print_bootstrap(file_path=file_path, dataset_type=dataset_type, 
#                         llm_type=llm_type, evaluate_paths=evaluate_paths, 
#                         subset=subset, nbr_runs=nbr_runs)
# main()