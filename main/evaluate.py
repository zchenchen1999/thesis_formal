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

import utils

# %%
# 從 json 字串中抓取答案
def extract_answer_from_string(string_):
    # 將換行移除，才能成功轉換
    string_ = string_.replace("\n", "")
    # print(string_)
    try:
        # 可以成功轉換
        dict_result = json.loads(string_)
        answer = dict_result['ADEs']
        # answer = dict_result['Drug_Effects']
    except:
        # 無法轉換，使用正則表達式抓取
        pattern = r'\[\s*(?:\[\"[^\"]*\",\s?\"[^\"]*\"\],?\s?)*\s*\]|\[\]'
        matches = re.findall(pattern, string_)
        if (len(matches) < 1):
            print(f"RE無法抓取到: {string_}")
        try:
            # 將 match 到的字串轉為 list
            # 使用 ast.literal_eval() 将字符串转换为真正的列表
            answer = ast.literal_eval(matches[0])
        except:
            print(f"無法轉為 list: {matches[0]}")
    return answer

def list_unique(data):
    # 使用集合來移除重複項目
    unique_data = []
    seen = set()
    for item in data:
        tuple_item = tuple(item)
        if tuple_item not in seen:
            seen.add(tuple_item)
            unique_data.append(item)
    return unique_data

# Function to swap elements
def swap_elements(array):
    return [[el[1], el[0], el[2]] for el in array]

# def change_label(array):
#     return [[label_dict[el[0]], el[1], el[2]] for el in array]


def is_similar(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio() >= 0.8

def evaluate_ade(ground_truth, result):
    # 将每个关系对转换为排序后的元组，并使用集合去重
    # 第0項: head, 第1項: relation,  第2項: tail
    ground_truth = set(tuple([ade[1].lower()] + sorted([ade[0].lower(), ade[2].lower()])) for ade in ground_truth)
    result = set(tuple([ade[1].lower()] + sorted([ade[0].lower(), ade[2].lower()])) for ade in result)
    # 轉換後： 第0項: relation, 第1項: head,  第2項: tail
    # print(ground_truth)
    # print(result)
    # ground_truth = set(tuple(sorted([x.lower() for x in ade])) for ade in ground_truth)
    # result = set(tuple(sorted([x.lower() for x in ade])) for ade in result)


    # 計算正確的三元組數量
    correct_triplets = 0
    for res in result:
        for gt in ground_truth:
            if res[0]== gt[0] and is_similar(res[1], gt[1]) and is_similar(res[2], gt[2]):
                correct_triplets += 1
                break

    TP = correct_triplets
    FP = len(result) - TP
    if (FP < 0): print(FP)
    FN = len(ground_truth) - TP

    # 計算精確度
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    # 計算召回率
    recall = TP / (FN + TP) if (FN + TP) > 0 else 0
    # 計算 F1 分數
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return TP, FP, FN, recall, precision, f1_score

def evaluate_sigle(df):
    df[['tp', 'fp', 'fn', 'micro recall', 'micro precision', 'micro f1']] = df.apply(lambda row: evaluate_ade(row['relation_in_text'], row['list_str']), axis=1, result_type='expand')
    return df

def evaluate_total(df):
    TP = df['tp'].to_list()
    FP = df['fp'].to_list()
    FN = df['fn'].to_list()
    
    total_precision = sum(TP)/(sum(TP)+sum(FP))
    total_recall = sum(TP)/(sum(TP)+sum(FN))
    total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall)

    print(f'avg micro precision: {total_precision}')
    print(f'avg micro recall: {total_recall}')
    print(f'avg micro f1: {total_f1}')

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
    return 


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
# 找尋最佳 weights 與 threshold
def optimize_weights_and_threshold(path, r_name=[]):
    df = pd.read_csv(path)
    df = preprocess(df, r_name)

    # 過濾出每個欄位都有結果的資料
    print(f"test shape before: {df.shape}") # 新增
    for r in r_name: # 新增
        df = df[df[r].apply(lambda x: len(x) >= 1)] # 新增

    print(f"test shape after: {df.shape}")
    # df = df[df['consecutive_result'] != "{}"]
    # print(df.shape)
    # df = preprocess(df, ['c_result', 'consecutive_result', 'path_two_result', 'path_three_result', 'd_result'])

    # 定義搜索範圍
    param_grid = {f'{name}_weight': [i / 10.0 for i in range(0, 11)] for name in r_name}
    
    # 添加 a 和 b 的搜索範圍
    param_grid['a'] = [i / 10.0 for i in range(0, 11)]
    param_grid['b'] = [i / 10.0 for i in range(0, 11)]
    # param_grid['b'] = [0] # 新增
    param_grid_all = list(ParameterGrid(param_grid))

    # # 排除掉 c_weight, m_weight, d_weight 總和不為1的組合
    # param_grid_filter = [p for p in param_grid_all if abs(p['c_weight'] + p['m_weight'] + p['d_weight'] - 1.0) < 1e-3]

    # 動態過濾權重總和不等於 1 的組合
    param_grid_filter = [
        p for p in param_grid_all 
        if abs(sum(p[f'{name}_weight'] for name in r_name) - 1.0) < 1e-3
    ]

    # param_grid_filter = [i / 10.0 for i in range(0, 11)]

    best_score = -float('inf')
    best_params = None
    best_report = ""

    for params in param_grid_filter:
        df_tmp = df.copy()

        # 動態提取每個結果的權重
        weights = {name: params[f'{name}_weight'] for name in r_name}
        a = params['a']
        b = params['b']
        
        # df_tmp['predict'] = df_tmp.apply(lambda x: get_combine_result(x['consecutive_result'], x['c_result'], x['d_result'], c_weight, m_weight, d_weight, a, b), axis=1)
        # df_tmp['predict'] = df_tmp.apply(lambda x: get_combine_result(x['c_result'], x['m_result'], x['d_result'], c_weight, m_weight, d_weight, a, b), axis=1)
        df_tmp['predict'] = df_tmp.apply(lambda x: get_combine_result(
            *[(x[name], weights[name]) for name in r_name],
            parm_a=a, parm_b=b), axis=1)
        # df_tmp['predict'] = df_tmp.apply(lambda x: get_combine_result(x['c_result'], x['m_result'], x['d_result'], 1.0, 0.0, 0.0, params, 0.1), axis=1)
        # df_tmp['predict'] = df_tmp.apply(lambda x: get_context_result(x['c_result'], 0), axis=1)
    
        ground_truth = df_tmp['ground_truth'].to_list()
        predict = df_tmp['predict'].to_list()

        # 明確指定 labels 和 target_names 的順序
        labels = ['none_of_above', 'has_adverse_drug_reaction', 'has_positive_reaction', 'has_negative_reaction']
        target_names = ['none_of_above', 'has_adverse_drug_reaction', 'has_positive_reaction', 'has_negative_reaction']
        report = classification_report(ground_truth, predict, labels=labels, target_names=target_names, output_dict=True)

        weighted_avg_f1 = report['weighted avg']['f1-score']
        # print(report)
        # print(weighted_avg_f1)

        if weighted_avg_f1 > best_score:
            best_score = weighted_avg_f1
            best_params = params
            best_report = report
        # break
    # print(f"最佳分數: {best_score}")
    # print(f"最佳參數: {best_params}")
    # print(f"最佳報告: {best_report}")
    return best_params

# r_name = ['c_result', 'm_result', 'd_result']
# optimize_weights_and_threshold('/home/zchenchen1999/MindMap-DDI/data/Web_MD/result/multi_path_result/all_test.csv', r_name)
# %%
#################################################
#          衡量- LLM evidence 比較 - 總表         #
#################################################
# 衡量單一 LLM 回傳內容 (最高機率) - 總表
# webmd, bc5cdr
data = "bc5cdr"
# gpt, openllm
llm_type = "gpt"
train_path = '/home/zchenchen1999/thesis_formal/main/result/bc5cdr/replace_pronoun/zeroshot_cot/12_05_bc5cdr_train_r_dsm_gpt-3.5-turbo.csv'

r_name = ['multi_hop_result'] # 想要評估的欄位  [original_result, consecutive_result, multi_hop_result, default_result, umls_result
df = pd.read_csv(train_path)
# df = df[df[r_name[0]] != '{}'] # 過濾掉沒結果的row
print(r_name[0])
print(df.shape)
df = preprocess(df, r_name+['original_result'] if ('original_result' not in r_name) else r_name)
# type(df.loc[0, 'consecutive_result'])
# df['predict'] = df.apply(lambda x: get_highest_pro_result(x[r_name[0]])
if (llm_type == "gpt"):
    df['predict'] = df.apply(lambda x: get_highest_pro_result(x['original_result'], data) if x[r_name[0]] == {} else get_highest_pro_result(x[r_name[0]], data), axis=1)
else:
    print(data)
    df['predict'] = df.apply(lambda x: get_open_llm_result(x['original_result'], data) if x[r_name[0]] == {} else get_open_llm_result(x[r_name[0]], data), axis=1)

ground_truth = df['ground_truth'].to_list()
predict = df['predict'].to_list()
print(predict)

# 明確指定 labels 和 target_names 的順序
if (data == "webmd"):
    labels = ['none_of_above', 'has_adverse_drug_reaction', 'has_positive_reaction', 'has_negative_reaction']
    target_names = ['none_of_above', 'has_adverse_drug_reaction', 'has_positive_reaction', 'has_negative_reaction']
else:
    labels = ['none_of_above', 'has_adverse_drug_reaction']
    target_names = ['none_of_above', 'has_adverse_drug_reaction']

report = classification_report(ground_truth, predict, labels=labels, target_names=target_names, output_dict=True)
report
# %%
#################################################
#        衡量- LLM evidence 比較 - 各自比較        #
#################################################
# webmd, bc5cdr
data = "webmd"
# gpt, openllm
llm_type = "gpt"
train_path = '/home/zchenchen1999/thesis_formal/main/result/webmd/replace_pronoun/no_reasoning/11_15_train_r_dsm_gpt-4o-mini.csv'

r_name = ['default_result'] # 想要評估的欄位  [original_result, consecutive_result, multi_hop_result, default_result, umls_result

df = pd.read_csv(train_path)
df = df[df[r_name[0]] != '{}'] # 過濾掉沒結果的row
# df = df[df['default_reasoning'] != '{}'] # [original_reasoning, consecutive_reasoning, multi_hop_reasoning, default_reasoning, umls_reasoning]
print(r_name[0])
print(df.shape)

df = preprocess(df, r_name+['original_result'] if ('original_result' not in r_name) else r_name)
if (llm_type == "gpt"):
    df['predict_original'] = df.apply(lambda x: get_highest_pro_result(x['original_result'], data), axis=1)
    df['predict_compare'] = df.apply(lambda x: get_highest_pro_result(x[r_name[0]], data), axis=1)
else:
    df['predict_original'] = df.apply(lambda x: get_open_llm_result(x['original_result'], data), axis=1)
    df['predict_compare'] = df.apply(lambda x: get_open_llm_result(x[r_name[0]], data), axis=1)

ground_truth = df['ground_truth'].to_list()
predict_original = df['predict_original'].to_list()
predict_compare = df['predict_compare'].to_list()

# 明確指定 labels 和 target_names 的順序
if (data == "webmd"):
    labels = ['none_of_above', 'has_adverse_drug_reaction', 'has_positive_reaction', 'has_negative_reaction']
    target_names = ['none_of_above', 'has_adverse_drug_reaction', 'has_positive_reaction', 'has_negative_reaction']
else:
    labels = ['none_of_above', 'has_adverse_drug_reaction']
    target_names = ['none_of_above', 'has_adverse_drug_reaction']

report_original = classification_report(ground_truth, predict_original, labels=labels, target_names=target_names, output_dict=True)
report_compare = classification_report(ground_truth, predict_compare, labels=labels, target_names=target_names, output_dict=True)

print("original")
print(json.dumps(report_original, indent=4))
print("\n=====================\n")
print(r_name[0])
print(json.dumps(report_compare, indent=4))
# %%
# #################################################
# #                衡量-最佳參數組合                 #
# #################################################
# # 測試檔案路徑
# test_path = '/home/zchenchen1999/MindMap-DDI/data/Web_MD/result/multi_path_result/10_12_test_r_dsm.csv'
# train_path = '/home/zchenchen1999/MindMap-DDI/data/Web_MD/result/multi_path_result/10_12_train_r_dsm.csv'
# # test set 先取得最佳參數
# r_name = ['original_result', 'consecutive_result', 'default_result'] # 想要評估的欄位
# best_params = optimize_weights_and_threshold(test_path, r_name)
# print(f'最佳參數： {best_params}')

# # 衡量多種 path 結果
# df = pd.read_csv(train_path)
# df = preprocess(df, r_name)

# # 過濾出每個欄位都有結果的資料
# for r in r_name: # 新增
#     df = df[df[r].apply(lambda x: len(x) >= 1)] # 新增
# print(f"train shape: {df.shape}") # 新增

# df['predict'] = df.apply(lambda x: get_combine_result(
#     *[(x[name], best_params[f'{name}_weight']) for name in r_name],
#     parm_a=best_params['a'], parm_b=best_params['b']), axis=1)

# ground_truth =  df['ground_truth'].to_list()
# predict = df['predict'].to_list()

# print(df['ground_truth'].value_counts())
# print(df['predict'].value_counts())

# # 明確指定 labels 和 target_names 的順序
# labels = ['none_of_above', 'has_adverse_drug_reaction', 'has_positive_reaction', 'has_negative_reaction']
# target_names = ['none_of_above', 'has_adverse_drug_reaction', 'has_positive_reaction', 'has_negative_reaction']
# # Generate the classification report
# report = classification_report(ground_truth, predict, labels=labels, target_names=target_names, output_dict=True)
# report
# # %%
# #################################################
# #                衡量-純 LLM 回傳內容             #
# #################################################
# train_path = '/home/zchenchen1999/thesis/main/result/zeroshot_cot/11_15_train_r_dsm_gpt-3.5-turbo.csv'
# r_name = ['original_result'] # 想要評估的欄位  [original_result, consecutive_result, multi_hop_result, default_result, umls_result
# df = pd.read_csv(train_path)
# df = df[df[r_name[0]] != '{}'] # 過濾掉沒結果的row
# print(r_name[0])
# print(df.shape)
# df = preprocess(df, r_name)
# # type(df.loc[0, 'consecutive_result'])
# df['predict'] = df.apply(lambda x: get_response_result(x[r_name[0]]), axis=1)
# ground_truth = df['ground_truth'].to_list()
# predict = df['predict'].to_list()
# # 明確指定 labels 和 target_names 的順序
# labels = ['none_of_above', 'has_adverse_drug_reaction', 'has_positive_reaction', 'has_negative_reaction']
# target_names = ['none_of_above', 'has_adverse_drug_reaction', 'has_positive_reaction', 'has_negative_reaction']
# report = classification_report(ground_truth, predict, labels=labels, target_names=target_names, output_dict=True)
# report













# %%
# %%
# train_path = ''

# df = pd.read_csv(train_path)

# # 定義 reasoning 和 result 欄位的對應關係
# paths = ['original', 'consecutive', 'multi_hop', 'default']
# reasoning_columns = [f"{path}_reasoning" for path in paths]
# result_columns = [f"{path}_result" for path in paths]

# # 遍歷每一個 reasoning 和 result 欄位
# for reasoning_col, result_col in zip(reasoning_columns, result_columns):
#     # 找出 reasoning 欄位為 '{}' 的行
#     mask = df[reasoning_col] == '{}'
#     # 將對應的 result 欄位改為 '{}'
#     df.loc[mask, result_col] = '{}'
# df
# df.to_csv(train_path, index=False)
# # %%
