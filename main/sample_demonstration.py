# %%
import pandas as pd
from evidence_path_rule import extract_path, get_evidence_by_entity_pair
import ast
import random
from create_sentences import get_vertex_set, get_sentences
from evidence_path_umls import get_umls_evidence
# %%
webmd_mapping_dict = {
    "has_positive_reaction": "A",
    "has_negative_reaction": "B",
    "has_adverse_drug_reaction": "C",
    "none_of_above": "D"
}

bc5cdr_mapping_dict = {
    "has_adverse_drug_reaction": "A",
    "none_of_above": "B"
}

######################################
# 無 reasoning 的 demonstration
######################################
context_templete = """Context: {}

Given the context, the relation between "{}" and "{}" is {}"""

context_evidence_templete = """Context: {}

Evidence sentences in the context:
```
{}
```

Given the context and focusing on the evidence sentences, the relation between "{}" and "{}" is {}"""

context_umls_templete = """Context: {}

Knowledge Base Evidences:
```
{}
```

Given the context and using the knowledge base evidences as supplementary information, the relation between "{}" and "{}" is {}"""

######################################
# 有 reasoning 的 demonstration
######################################
context_reasoning_templete = """Context: {}

Given the context, the relation between "{}" and "{}" is {}. It is because: 
{}
"""

context_evidence_reasoning_templete = """Context: {}

Evidence sentences in the context:
```
{}
```

Given the context and focusing on the evidence sentences, the relation between "{}" and "{}" is {}. It is because: 
{}
"""

context_umls_reasoning_templete = """Context: {}

Knowledge Base Evidences:
```
{}
```

Given the context and using the knowledge base evidences as supplementary information, the relation between "{}" and "{}" is {}. It is because: 
{}
"""

###########################################
# 取得 demonstration string
###########################################
def format_demonstration(context, head_entity, tail_entity, relation_groundTruth, evidence=None, umls_path=None, reason=None, data="webmd"):
    format_string = ""

    if data == "webmd":
        mapping_dict = webmd_mapping_dict
    else:
        mapping_dict = bc5cdr_mapping_dict
    # print(mapping_dict)
    # 有提供 reason
    if (reason):
        # 有 evidence
        if (evidence):
            if (evidence != ""):
                format_string = context_evidence_reasoning_templete.format(context, evidence, head_entity, tail_entity, mapping_dict[relation_groundTruth], reason)
        # 有 umls path
        elif (umls_path):
            if (umls_path != ""):
                format_string = context_umls_reasoning_templete.format(context, umls_path, head_entity, tail_entity, mapping_dict[relation_groundTruth], reason)
        # 都沒有
        else:
            format_string = context_reasoning_templete.format(context, head_entity, tail_entity, mapping_dict[relation_groundTruth], reason)
    # 無提供 reason
    else:
        # 有 evidence
        if (evidence):
            if (evidence != ""):
                format_string = context_evidence_templete.format(context, evidence, head_entity, tail_entity, mapping_dict[relation_groundTruth])
        # 有 umls path
        elif (umls_path):
            if (umls_path != ""):
                format_string = context_umls_templete.format(context, umls_path, head_entity, tail_entity, mapping_dict[relation_groundTruth])
        # 都沒有
        else:
            format_string = context_templete.format(context, head_entity, tail_entity, mapping_dict[relation_groundTruth])
    return format_string

###########################################
# test set 取得不同 path 結果 string 並回傳
###########################################
def test_get_demonstration(df, search_ecidence_type="", has_reason=False, data=["webmd", 'bc5cdr']):
    # 有 reasoning 的 demonstration
    if (has_reason):
        # original
        df['original_demonstration'] = df.apply(lambda x: format_demonstration(x[f'text{search_ecidence_type}'], x['drugs'][0], x['symptoms'][0], x['ground_truth'], reason=x['original_reason'], data=data), axis=1)
        # consecutive
        df['consecutive_demonstration'] = df.apply(lambda x: format_demonstration(x[f'text{search_ecidence_type}'], x['drugs'][0], x['symptoms'][0], x['ground_truth'], evidence=x['consecutive_evidence'], reason=x['consecutive_reason'], data=data), axis=1)
        # multi-hop
        df['multi_hop_demonstration'] = df.apply(lambda x: format_demonstration(x[f'text{search_ecidence_type}'], x['drugs'][0], x['symptoms'][0], x['ground_truth'], evidence=x['multi_hop_evidence'], reason=x['multi_hop_reason'], data=data), axis=1)
        # default
        df['default_demonstration'] = df.apply(lambda x: format_demonstration(x[f'text{search_ecidence_type}'], x['drugs'][0], x['symptoms'][0], x['ground_truth'], evidence=x['default_evidence'], reason=x['default_reason'], data=data), axis=1)
        # umls
        # df['umls_demonstration'] = df.apply(lambda x: format_demonstration(x[f'text{search_ecidence_type}'], x['drugs'][0], x['symptoms'][0], x['ground_truth'], umls_path=x['umls_evidence'], reason=x['umls_reason'], data=data), axis=1)
    # 無 reasoning 的 demonstration
    else:
        # original
        df['original_demonstration'] = df.apply(lambda x: format_demonstration(x[f'text{search_ecidence_type}'], x['drugs'][0], x['symptoms'][0], x['ground_truth'], data=data), axis=1)
        # consecutive
        df['consecutive_demonstration'] = df.apply(lambda x: format_demonstration(x[f'text{search_ecidence_type}'], x['drugs'][0], x['symptoms'][0], x['ground_truth'], evidence=x['consecutive_evidence'], data=data), axis=1)
        # multi-hop
        df['multi_hop_demonstration'] = df.apply(lambda x: format_demonstration(x[f'text{search_ecidence_type}'], x['drugs'][0], x['symptoms'][0], x['ground_truth'], evidence=x['multi_hop_evidence'], data=data), axis=1)
        # default
        df['default_demonstration'] = df.apply(lambda x: format_demonstration(x[f'text{search_ecidence_type}'], x['drugs'][0], x['symptoms'][0], x['ground_truth'], evidence=x['default_evidence'], data=data), axis=1)
        # umls
        # df['umls_demonstration'] = df.apply(lambda x: format_demonstration(x[f'text{search_ecidence_type}'], x['drugs'][0], x['symptoms'][0], x['ground_truth'], umls_path=x['umls_evidence'], data=data), axis=1)
    return df



###########################################
# sample demonstrations 
###########################################
# path_type=["original", "consecutive", "multi_hop", "default", "umls"]
def sample_demonstration_prompt(df, sample_num, path_type="original"):
    # 篩選某種 path 有 evidence 的資料
    if (path_type != "original"):
        df = df.dropna(subset=[f'{path_type}_evidence'])
    df = df[df[f'{path_type}_demonstration'] != ""]
    
    # 每一種 anser 都抽取 sample_num 個示例
    grouped = df.groupby('ground_truth')
    sampled_df = grouped.apply(lambda x: x.sample(n=sample_num)).reset_index(drop=True)
    dict_records = sampled_df.to_dict(orient='records')
    
    demonstration_string = ""
    for index, i in enumerate(dict_records):
        demonstration_string += f"""===
Example {index+1}
{i[f'{path_type}_demonstration']}

"""
    return demonstration_string


# %%
# df = pd.read_csv('/home/zchenchen1999/MindMap-DDI/data/Web_MD/WebMD_annotated_v2_exploded_reasoning.csv')
# # 轉換成 list
# df = utils.str_to_list(df, 
#         ['drugs', 'symptoms', 'relations', 
#         'sents', 'spacy_entity', 'scispacy_entity',
#         'sents_replace_pronoun', 'spacy_entity_replace_pronoun', 'scispacy_entity_replace_pronoun']
#     )

# # 分成兩個資料集
# train_set = df.groupby('ground_truth', group_keys=False).apply(lambda x: x.sample(frac=0.7, random_state=42))
# test_set = df.drop(train_set.index) # 剩下的資料拿去抽樣 demonstration

# test_set = test_get_paths_evidence_string(test_set, context_type="_replace_pronoun", has_reason=True)
# # test_set.to_csv('/home/zchenchen1999/MindMap-DDI/data/Web_MD/test.csv')

# sample_demonstration_prompt(test_set, 1, path_type="multi_hop")
# %%





# # example 只有 context
# def sample_data_context(df, sample_num, context_type=""):
#     sample_result_df = pd.DataFrame(columns=['demon_string', 'relation_type'])

#     for index, row in df.iterrows():
#         relation_dict = {}
#         for k in row['relations']:
#             key = f"{k[0]}_{k[2]}"
#             relation_dict[key] = k[1]

#         for drug in row['drugs']:
#             for symptom in row['symptoms']:

#                 combine_key = f"{drug}_{symptom}"
#                 if not (combine_key in relation_dict.keys()):
#                     relation_type = "none_of_above"
#                 else:
#                     relation_type = relation_dict[combine_key]

#                 de_string = context_templete.format(row[f'text{context_type}'], drug, symptom, mapping_dict[relation_type])
#                 new_df = pd.DataFrame([{'demon_string': de_string, 'relation_type': relation_type}])
#                 sample_result_df = pd.concat([sample_result_df, new_df], ignore_index=True)

#     sample_result_df = sample_result_df.groupby('relation_type').apply(lambda x: x.sample(n=min(len(x), sample_num))).reset_index(drop=True)
#     dict_records = sample_result_df.to_dict(orient='records')

#     demonstration_string = ""
#     for index, i in enumerate(dict_records):
#         demonstration_string += f"""===
# Example {index+1}
# {i['demon_string']}

# """

#     return demonstration_string+"==="

# # example 只有 evidence
# def sample_data_evidence(df, sample_num, context_type=""):
#     sample_result_df = pd.DataFrame(columns=['demon_string', 'relation_type'])

#     for index, row in df.iterrows():

#         vertexSet = get_vertex_set(row['all_entity'], row[f'sents{context_type}'])
#         merge = extract_path(row[f'sents{context_type}'], vertexSet, True, path_type="merge")
#         default = extract_path(row[f'sents{context_type}'], vertexSet, True, path_type="default")

#         relation_dict = {}
#         for k in row['relations']:
#             key = f"{k[0]}_{k[2]}"
#             relation_dict[key] = k[1]

#         for drug in row['drugs']:
#             for symptom in row['symptoms']:
#                 merge_string = get_evidence_by_entity_pair(drug, symptom, row['all_entity'], row[f'sents{context_type}'], merge)
#                 default_string = get_evidence_by_entity_pair(drug, symptom, row['all_entity'], row[f'sents{context_type}'], default)

#                 combine_key = f"{drug}_{symptom}"
#                 if not (combine_key in relation_dict.keys()):
#                     relation_type = "none_of_above"
#                 else:
#                     relation_type = relation_dict[combine_key]

#                 merge_string_format = context_templete.format(merge_string, drug, symptom, mapping_dict[relation_type])
#                 default_string_format = context_templete.format(default_string, drug, symptom, mapping_dict[relation_type])
#                 new_df = pd.DataFrame([{'merge_string_format': merge_string_format, 'default_string_format': default_string_format, 'relation_type': relation_type}])
#                 sample_result_df = pd.concat([sample_result_df, new_df], ignore_index=True)

#     sample_result_df = sample_result_df.groupby('relation_type').apply(lambda x: x.sample(n=min(len(x), sample_num))).reset_index(drop=True)
#     dict_records = sample_result_df.to_dict(orient='records')

#     merge_demonstration_string = ""
#     default_demonstration_string = ""
#     for index, i in enumerate(dict_records):
#         merge_demonstration_string += f"""===
# Example {index+1}
# {i['merge_string_format']}

# """
#         default_demonstration_string += f"""===
# Example {index+1}
# {i['default_string_format']}

# """


#     return merge_demonstration_string+"===", default_demonstration_string+"==="

# # example 有 context + evidence
# def sample_data_context_evidence(df, sample_num, context_type=""):
#     sample_result_df = pd.DataFrame(columns=['demon_string', 'relation_type'])

#     for index, row in df.iterrows():

#         vertexSet = get_vertex_set(row['all_entity'], row[f'sents{context_type}'])
#         merge = extract_path(row[f'sents{context_type}'], vertexSet, True, path_type="merge")
#         default = extract_path(row[f'sents{context_type}'], vertexSet, True, path_type="default")

#         relation_dict = {}
#         for k in row['relations']:
#             key = f"{k[0]}_{k[2]}"
#             relation_dict[key] = k[1]

#         for drug in row['drugs']:
#             for symptom in row['symptoms']:
#                 merge_string = get_evidence_by_entity_pair(drug, symptom, row['all_entity'], row[f'sents{context_type}'], merge)
#                 default_string = get_evidence_by_entity_pair(drug, symptom, row['all_entity'], row[f'sents{context_type}'], default)

#                 combine_key = f"{drug}_{symptom}"
#                 if not (combine_key in relation_dict.keys()):
#                     relation_type = "none_of_above"
#                 else:
#                     relation_type = relation_dict[combine_key]

#                 merge_string_format = context_evidence_templete.format(row[f'text{context_type}'], merge_string, drug, symptom, mapping_dict[relation_type])
#                 default_string_format = context_evidence_templete.format(row[f'text{context_type}'], default_string, drug, symptom, mapping_dict[relation_type])
#                 new_df = pd.DataFrame([{'merge_string_format': merge_string_format, 'default_string_format': default_string_format, 'relation_type': relation_type}])
#                 sample_result_df = pd.concat([sample_result_df, new_df], ignore_index=True)

#     sample_result_df = sample_result_df.groupby('relation_type').apply(lambda x: x.sample(n=min(len(x), sample_num))).reset_index(drop=True)
#     dict_records = sample_result_df.to_dict(orient='records')

#     merge_demonstration_string = ""
#     default_demonstration_string = ""
#     for index, i in enumerate(dict_records):
#         merge_demonstration_string += f"""===
# Example {index+1}
# {i['merge_string_format']}

# """
#         default_demonstration_string += f"""===
# Example {index+1}
# {i['default_string_format']}

# """

#     return merge_demonstration_string+"===", default_demonstration_string+"==="