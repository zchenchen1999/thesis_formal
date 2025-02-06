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
        # # intersection
        # df['intersection_demonstration'] = df.apply(lambda x: format_demonstration(x[f'text{search_ecidence_type}'], x['drugs'][0], x['symptoms'][0], x['ground_truth'], evidence=x['intersection_evidence'], data=data), axis=1)
        # # union
        # df['union_demonstration'] = df.apply(lambda x: format_demonstration(x[f'text{search_ecidence_type}'], x['drugs'][0], x['symptoms'][0], x['ground_truth'], evidence=x['union_evidence'], data=data), axis=1)
        # umls
        # df['umls_demonstration'] = df.apply(lambda x: format_demonstration(x[f'text{search_ecidence_type}'], x['drugs'][0], x['symptoms'][0], x['ground_truth'], umls_path=x['umls_evidence'], data=data), axis=1)
    return df



###########################################
# sample demonstrations 
###########################################
# path_type=["original", "consecutive", "multi_hop", "default", "intersection", "union", "umls"]
def sample_demonstration_prompt(df, sample_num, path_type="original"):
    # 篩選某種 path 有 evidence 的資料
    if (path_type != "original"):
        df = df.dropna(subset=[f'{path_type}_evidence'])
    df = df[df[f'{path_type}_demonstration'] != ""]
    
    # 每一種 anser 都抽取 sample_num 個示例
    grouped = df.groupby('ground_truth')
    # sampled_df = grouped.apply(lambda x: x.sample(n=sample_num)).reset_index(drop=True)
    sampled_df = grouped.apply(
            lambda x: x.sample(n=min(sample_num, len(x)))  # 動態調整 sample_num
        ).reset_index(drop=True)
    # 打亂順序
    sampled_df = sampled_df.sample(frac=1).reset_index(drop=True)
    dict_records = sampled_df.to_dict(orient='records')
    
    demonstration_string = ""
    for index, i in enumerate(dict_records):
        demonstration_string += f"""===
Example {index+1}
{i[f'{path_type}_demonstration']}

"""
    return demonstration_string