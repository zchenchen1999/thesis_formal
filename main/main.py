# %%
from tqdm import tqdm
tqdm.pandas(desc="Processing rows")
import pandas as pd

from typing import Dict, Any, List

from langchain_core.prompts import ChatPromptTemplate
# from langchain.chat_models import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI


# system prompt (統一)
from RE_prompt import webMD_prompt_system, webMD_vanilla_ICL_prompt_user, webMD_S2PR_evidence_prompt_user
from RE_prompt import bc5cdr_prompt_system, bc5cdr_vanilla_ICL_prompt_user, bc5cdr_S2PR_evidence_prompt_user

# 抽取 demonstration
from sample_demonstration import sample_demonstration_prompt, test_get_demonstration

# 找尋 evidence path
from create_sentences import get_vertex_set, get_sentences
from evidence_path_rule import extract_path, get_evidence_by_entity_pair
from evidence_path_graph import get_evidence_sents
# 找尋 umls path
from evidence_path_umls import get_umls_evidence

import utils
from open_llm import get_open_llm_result

# %%
def get_all_triplets(row, df_test, search_ecidence_type="", sample_num=1, model="gpt-3.5-turbo", parse=False, data="webmd"):

    if (data == "webmd"):
        valid_responses = ["a", "b", "c", "d"]
        system_prompt = webMD_prompt_system
        ICL_user_prompt = webMD_vanilla_ICL_prompt_user
        evidence_user_prompt = webMD_S2PR_evidence_prompt_user
    else:
        valid_responses = ["a", "b"]
        system_prompt = bc5cdr_prompt_system
        ICL_user_prompt = bc5cdr_vanilla_ICL_prompt_user
        evidence_user_prompt = bc5cdr_S2PR_evidence_prompt_user


    context = row[f'text{search_ecidence_type}']
    sents = row[f'sents{search_ecidence_type}']
    head_mention = row['drugs'][0]
    tail_mention = row['symptoms'][0]
    all_drug_metions = row['all_drug']
    all_symptom_mentions = row['all_symptom']
    # spacy_entity = row[f'spacy_entity{search_ecidence_type}']
    scispacy_entity = row[f'scispacy_entity{search_ecidence_type}']


    print(f"head: {head_mention}, tail: {tail_mention}")

    ###################################
    # 合併各種 entity (用於 search path)
    ###################################
    entity_list = utils.add_all_entities(all_drug_metions, all_symptom_mentions, scispacy_entity)

    ################################
    # 取得不同 path 的 evidence 
    ################################
    vertexSet = get_vertex_set(entity_list, sents)
    # consecutive
    consecutive = extract_path(sents, vertexSet, True, path_type="consecutive")
    consecutive_string = get_evidence_by_entity_pair(head_mention, tail_mention, entity_list, sents, consecutive)
    # multi_hop
    sentences_index, link_path_string, multi_hop_string = get_evidence_sents(sents, entity_list, head_mention, tail_mention)
    # default
    default = extract_path(sents, vertexSet, True, path_type="default")
    default_string = get_evidence_by_entity_pair(head_mention, tail_mention, entity_list, sents, default)
    # UMLS path
    # _, umls_string = get_umls_evidence(head_mention, tail_mention, all_drug_metions, all_symptom_mentions, scispacy_entity)

    ################################
    # 取得不同 path 的 demonstration 
    ################################
    original_demonstrations = sample_demonstration_prompt(df_test, sample_num, path_type="original")
    consecutive_demonstrations = sample_demonstration_prompt(df_test, sample_num, path_type="consecutive")
    multi_hop_demonstrations = sample_demonstration_prompt(df_test, sample_num, path_type="multi_hop")
    default_demonstrations = sample_demonstration_prompt(df_test, sample_num, path_type="default")
    # umls_demonstrations = sample_demonstration_prompt(df_test, 1, path_type="umls")
    
    #################
    # 詢問 LLM 分類結果
    #################
    original_result = {}
    consecutive_result = {}
    multi_hop_result = {}
    default_result = {}


    print(f"本次使用模型為： {model}")
    # original
    if (context == ""):
        original_result = {}
    else:
        llm_var = {
            'demonstrations': original_demonstrations, 
            'context': context,
            'head_entity': head_mention,
            'tail_entity': tail_mention
        }
        if (model == "gpt-3.5-turbo" or model == "gpt-4o-mini"):
            # original_result = {} 
            original_response, original_top_classes, original_final_prompt = utils.get_llm_result(
                system_prompt, ICL_user_prompt, var_dict=llm_var, model=model)
            original_result['response'] = original_response
            original_result['top_logprobs'] = original_top_classes
            original_result['prompt'] = original_final_prompt
        else:
            original_response = get_open_llm_result(
                    system_prompt, ICL_user_prompt, var_dict=llm_var, data=data, parse=parse
                )
            original_result['response'] = original_response

    # consecutive
    if (consecutive_string == ""):
        consecutive_result = {}
    else:
        # consecutive_result = consecutive_string
        llm_var = {
            'demonstrations': consecutive_demonstrations, 
            'context': context,
            'evidence': consecutive_string,
            'head_entity': head_mention,
            'tail_entity': tail_mention
        }
        if (model == "gpt-3.5-turbo" or model == "gpt-4o-mini"):
            consecutive_response, consecutive_top_classes, consecutive_final_prompt = utils.get_llm_result(
                system_prompt, evidence_user_prompt, var_dict=llm_var, model=model)
            consecutive_result['response'] = consecutive_response
            consecutive_result['top_logprobs'] = consecutive_top_classes
            consecutive_result['evidence'] = consecutive_string
            consecutive_result['prompt'] = consecutive_final_prompt
        else:
            consecutive_response = get_open_llm_result(
                    system_prompt, evidence_user_prompt, var_dict=llm_var, data=data, parse=parse
                )
            consecutive_result['response'] = consecutive_response
        
    
    # multi_hop
    if (multi_hop_string == ""):
        multi_hop_result = {}
    else:
        # multi_hop_result = multi_hop_string
        llm_var = {
            'demonstrations': multi_hop_demonstrations, 
            'context': context,
            'evidence': multi_hop_string,
            'head_entity': head_mention,
            'tail_entity': tail_mention
        }
        if (model == "gpt-3.5-turbo" or model == "gpt-4o-mini"):
            multi_hop_response, multi_hop_top_classes, multi_hop_final_prompt = utils.get_llm_result(
                system_prompt, evidence_user_prompt, var_dict=llm_var, model=model)
            multi_hop_result['response'] = multi_hop_response
            multi_hop_result['top_logprobs'] = multi_hop_top_classes
            multi_hop_result['evidence'] = multi_hop_string
            multi_hop_result['prompt'] = multi_hop_final_prompt
            multi_hop_result['sentence_index'] = sentences_index
        else:
            multi_hop_response = get_open_llm_result(
                    system_prompt, evidence_user_prompt, var_dict=llm_var, data=data, parse=parse
                )
            multi_hop_result['response'] = multi_hop_response

    # # default
    if (default_string == ""):
        default_result = {}
    else:
        # default_result = default_string
        llm_var = {
            'demonstrations': default_demonstrations, 
            'context': context,
            'evidence': default_string,
            'head_entity': head_mention,
            'tail_entity': tail_mention
        }
        if (model == "gpt-3.5-turbo" or model == "gpt-4o-mini"):
            default_response, default_top_classes, default_final_prompt = utils.get_llm_result(
                system_prompt, evidence_user_prompt, var_dict=llm_var, model=model)
            default_result['response'] = default_response
            default_result['top_logprobs'] = default_top_classes
            default_result['evidence'] = default_string
            default_result['prompt'] = default_final_prompt
        else:
            default_response = get_open_llm_result(
                    system_prompt, evidence_user_prompt, var_dict=llm_var, data=data, parse=parse
                )
            default_result['response'] = default_response
        

    ##################
    # 轉為 json 並縮排
    ##################
    original_result = utils.dict_to_json(original_result)
    consecutive_result = utils.dict_to_json(consecutive_result)
    multi_hop_result = utils.dict_to_json(multi_hop_result)
    default_result = utils.dict_to_json(default_result)

    # return "", "", "", default_string
    return original_result, consecutive_result, multi_hop_result, default_result

# %%
def main(date, inference, search_ecidence_type, sample_num, has_reason, model="gpt-3.5-turbo", parse=False, data=["webmd", "bc5cdr"]):

    if (data == "webmd"):
        store_path = "/home/zchenchen1999/thesis_formal/main/result/webmd"
        df_train = pd.read_csv('/home/zchenchen1999/thesis_formal/main/preprocessed_data/WebMD/WebMD_annotated_v2_exploded_reasoning_train.csv')
        df_test = pd.read_csv('/home/zchenchen1999/thesis_formal/main/preprocessed_data/WebMD/WebMD_annotated_v2_exploded_reasoning_test.csv')
    else:
        store_path = "/home/zchenchen1999/thesis_formal/main/result/bc5cdr"
        df_train = pd.read_csv('/home/zchenchen1999/thesis_formal/main/preprocessed_data/BC5CDR/BC5CDR_preprocess_train_smallset.csv')
        df_test = pd.read_csv('/home/zchenchen1999/thesis_formal/main/preprocessed_data/BC5CDR/BC5CDR_preprocess_test.csv')

    # 轉換成 list
    df_train = utils.str_to_list(df_train, 
            ['drugs', 'symptoms', 'relations', 
            'sents', 'spacy_entity', 'scispacy_entity',
            'sents_replace_pronoun', 'spacy_entity_replace_pronoun', 'scispacy_entity_replace_pronoun',
            'all_drug', 'all_symptom']
        )
    df_test = utils.str_to_list(df_test, 
            ['drugs', 'symptoms', 'relations', 
            'sents', 'spacy_entity', 'scispacy_entity',
            'sents_replace_pronoun', 'spacy_entity_replace_pronoun', 'scispacy_entity_replace_pronoun',
            'all_drug', 'all_symptom']
        )

    if (inference == 'test'):
        # train, test 調換
        tmp = df_train.copy()
        df_train = df_test
        df_test = tmp

    # test set 取得 demonstration
    df_test = test_get_demonstration(df_test, search_ecidence_type=search_ecidence_type, has_reason=has_reason, data=data)

    # df_train = df_train.head(1)
    print(f"test set (for sample) shape: {df_test.shape}")
    print(f"train set (for inference) shape: {df_train.shape}")

    # inference
    df_train[['original_result', 'consecutive_result', 'multi_hop_result', 'default_result']] = df_train.progress_apply(
        lambda row: get_all_triplets(row, df_test, search_ecidence_type=search_ecidence_type, sample_num=sample_num, model=model, parse=parse, data=data), axis=1, result_type='expand')
    df_train = df_train[['sents', 'sents_replace_pronoun', 'drugs', 'symptoms', 'ground_truth', 'original_result', 'consecutive_result', 'multi_hop_result', 'default_result']]

    # df_train['has_multi_hop_path'] = df_train['sentences_index'].apply(lambda x: len(x[0]) > 1)
    # df_train['multi_path_count'] = df_train['sentences_index'].apply(lambda x: len(x))

    if (has_reason):
        reason_path = 'reasoning'
    else:
        reason_path = 'no_reasoning'

    if (search_ecidence_type == "_replace_pronoun"):
        if_replace_pronoun_path = "replace_pronoun"
    else:
        if_replace_pronoun_path = "original_context"
    


    df_train.to_csv(f'{store_path}/{if_replace_pronoun_path}/{reason_path}/{date}_{inference}_r_ds_{model}.csv')
    print(f'結果已存到以下路徑：{store_path}/{if_replace_pronoun_path}/{reason_path}/{date}_{inference}_r_ds_{model}.csv')
# %%
# 參數設定
date = "11_28-2"
# ['', '_replace_pronoun']
search_ecidence_type = "_replace_pronoun"
# 1, 2
sample_num = 2
# [True, False]
has_reason = False
# 'train', 'test'
inference = "train"
# gpt-3.5-turbo, gpt-4o-mini, Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf, Meta-Llama-3.1-8B-Instruct-Doctor.Q4_K_M.gguf
model = "Meta-Llama-3.1-8B-Instruct-Doctor.Q4_K_M.gguf"
# True, False
parse = False
# webmd, bc5cdr
data = "bc5cdr"

main(date=date, inference=inference, search_ecidence_type=search_ecidence_type, sample_num=sample_num, has_reason=has_reason, model=model, parse=parse, data=data)