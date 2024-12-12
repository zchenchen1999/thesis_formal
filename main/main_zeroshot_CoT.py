# %%
from tqdm import tqdm
tqdm.pandas(desc="Processing rows")
import pandas as pd

# webmd prompt
from RE_prompt import webMD_prompt_system_reasoning
from RE_prompt import webMD_vanilla_ICL_prompt_user_reasoning
from RE_prompt import webMD_S2PR_evidence_prompt_user_reasoning
# bc5cdr prompt
from RE_prompt import bc5cdr_prompt_system_reasoning
from RE_prompt import bc5cdr_vanilla_ICL_prompt_user_reasoning
from RE_prompt import bc5cdr_S2PR_evidence_prompt_user_reasoning

# 找尋 evidence path
from create_sentences import get_vertex_set, get_sentences
from evidence_path_rule import extract_path, get_evidence_by_entity_pair
from evidence_path_graph import get_evidence_sents
# 找尋 umls path
# from evidence_path_umls import get_umls_evidence

import utils
from open_llm import get_open_llm_result
# %%
def get_open_llm_stage1(row, search_ecidence_type="", model="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", data="webmd"):

    context = row[f'text{search_ecidence_type}']
    sents = row[f'sents{search_ecidence_type}']
    head_mention = row['drugs'][0]
    tail_mention = row['symptoms'][0]
    all_drug_metions = row['all_drug']
    all_symptom_mentions = row['all_symptom']
    # spacy_entity = row[f'spacy_entity{search_ecidence_type}']
    scispacy_entity = row[f'scispacy_entity{search_ecidence_type}']

    print(f"head: {head_mention}, tail: {tail_mention}")

    if (data == "webmd"):
        # CoT prompt 最後引導輸出答案字串
        answer_prefix = f"Therefore, the answer (i.e. the relation between {head_mention} and {tail_mention}, just answer one of A, B, C or D, and do not answer the other words.) is "
        system_prompt = webMD_prompt_system_reasoning
        ICL_user_prompt = webMD_vanilla_ICL_prompt_user_reasoning
        evidence_user_prompt = webMD_S2PR_evidence_prompt_user_reasoning
    else:
        # CoT prompt 最後引導輸出答案字串
        answer_prefix = f"Therefore, the answer (i.e. the relation between {head_mention} and {tail_mention}, just answer one of A or B, and do not answer the other words.) is "
        system_prompt = bc5cdr_prompt_system_reasoning
        ICL_user_prompt = bc5cdr_vanilla_ICL_prompt_user_reasoning
        evidence_user_prompt = bc5cdr_S2PR_evidence_prompt_user_reasoning

    ###################################
    # 合併各種 entity (用於 search path)
    ###################################
    entity_list = utils.add_all_entities(all_drug_metions, all_symptom_mentions, scispacy_entity)
    # entity_list = utils.add_all_entities(all_drug_metions, all_symptom_mentions)

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


    print(f"本次使用模型為： {model}")
    # original
    if (context == ""):
        original_reasoning = {}
    else:
        llm_var_stage1 = {
            'context': context,
            'head_entity': head_mention,
            'tail_entity': tail_mention,
            'reasoning': "",
            'anwser_prefix': ""
        }
        if (model == "gpt-3.5-turbo" or model == "gpt-4o-mini"):
            original_reasoning, _, _ = utils.get_llm_result(system_prompt, ICL_user_prompt, var_dict=llm_var_stage1, model=model)
        else:
            original_reasoning = get_open_llm_result(system_prompt, ICL_user_prompt, var_dict=llm_var_stage1, data=data, parse=None)

    # consecutive
    if (consecutive_string == ""):
        consecutive_reasoning = {}
    else:
        # consecutive_result = consecutive_string
        llm_var_stage1 = {
            'context': context,
            'evidence': consecutive_string,
            'head_entity': head_mention,
            'tail_entity': tail_mention,
            'reasoning': "",
            'anwser_prefix': ""
        }
        if (model == "gpt-3.5-turbo" or model == "gpt-4o-mini"):
            consecutive_reasoning, _, _ = utils.get_llm_result(system_prompt, evidence_user_prompt, var_dict=llm_var_stage1, model=model)
        else:
            consecutive_reasoning = get_open_llm_result(system_prompt, evidence_user_prompt, var_dict=llm_var_stage1, data=data, parse=None)
    
    # multi_hop
    if (multi_hop_string == ""):
        multi_hop_reasoning = {}
    else:
        # multi_hop_result = multi_hop_string
        llm_var_stage1 = {
            'context': context,
            'evidence': multi_hop_string,
            'head_entity': head_mention,
            'tail_entity': tail_mention,
            'reasoning': "",
            'anwser_prefix': ""
        }
        if (model == "gpt-3.5-turbo" or model == "gpt-4o-mini"):
            multi_hop_reasoning, _, _ = utils.get_llm_result(system_prompt, evidence_user_prompt, var_dict=llm_var_stage1, model=model)
        else:
            multi_hop_reasoning = get_open_llm_result(system_prompt, evidence_user_prompt, var_dict=llm_var_stage1, data=data, parse=None)

    # default
    if (default_string == ""):
        default_reasoning = {}
    else:
        # default_result = default_string
        llm_var_stage1 = {
            'context': context,
            'evidence': default_string,
            'head_entity': head_mention,
            'tail_entity': tail_mention,
            'reasoning': "",
            'anwser_prefix': ""
        }
        if (model == "gpt-3.5-turbo" or model == "gpt-4o-mini"):
            default_reasoning, _, _ = utils.get_llm_result(system_prompt, evidence_user_prompt, var_dict=llm_var_stage1, model=model)
        else:
            default_reasoning = get_open_llm_result(system_prompt, evidence_user_prompt, var_dict=llm_var_stage1, data=data, parse=None)

    return original_reasoning, consecutive_reasoning, multi_hop_reasoning, default_reasoning

def get_open_llm_stage2(row, search_ecidence_type="", model="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", data="webmd", parse=True):

    context = row[f'text{search_ecidence_type}']
    sents = row[f'sents{search_ecidence_type}']
    head_mention = row['drugs'][0]
    tail_mention = row['symptoms'][0]
    all_drug_metions = row['all_drug']
    all_symptom_mentions = row['all_symptom']
    # spacy_entity = row[f'spacy_entity{search_ecidence_type}']
    scispacy_entity = row[f'scispacy_entity{search_ecidence_type}']

    # reasoning
    original_reasoning = row['original_reasoning']
    consecutive_reasoning = row['consecutive_reasoning']
    multi_hop_reasoning = row['multi_hop_reasoning']
    default_reasoning = row['default_reasoning']

    print(f"head: {head_mention}, tail: {tail_mention}")

    if (data == "webmd"):
        # CoT prompt 最後引導輸出答案字串
        answer_prefix = f"Therefore, the answer (i.e. the relation between {head_mention} and {tail_mention}, just answer one of A, B, C or D, and do not answer the other words.) is: "
        system_prompt = webMD_prompt_system_reasoning
        ICL_user_prompt = webMD_vanilla_ICL_prompt_user_reasoning
        evidence_user_prompt = webMD_S2PR_evidence_prompt_user_reasoning
    else:
        # CoT prompt 最後引導輸出答案字串
        answer_prefix = f"Therefore, the answer (i.e. the relation between {head_mention} and {tail_mention}, just answer one of A or B, and do not answer the other words.) is: "
        system_prompt = bc5cdr_prompt_system_reasoning
        ICL_user_prompt = bc5cdr_vanilla_ICL_prompt_user_reasoning
        evidence_user_prompt = bc5cdr_S2PR_evidence_prompt_user_reasoning

    ###################################
    # 合併各種 entity (用於 search path)
    ###################################
    entity_list = utils.add_all_entities(all_drug_metions, all_symptom_mentions, scispacy_entity)
    # entity_list = utils.add_all_entities(all_drug_metions, all_symptom_mentions)

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


    #################
    # 詢問 LLM 分類結果
    #################
    original_result = {}
    consecutive_result = {}
    multi_hop_result = {}
    default_result = {}

    print(f"本次使用模型為： {model}")
    # original
    if (original_reasoning == "{}"):
        original_result = {}
    else:
        llm_var_stage2 = {
            'context': context,
            'head_entity': head_mention,
            'tail_entity': tail_mention,
            'reasoning': original_reasoning,
            'anwser_prefix': answer_prefix
        }
        if (model == "gpt-3.5-turbo" or model == "gpt-4o-mini"):
            original_response, _, _ = utils.get_llm_result(system_prompt, ICL_user_prompt, var_dict=llm_var_stage2, model=model)
        else:
            original_response = get_open_llm_result(system_prompt, ICL_user_prompt, var_dict=llm_var_stage2, data=data, parse=parse)
        original_result['response'] = original_response
        original_result['reasoning'] = original_reasoning

    # consecutive
    if (consecutive_reasoning == "{}"):
        consecutive_result = {}
    else:
        # consecutive_result = consecutive_string
        llm_var_stage2 = {
            'context': context,
            'evidence': consecutive_string,
            'head_entity': head_mention,
            'tail_entity': tail_mention,
            'reasoning': consecutive_reasoning,
            'anwser_prefix': answer_prefix
        }
        if (model == "gpt-3.5-turbo" or model == "gpt-4o-mini"):
            consecutive_response, _, _ = utils.get_llm_result(system_prompt, evidence_user_prompt, var_dict=llm_var_stage2, model=model)
        else:
            consecutive_response = get_open_llm_result(system_prompt, evidence_user_prompt, var_dict=llm_var_stage2, data=data, parse=parse)
        consecutive_result['response'] = consecutive_response
        consecutive_result['reasoning'] = consecutive_reasoning
    
    # multi_hop
    if (multi_hop_reasoning == "{}"):
        multi_hop_result = {}
    else:
        # multi_hop_result = multi_hop_string
        llm_var_stage2 = {
            'context': context,
            'evidence': multi_hop_string,
            'head_entity': head_mention,
            'tail_entity': tail_mention,
            'reasoning': multi_hop_reasoning,
            'anwser_prefix': answer_prefix
        }
        if (model == "gpt-3.5-turbo" or model == "gpt-4o-mini"):
            multi_hop_response, _, _ = utils.get_llm_result(system_prompt, evidence_user_prompt, var_dict=llm_var_stage2, model=model)
        else:
            multi_hop_response = get_open_llm_result(system_prompt, evidence_user_prompt, var_dict=llm_var_stage2, data=data, parse=parse)
        multi_hop_result['response'] = multi_hop_response
        multi_hop_result['reasoning'] = multi_hop_reasoning

    # default
    if (default_reasoning == "{}"):
        default_result = {}
    else:
        # default_result = default_string
        llm_var_stage2 = {
            'context': context,
            'evidence': default_string,
            'head_entity': head_mention,
            'tail_entity': tail_mention,
            'reasoning': default_reasoning,
            'anwser_prefix': answer_prefix
        }
        if (model == "gpt-3.5-turbo" or model == "gpt-4o-mini"):
            default_response, _, _ = utils.get_llm_result(system_prompt, evidence_user_prompt, var_dict=llm_var_stage2, model=model)
        else:
            default_response = get_open_llm_result(system_prompt, evidence_user_prompt, var_dict=llm_var_stage2, data=data, parse=parse)
        default_result['response'] = default_response
        default_result['reasoning'] = default_reasoning

    ##################
    # 轉為 json 並縮排
    ##################
    original_result = utils.dict_to_json(original_result)
    consecutive_result = utils.dict_to_json(consecutive_result)
    multi_hop_result = utils.dict_to_json(multi_hop_result)
    default_result = utils.dict_to_json(default_result)

    return original_result, consecutive_result, multi_hop_result, default_result

# GPT
def get_all_triplets(row, search_ecidence_type="", model="gpt-3.5-turbo", data="webmd"):

    context = row[f'text{search_ecidence_type}']
    sents = row[f'sents{search_ecidence_type}']
    head_mention = row['drugs'][0]
    tail_mention = row['symptoms'][0]
    all_drug_metions = row['all_drug']
    all_symptom_mentions = row['all_symptom']
    # spacy_entity = row[f'spacy_entity{search_ecidence_type}']
    scispacy_entity = row[f'scispacy_entity{search_ecidence_type}']

    print(f"head: {head_mention}, tail: {tail_mention}")

    if (data == "webmd"):
        # CoT prompt 最後引導輸出答案字串
        answer_prefix = f"Therefore, the answer (i.e. the relation between {head_mention} and {tail_mention}, just answer one of A, B, C or D, and do not answer the other words.) is "

        system_prompt = webMD_prompt_system_reasoning
        ICL_user_prompt = webMD_vanilla_ICL_prompt_user_reasoning
        evidence_user_prompt = webMD_S2PR_evidence_prompt_user_reasoning
    else:
        # CoT prompt 最後引導輸出答案字串
        answer_prefix = f"Therefore, the answer (i.e. the relation between {head_mention} and {tail_mention}, just answer one of A or B, and do not answer the other words.) is "

        system_prompt = bc5cdr_prompt_system_reasoning
        ICL_user_prompt = bc5cdr_vanilla_ICL_prompt_user_reasoning
        evidence_user_prompt = bc5cdr_S2PR_evidence_prompt_user_reasoning

    ###################################
    # 合併各種 entity (用於 search path)
    ###################################
    entity_list = utils.add_all_entities(all_drug_metions, all_symptom_mentions, scispacy_entity)
    # entity_list = utils.add_all_entities(all_drug_metions, all_symptom_mentions)

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
        llm_var_stage1 = {
            'context': context,
            'head_entity': head_mention,
            'tail_entity': tail_mention,
            'reasoning': "",
            'anwser_prefix': ""
        }
        original_reasoning, _, _ = utils.get_llm_result(system_prompt, ICL_user_prompt, var_dict=llm_var_stage1, model=model)
        llm_var_stage2 = {
            'context': context,
            'head_entity': head_mention,
            'tail_entity': tail_mention,
            'reasoning': original_reasoning,
            'anwser_prefix': answer_prefix
        }
        original_response, original_top_classes, original_final_prompt = utils.get_llm_result(
            system_prompt, ICL_user_prompt, var_dict=llm_var_stage2, model=model)
        original_result['response'] = original_response
        original_result['reasoning'] = original_reasoning
        original_result['top_logprobs'] = original_top_classes
        original_result['prompt'] = original_final_prompt

    # consecutive
    if (consecutive_string == ""):
        consecutive_result = {}
    else:
        # consecutive_result = consecutive_string
        llm_var_stage1 = {
            'context': context,
            'evidence': consecutive_string,
            'head_entity': head_mention,
            'tail_entity': tail_mention,
            'reasoning': "",
            'anwser_prefix': ""
        }
        consecutive_reasoning, _, _ = utils.get_llm_result(
            system_prompt, evidence_user_prompt, var_dict=llm_var_stage1, model=model)
        llm_var_stage2 = {
            'context': context,
            'evidence': consecutive_string,
            'head_entity': head_mention,
            'tail_entity': tail_mention,
            'reasoning': consecutive_reasoning,
            'anwser_prefix': answer_prefix
        }
        consecutive_response, consecutive_top_classes, consecutive_final_prompt = utils.get_llm_result(
            system_prompt, evidence_user_prompt, var_dict=llm_var_stage2, model=model)
        consecutive_result['response'] = consecutive_response
        consecutive_result['reasoning'] = consecutive_reasoning
        consecutive_result['top_logprobs'] = consecutive_top_classes
        consecutive_result['evidence'] = consecutive_string
        consecutive_result['prompt'] = consecutive_final_prompt
    
    # multi_hop
    if (multi_hop_string == ""):
        multi_hop_result = {}
    else:
        # multi_hop_result = multi_hop_string
        llm_var_stage1 = {
            'context': context,
            'evidence': multi_hop_string,
            'head_entity': head_mention,
            'tail_entity': tail_mention,
            'reasoning': "",
            'anwser_prefix': ""
        }
        multi_hop_reasoning, _, _ = utils.get_llm_result(
            system_prompt, evidence_user_prompt, var_dict=llm_var_stage1, model=model)
        llm_var_stage2 = {
            'context': context,
            'evidence': multi_hop_string,
            'head_entity': head_mention,
            'tail_entity': tail_mention,
            'reasoning': multi_hop_reasoning,
            'anwser_prefix': answer_prefix
        }
        multi_hop_response, multi_hop_top_classes, multi_hop_final_prompt = utils.get_llm_result(
            system_prompt, evidence_user_prompt, var_dict=llm_var_stage2, model=model)
        multi_hop_result['response'] = multi_hop_response
        multi_hop_result['reasoning'] = multi_hop_reasoning
        multi_hop_result['top_logprobs'] = multi_hop_top_classes
        multi_hop_result['evidence'] = multi_hop_string
        multi_hop_result['prompt'] = multi_hop_final_prompt
        multi_hop_result['sentence_index'] = sentences_index

    # default
    if (default_string == ""):
        default_result = {}
    else:
        # default_result = default_string
        llm_var_stage1 = {
            'context': context,
            'evidence': default_string,
            'head_entity': head_mention,
            'tail_entity': tail_mention,
            'reasoning': "",
            'anwser_prefix': ""
        }
        default_reasoning, _, _ = utils.get_llm_result(
            system_prompt, evidence_user_prompt, var_dict=llm_var_stage1, model=model)
        llm_var_stage2 = {
            'context': context,
            'evidence': default_string,
            'head_entity': head_mention,
            'tail_entity': tail_mention,
            'reasoning': default_reasoning,
            'anwser_prefix': answer_prefix
        }
        default_response, default_top_classes, default_final_prompt = utils.get_llm_result(
            system_prompt, evidence_user_prompt, var_dict=llm_var_stage2, model=model)
        default_result['response'] = default_response
        default_result['reasoning'] = default_reasoning
        default_result['top_logprobs'] = default_top_classes
        default_result['evidence'] = default_string
        default_result['prompt'] = default_final_prompt

    ##################
    # 轉為 json 並縮排
    ##################
    original_result = utils.dict_to_json(original_result)
    consecutive_result = utils.dict_to_json(consecutive_result)
    multi_hop_result = utils.dict_to_json(multi_hop_result)
    default_result = utils.dict_to_json(default_result)

    # return "", "", "", default_string
    return original_result, consecutive_result, multi_hop_result, default_result


def main_openllm_stage1(date, inference, search_ecidence_type, model="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", data="webmd"):
    if (data == "webmd"):
        df_train = pd.read_csv('/home/zchenchen1999/thesis_formal/main/preprocessed_data/WebMD/WebMD_annotated_v2_exploded_reasoning_train.csv')
    else:
        df_train = pd.read_csv('/home/zchenchen1999/thesis_formal/main/preprocessed_data/BC5CDR/BC5CDR_preprocess_train_smallset.csv')

    # 轉換成 list
    df_train = utils.str_to_list(df_train, 
            ['drugs', 'symptoms', 'relations', 
            'sents', 'spacy_entity', 'scispacy_entity',
            'sents_replace_pronoun', 'spacy_entity_replace_pronoun', 'scispacy_entity_replace_pronoun',
            'all_drug', 'all_symptom']
        )
    
    # df_train = df_train.head(2)
    print(df_train.shape)

    # stage1
    df_train[['original_reasoning', 'consecutive_reasoning', 'multi_hop_reasoning', 'default_reasoning']] = df_train.progress_apply(
        lambda row: get_open_llm_stage1(row, search_ecidence_type=search_ecidence_type, model=model, data=data), axis=1, result_type='expand')
    
    path = f'/home/zchenchen1999/thesis_formal/main/result/{data}/replace_pronoun/zeroshot_cot/{date}_{data}_{inference}_r_dsm_{model}_stage1.csv'
    df_train.to_csv(path)
    print(f'結果已存到以下路徑：{path}')
    return path

def main_openllm_stage2(date, inference, search_ecidence_type, model="Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", data="webmd", file_path="", parse=True):

    df_train = pd.read_csv(file_path)

    # 轉換成 list
    df_train = utils.str_to_list(df_train, 
            ['drugs', 'symptoms', 'relations', 
            'sents', 'spacy_entity', 'scispacy_entity',
            'sents_replace_pronoun', 'spacy_entity_replace_pronoun', 'scispacy_entity_replace_pronoun',
            'all_drug', 'all_symptom']
        )
    
    # df_train = df_train.head(5)
    print(df_train.shape)

    # inference
    # stage2
    df_train[['original_result', 'consecutive_result', 'multi_hop_result', 'default_result']] = df_train.progress_apply(
        lambda row: get_open_llm_stage2(row, search_ecidence_type=search_ecidence_type, model=model, data=data, parse=parse), axis=1, result_type='expand')
    df_train.to_csv(f'/home/zchenchen1999/thesis_formal/main/result/{data}/replace_pronoun/zeroshot_cot/{date}_{data}_{inference}_r_dsm_{model}_stage2.csv')
    print(f'結果已存到以下路徑：/home/zchenchen1999/thesis_formal/main/result/{data}/replace_pronoun/zeroshot_cot/{date}_{data}_{inference}_r_dsm_{model}_stage2.csv')

def main_gpt_all_stage(date, inference, search_ecidence_type, model="gpt-3.5-turbo", data="webmd"):
    if (data == "webmd"):
        df_train = pd.read_csv('/home/zchenchen1999/thesis_formal/main/preprocessed_data/WebMD/WebMD_annotated_v2_exploded_reasoning_train.csv')
    else:
        df_train = pd.read_csv('/home/zchenchen1999/thesis_formal/main/preprocessed_data/BC5CDR/BC5CDR_preprocess_train_smallset.csv')

    # 轉換成 list
    df_train = utils.str_to_list(df_train, 
            ['drugs', 'symptoms', 'relations', 
            'sents', 'spacy_entity', 'scispacy_entity',
            'sents_replace_pronoun', 'spacy_entity_replace_pronoun', 'scispacy_entity_replace_pronoun',
            'all_drug', 'all_symptom']
        )
    
    # df_train = df_train.head(5)
    print(df_train.shape)

    # inference
    df_train[['original_result', 'consecutive_result', 'multi_hop_result', 'default_result']] = df_train.progress_apply(
        lambda row: get_all_triplets(row, search_ecidence_type=search_ecidence_type, model=model, data=data), axis=1, result_type='expand')
    df_train = df_train[['sents', 'drugs', 'symptoms', 'ground_truth', 'original_result', 'consecutive_result', 'multi_hop_result', 'default_result']]
    df_train.to_csv(f'/home/zchenchen1999/thesis_formal/main/result/{data}/replace_pronoun/zeroshot_cot/{date}_{data}_{inference}_r_dsm_{model}.csv')
    print(f'結果已存到以下路徑：/home/zchenchen1999/thesis_formal/main/result/{data}/replace_pronoun/zeroshot_cot/{date}_{data}_{inference}_r_dsm_{model}.csv')

# %%
###############
# GPT inference
###############

# 參數設定
date = "12_05"
# ['', '_replace_pronoun']
search_ecidence_type = "_replace_pronoun"
# 'train', 'test'
inference = "train"
# webmd, bc5cdr
data = "bc5cdr"
# gpt-3.5-turbo, gpt-4o-mini, Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf, Meta-Llama-3.1-8B-Instruct-Doctor.Q4_K_M.gguf
model = "gpt-4o-mini"

main_gpt_all_stage(date=date, inference=inference, search_ecidence_type=search_ecidence_type, model=model, data=data)
# # %%
# #############################
# # Open LLM inference - stage1
# #############################
# print("正在處理webmd-1")
# # 參數設定
# date = "12_05"
# # ['', '_replace_pronoun']
# search_ecidence_type = "_replace_pronoun"
# # 'train', 'test'
# inference = "train"
# # webmd, bc5cdr
# data = "webmd"
# # Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf, Meta-Llama-3.1-8B-Instruct-Doctor.Q4_K_M.gguf
# model = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"

# path_webmd = main_openllm_stage1(date=date, inference=inference, search_ecidence_type=search_ecidence_type, model=model, data=data)
# # %%
# #############################
# # Open LLM inference - stage2
# #############################
# print("正在處理webmd-2")
# # 參數設定
# date = "12_05"
# # ['', '_replace_pronoun']
# search_ecidence_type = "_replace_pronoun"
# # 'train', 'test'
# inference = "train"
# # webmd, bc5cdr
# data = "webmd"
# # Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf, Meta-Llama-3.1-8B-Instruct-Doctor.Q4_K_M.gguf
# model = "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
# # 檔案路徑（stage1 結果儲存路徑）
# file_path = path_webmd
# # file_path = "/home/zchenchen1999/thesis_formal/main/result/webmd/replace_pronoun/zeroshot_cot/12_05_webmd_train_r_dsm_Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf_stage1.csv"
# # True, Fasle
# parse = True

# main_openllm_stage2(date=date, inference=inference, search_ecidence_type=search_ecidence_type, model=model, data=data, file_path=file_path, parse=parse)

# %%
