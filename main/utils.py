# %%
import ast
import pandas as pd
from typing import Dict, Any, List
import re
import json
import numpy as np
import difflib
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from sklearn.metrics.pairwise import cosine_similarity



from langchain_core.prompts import ChatPromptTemplate
# from langchain.chat_models import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

from create_sentences import get_vertex_set, get_sentences
from evidence_path_rule import extract_path, get_evidence_by_entity_pair
from evidence_path_graph import get_evidence_sents

import spacy
import scispacy
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector

nlp_umls = spacy.load("en_core_sci_lg")
nlp_umls.add_pipe("abbreviation_detector")
nlp_umls.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})

# 讀取環境變數
from dotenv import load_dotenv
import os
# 加載 .env 文件中的環境變數
load_dotenv('./enviroment.env')
# 讀取環境變數
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
UMLS_API_KEY = os.getenv("UMLS_API_KEY")
# %%
class CustomHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        
        formatted_prompts = "\n".join(prompts)
        # _log.info(f"\n\n==========================Prompt==========================\n{formatted_prompts}\n=========================================================\n\n")
        print((f"==========================Prompt==========================\n{formatted_prompts}\n=========================================================\n\n"))

###########################################
# LLM inference
###########################################
def get_llm_result(system_prompt, user_prompt, var_dict={}, model="gpt-4o-mini"):
    # chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=model, 
    # temperature=0.7, logprobs=True, top_logprobs=10)
    chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=model, 
    temperature=0.7, logprobs=True, top_logprobs=10, callbacks=[CustomHandler()])

    prompt = ChatPromptTemplate.from_messages(
        messages = [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
    )

    chain = prompt | chat

    # final_prompt = prompt.format({"demonstrations": demonstrations, "context": context, "evidence": evidence, "head_entity": head_entity, "tail_entity": tail_entity})
    final_prompt = prompt.format(**var_dict)

    response = chain.invoke(var_dict)
    top_classes = response.response_metadata["logprobs"]["content"][0]['top_logprobs']

    return response.content, top_classes, final_prompt

###########################################
# 取得各種 evidence string
###########################################
# consecutive
def get_consecutive_string(head_entity, tail_entity, sents, all_drugs, all_symptoms, spacy_entity, scispacy_entity):
    # 結合所有 entity
    entity_list = add_all_entities(all_drugs, all_symptoms, spacy_entity, scispacy_entity)
    # 取得 vertex
    vertexSet = get_vertex_set(entity_list, sents)
    # 取得 consecutive 的 path
    consecutive = extract_path(sents, vertexSet, True, path_type="consecutive")
    # 取得 string
    consecutive_string = get_evidence_by_entity_pair(head_entity, tail_entity, entity_list, sents, consecutive)
    return consecutive_string

# multi-hop
def get_multi_hop_string(head_entity, tail_entity, sents, all_drugs, all_symptoms, spacy_entity, scispacy_entity):
    # 結合所有 entity
    entity_list = add_all_entities(all_drugs, all_symptoms, scispacy_entity)
    _, _, graph_base_string = get_evidence_sents(sents, entity_list, head_entity, tail_entity)
    return graph_base_string

# default
def get_default_string(head_entity, tail_entity, sents, all_drugs, all_symptoms, spacy_entity, scispacy_entity):
    # 結合所有 entity
    entity_list = add_all_entities(all_drugs, all_symptoms, spacy_entity, scispacy_entity)
    # 取得 vertex
    vertexSet = get_vertex_set(entity_list, sents)
    # 取得 consecutive 的 path
    default = extract_path(sents, vertexSet, True, path_type="default")
    # 取得 string
    default_string = get_evidence_by_entity_pair(head_entity, tail_entity, entity_list, sents, default)
    return default_string

###########################################
# 合併所有種類 entity
###########################################
def add_all_entities(*args):
    unique_elements = set()

    for arg in args:
        for item in arg:
            if isinstance(item, tuple):
                unique_elements.add(item[0])  # 如果是 tuple，取第 0 項
            else:
                unique_elements.add(item)  # 如果不是 tuple，直接取值
    
    return list(unique_elements)

###########################################
# explode_relation 子 function
# 對於某對 entity pair 查詢並分配 ground truth
###########################################
def assign_ground_truth(row):
    drug = row['drugs'][0]
    symptom = row['symptoms'][0]
    relations = row['relations']
    
    for relation in relations:
        if relation[0] == drug and relation[2] == symptom:
            return relation[1]  # Return the relation if both drug and symptom match
    return 'none_of_above'  # If no match is found

###########################################
# 格式轉換
###########################################
# dict 轉為 json 並且縮排
def dict_to_json(target_dict):
    return json.dumps(target_dict, indent=4)

# json 轉為 dict
def json_to_dict(json_str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None  # 如果解析失敗，返回 None 或者其他錯誤處理邏輯

# 轉換 list string 成真正的 list
def str_to_list(df, columns):
    for c in columns:
        df[c] = df[c].apply(lambda x: ast.literal_eval(x))
    return df

# 將 relation 三元組轉換成單筆資料
def explode_relation(df):
    df_exploded = df.explode('drugs').explode('symptoms')
    df_exploded['ground_truth'] = df_exploded.apply(assign_ground_truth, axis=1)
    return df_exploded

##########################################
# parse LLM 輸出的文字
##########################################
# 匹配 {"key": "value"} 的字串
def relative_sentence_parse_code_string_to_ans(code_string):
    # 正則表達式匹配 JSON 結構
    pattern = re.compile(r'\{(?:[^{}]|(?:\{(?:[^{}]|(?:\{[^{}]*\}))*\}))*\}')

    matches = pattern.findall(code_string)
    ans = []

    for match in matches:
        try:
            # 將字串轉換為 JSON
            json_object = json.loads(match.replace("\\", ""))
            # json_objects.append(json_object)
            sentence = list(json_object.values())[0]
            sentence = sentence.replace("**", "")
            ans.append(sentence)
        except json.JSONDecodeError:
            continue
    
    return ans

##########################################
# 詞語相似度比對
##########################################
def find_most_similar_word_scispacy(word, word_list):
    # 將詞轉換為 spaCy 的文檔格式
    word_doc = nlp_umls(word)
    # 計算每個詞與原詞的相似度，選擇最相似的詞
    most_similar_word = max(word_list, key=lambda x: word_doc.similarity(nlp_umls(x)))
    return most_similar_word

def find_most_similar_word_difflib(word, word_list):
    # 使用 difflib.get_close_matches 找到與 word 最相似的詞
    closest_match = difflib.get_close_matches(word, word_list, n=1, cutoff=0.85)
    return closest_match[0] if closest_match else None
# %%
##########################################
# scispacy 操作
##########################################

# 使用 scispacy 連結 mentions 到 UMLS 的 concepts (list)
def get_concepts_in_umls_scispacy(mention):
    doc = nlp_umls(mention)
    # 沒有抓到
    if (len(doc.ents) < 1):
        return None, None
    doc_entity = doc.ents[0]
    linker = nlp_umls.get_pipe("scispacy_linker")
    # 沒有 linking 到目標
    if (len(doc_entity._.kb_ents) < 1):
        return None, None
    else:
        match_entity = doc_entity._.kb_ents[0]
        match_entity_id = linker.kb.cui_to_entity[match_entity[0]].concept_id
        match_entity_name = linker.kb.cui_to_entity[match_entity[0]].canonical_name
        return match_entity_name, match_entity_id
    
# 定義函數來獲取單詞的向量
def get_embedding(word):
    return nlp_umls(word).vector

# 計算兩個向量之間的餘弦相似度
def cosine_similarity_vectors(vec1, vec2):
    return cosine_similarity([vec1], [vec2])[0][0]

##########################################
# UMLS KG 操作
##########################################
# 創建一個包含重試和超時的會話
def create_session_with_retries():
    session = requests.Session()

    # 定義重試策略
    retry_strategy = Retry(
        total=5,  # 總共最多重試 5 次
        backoff_factor=1,  # 每次重試前的等待時間 (1秒, 2秒, 4秒...)
        status_forcelist=[500, 502, 503, 504],  # 對這些狀態碼進行重試
        allowed_methods=["GET", "POST"],  # 允許哪些 HTTP 方法使用重試
        raise_on_status=False  # 重試時不拋出錯誤
    )

    # 將重試策略附加到 HTTPAdapter
    adapter = HTTPAdapter(max_retries=retry_strategy)
    
    # 將 adapter 裝載到 session 中
    session.mount("http://", adapter)
    session.mount("https://", adapter)

    return session

# 使用 session 發送 GET 請求，並設置超時時間
def fetch_with_timeout(url, timeout_duration, headers):
    session = create_session_with_retries()
    for attempt in range(5):  # 重試最多5次
        try:
            response = session.get(url, timeout=timeout_duration)  # 設置超時
            response.raise_for_status()  # 如果狀態碼為錯誤，會拋出 HTTPError
            return response
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            print(f"Attempt {attempt + 1} failed with error: {e}. Retrying...")
        except requests.exceptions.RequestException as e:
            # 其他類型的請求錯誤（非超時和連接錯誤），不進行重試
            return f"Request failed: {e}"

    # 如果重試5次後仍失敗，返回錯誤信息
    return "Request failed after 5 retries due to Timeout or ConnectionError"

# 使用 UMLS API 連結 mentions 到 UMLS 的 concepts (單一 mention)
def get_concept_in_umls_api(name, source='all'):
    # UMLS 基本 URI
    base_uri = 'https://uts-ws.nlm.nih.gov/rest'

    # 搜索 URI
    if (source == "all"):
        search_uri = f'{base_uri}/search/2023AB?partialSearch=true&apiKey={UMLS_API_KEY}&string={name}'
    else:
        search_uri = f'{base_uri}/search/2023AB?partialSearch=true&sabs={source}&apiKey={UMLS_API_KEY}&string={name}'

    # 添加 Authorization Header
    headers = {
        'Authorization': f'Bearer {UMLS_API_KEY}'
    }

    search_response = fetch_with_timeout(search_uri, 5, headers=headers)
    # search_response = requests.get(search_uri)
    # 檢查響應狀態並打印結果
    if isinstance(search_response, requests.Response):
        if search_response.status_code == 200:
            search_results = search_response.json()
            try:
                concept_id = search_results['result']['results'][0]['ui']
                concept_name = search_results['result']['results'][0]['name']
            except:
                return None, None
            return concept_name, concept_id 
        else:
            print(f"Failed to retrieve search results: {search_response.status_code}")
            return None, None
    else:
        # 如果 search_response 不是 response 對象（可能是錯誤訊息的字串），打印錯誤信息
        print(f"Search 「{name}」 Error occurred: {search_response}")
        return None, None
def get_mention_concept(mention, mention_type="drug"):
    # print(f"linking:  {mention}")
    source = {
        'drug': 'RXNORM',
        'symptom': 'SNOMEDCT_US',
        'both': 'RXNORM,SNOMEDCT_US'
    }

    tmp = mention

    # 先使用 UMLS 官方 api 查詢 concept (特定 source)
    if (mention_type=='None'):
        tmp_from_umls_api_name, tmp_from_umls_api_id = get_concept_in_umls_api(tmp, source='all')
    else:
        tmp_from_umls_api_name, tmp_from_umls_api_id = get_concept_in_umls_api(tmp, source=source[mention_type])

    if (tmp_from_umls_api_id != None):
        return tmp_from_umls_api_name, tmp_from_umls_api_id
    else:
        # 不指定 source
        tmp_from_umls_api_name, tmp_from_umls_api_id = get_concept_in_umls_api(tmp, source="all")
        if (tmp_from_umls_api_id != None):
            return tmp_from_umls_api_name, tmp_from_umls_api_id
        else:
            # 用 scispacy 對應
            tmp_from_umls_api_name, tmp_from_umls_api_id = get_concepts_in_umls_scispacy(tmp)
            return tmp_from_umls_api_name, tmp_from_umls_api_id
