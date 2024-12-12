# %%
from collections import defaultdict
import json
from tqdm import tqdm
from create_sentences import get_vertex_set
import itertools
# %%
# Remove redundency
def remove_redundency(path):
    for n1 in path.keys():
        for n2 in path[n1].keys():
            hash_set = set()
            new_list = []
            for t in path[n1][n2]:
                if tuple(t[0]) not in hash_set:
                    hash_set.add(tuple(t[0]))
                    new_list.append(t[0])
            unique_sorted_list = sorted(set(itertools.chain(*new_list)))
            path[n1][n2] = unique_sorted_list
            
    return path
# %%
# 每一個 document 都會執行
def extract_path(sents, vertexSet, keep_sent_order, path_type="merge"):
    # sents = sents
    nodes = [[] for _ in range(len(sents))]
    e2e_sent = defaultdict(dict)

    # create mention's list for each sentence
    # 紀錄每一個句子的 enity
    for ns_no, ns in enumerate(vertexSet):
        for n in ns:
            sent_id = int(n['sent_id'])
            nodes[sent_id].append(ns_no)

    # 找 2 entity 出現在同一句子中
    # 每一句子
    for sent_id in range(len(sents)):
        # 跑過這句子中所有的 entity
        for n1 in nodes[sent_id]:
            for n2 in nodes[sent_id]:
                # 相同不計
                if n1 == n2:
                    continue
                if n2 not in e2e_sent[n1]:
                    e2e_sent[n1][n2] = set()
                # 紀錄某兩個 entity 出現的句子 id
                e2e_sent[n1][n2].add(sent_id)

    # 2-hop Path
    path_two = defaultdict(dict)

    # entityNum: 這個 document 的 entity 數量
    # n1, n2, n3 分別為 document 中的某三個 entity
    entityNum = len(vertexSet)
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n1 == n2:
                continue
            for n3 in range(entityNum):
                # 三個 entity 都要不同，重複則不找
                if n3 == n1 or n3 == n2:
                    continue

                # n3 是 bridge entity
                # (n3 與 n1) (n2 與 n3) 都有連結的句子 => 成立 => 不進入 if 
                # (n3 與 n1) (n2 與 n3) 其中一個無連結的句子 => 不成立 => 進入 if => contiune 下一個 n3
                if not (n3 in e2e_sent[n1] and n2 in e2e_sent[n3]):
                    continue
                for s1 in e2e_sent[n1][n3]:
                    for s2 in e2e_sent[n3][n2]:
                        # 相同句子不計
                        if s1 == s2:
                            continue
                        if n2 not in path_two[n1]:
                            # 建立紀錄兩者連接的 sentence path
                            path_two[n1][n2] = []
                        cand_sents = [s1, s2]
                        if keep_sent_order == True:
                            cand_sents.sort()
                        # 紀錄 path 與 bridge entity
                        path_two[n1][n2].append((cand_sents, n3))

    # 3-hop Path
    path_three = defaultdict(dict)
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            if n1 == n2:
                continue
            for n3 in range(entityNum):
                if n3 == n1 or n3 == n2:
                    continue

                # n3(第一個 bridge), n1(head) 有連結， n2(tail) 與 n3(第一個 bridge) 之間有 bridge entity 來連結，代表 n1, n2 之間有 3-hop 關係
                if n3 in e2e_sent[n1] and n2 in path_two[n3]:
                    # n1(head), n3(第一個 bridge) 所有相連的句子
                    for cand1 in e2e_sent[n1][n3]:
                        # 所有 n3(第一個 bridge), n2(tail) 之間的 2-hop 組合
                        for cand2 in path_two[n3][n2]:
                            # 組合中不能包含 cand1 的 sentence
                            if cand1 in cand2[0]:
                                continue
                            # 不能是與 head 相同的 entity
                            if cand2[1] == n1:
                                continue
                            if n2 not in path_three[n1]:
                                # 建立紀錄兩者連接的 3-hop sentence path
                                path_three[n1][n2] = []
                            # 3 個 sentence 
                            cand_sents = [cand1] + cand2[0]
                            if keep_sent_order:
                                cand_sents.sort()
                            path_three[n1][n2].append((cand_sents, [n3, cand2[1]]))

    # Consecutive Path
    # 連續句子
    consecutive = defaultdict(dict)
    for h in range(entityNum):
        for t in range(h + 1, entityNum):
            for n1 in vertexSet[h]:
                for n2 in vertexSet[t]:
                    # 連續句子數
                    gap = abs(n1['sent_id'] - n2['sent_id'])
                    if gap > 2:
                        continue
                    if t not in consecutive[h]:
                        consecutive[h][t] = []
                        consecutive[t][h] = []
                    if n1['sent_id'] < n2['sent_id']:
                        beg, end = n1['sent_id'], n2['sent_id']
                    else:
                        beg, end = n2['sent_id'], n1['sent_id']

                    consecutive[h][t].append([[i for i in range(beg, end + 1)]])
                    consecutive[t][h].append([[i for i in range(beg, end + 1)]])

    # Merge
    # 兩兩 entity 之間多種 path 的 merge
    merge = defaultdict(dict)
    for n1 in range(entityNum):
        for n2 in range(entityNum):
            # 先取連續路徑
            if n2 in consecutive[n1]:
                if n2 in merge[n1]:
                    pass
                else:
                    merge[n1][n2] = consecutive[n1][n2]
                # # 是否已有 path
                # if n2 in merge[n1]:
                #     # 繼續加入
                #     merge[n1][n2] += consecutive[n1][n2]
                # else:
                #     merge[n1][n2] = consecutive[n1][n2]
            # 取 2-hop
            if n2 in path_two[n1]:
                if n2 in merge[n1]:
                    # 繼續加入
                    # merge[n1][n2] += path_two[n1][n2]
                    pass
                else:
                    merge[n1][n2] = path_two[n1][n2]
            # 取 3-hop
            if n2 in path_three[n1]:
                # 是否已有 path
                if n2 in merge[n1]:
                    # 繼續加入
                    # merge[n1][n2] += path_three[n1][n2]
                    pass
                else:
                    merge[n1][n2] = path_three[n1][n2]

    # for those no path add Default Path
    for h in range(len(vertexSet)):
        for t in range(len(vertexSet)):
            if h == t:
                continue
            if t in merge[h]:
                continue
            merge[h][t] = []
            for n1 in vertexSet[h]:
                for n2 in vertexSet[t]:
                    cand_sents = [n1['sent_id'], n2['sent_id']]
                    
                    if keep_sent_order:
                        cand_sents.sort()
                    merge[h][t].append([cand_sents])

    # Default Path
    default = defaultdict(dict)
    for h in range(len(vertexSet)):
        for t in range(len(vertexSet)):
            if h == t:
                continue
            if t in default[h]:
                continue
            default[h][t] = []
            for n1 in vertexSet[h]:
                for n2 in vertexSet[t]:
                    cand_sents = [n1['sent_id'], n2['sent_id']]
                    if keep_sent_order:
                        cand_sents.sort()
                    default[h][t].append([cand_sents])
    
    consecutive = remove_redundency(consecutive)
    path_two = remove_redundency(path_two)
    path_three = remove_redundency(path_three)
    # merge = remove_redundency(merge)
    default = remove_redundency(default)

    if (path_type == "default"):
        return default
    elif (path_type == "path_two"):
        return path_two
    elif (path_type == "path_three"):
        return path_three
    elif (path_type == "consecutive"):
        return consecutive
    else:
        return default


# %%
def get_evidence_by_entity_pair(head_entity, tail_entity, entity_list, sents, path):
    
    
    envidence_string = ""

    for i in path.keys():
        for j in path[i]:
            if (i == j):
                continue
            if (entity_list[i] == head_entity and entity_list[j] == tail_entity):
                for x_index, x in enumerate(path[i][j]):
                    envidence_string += f"{x_index+1}. {sents[x]}\n"
    # envidence_string = envidence_string[:-1] if envidence_string != "" else envidence_string
    # if (envidence_string == ""):
    #     print(path)
    return envidence_string


tmp_string = """

sentences:
{}

==================================
orignal

{}
~~~~~~~~~~~~~~~~~~
rewrite

{}
=================================="""
# 計算不同 rewirte 與 original 的 path 是否有差異
def count_path_differ(context, sents, rewrite_sents, drug_mentions, symptom_mentions, entity_list):

    vertex_orig = get_vertex_set(entity_list, sents)
    vertex_rewrite = get_vertex_set(entity_list, rewrite_sents)

    orig_consecutive, orig_path_two, orig_path_three, orig_merge = extract_path(sents, vertex_orig, keep_sent_order=True)
    rewrite_consecutive, rewrite_path_two, rewrite_path_three, rewrite_merge = extract_path(rewrite_sents, vertex_rewrite, keep_sent_order=True)

    differ_count = 0
    total_count = 0

    all = {}
    consecutive = {}
    two = {}
    three = {}

    for i in range(len(drug_mentions)):
        for j in range(len(symptom_mentions)):
            head_mention = drug_mentions[i]
            tail_mention = symptom_mentions[j]
            evidence_orig = get_evidence_by_entity_pair(head_mention, tail_mention, entity_list, sents, orig_merge)
            evidence_rewrite = get_evidence_by_entity_pair(head_mention, tail_mention, entity_list, sents, rewrite_merge)

            consecutive_orig = get_evidence_by_entity_pair(head_mention, tail_mention, entity_list, sents, orig_consecutive)
            consecutive_rewrite = get_evidence_by_entity_pair(head_mention, tail_mention, entity_list, sents, rewrite_consecutive)

            path_two_orig = get_evidence_by_entity_pair(head_mention, tail_mention, entity_list, sents, orig_path_two)
            path_two_rewrite = get_evidence_by_entity_pair(head_mention, tail_mention, entity_list, sents, rewrite_path_two)

            path_three_orig = get_evidence_by_entity_pair(head_mention, tail_mention, entity_list, sents, orig_path_three)
            path_three_rewrite = get_evidence_by_entity_pair(head_mention, tail_mention, entity_list, sents, rewrite_path_three)
            
            total_count += 1

            try:
                if not (orig_merge[i][j] == rewrite_merge[i][j]):
                    
                    differ_count += 1
                    all[f'{head_mention}_{tail_mention}'] = tmp_string.format(
                        "\n".join(rewrite_sents), 
                        evidence_orig, evidence_rewrite)
            except:
                pass
        
            try:
                if not (orig_consecutive[i][j] == rewrite_consecutive[i][j]):
                    consecutive[f'{head_mention}_{tail_mention}'] = tmp_string.format(
                        "\n".join(rewrite_sents), 
                        consecutive_orig, consecutive_rewrite)
                    if (consecutive_orig == ""):
                        print(orig_consecutive[i][j])
                    if (consecutive_rewrite == ""):
                        print(rewrite_consecutive[i][j])
            except:
                pass
                
            try:
                if not (orig_path_two[i][j] == rewrite_path_two[i][j]):
                    two[f'{head_mention}_{tail_mention}'] = tmp_string.format(
                        "\n".join(rewrite_sents), 
                        path_two_orig, path_two_rewrite)
            except:
                pass
            
            try:
                if not (orig_path_three[i][j] == rewrite_path_three[i][j]):
                    three[f'{head_mention}_{tail_mention}'] = tmp_string.format(
                        "\n".join(rewrite_sents), 
                        path_three_orig, path_three_rewrite)
            except:
                pass
    
    return json.dumps(consecutive, indent=4), json.dumps(two, indent=4), json.dumps(three, indent=4), json.dumps(all, indent=4), differ_count, total_count
