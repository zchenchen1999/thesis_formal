# %%
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
import re
# %%

begin_of_text = "<|begin_of_text|>"
system_header = "<|start_header_id|>system<|end_header_id|>"
user_header = "<|start_header_id|>user<|end_header_id|>"
assistant_header = "<|start_header_id|>assistant<|end_header_id|>"
eot_id = "<|eot_id|>"


# Callbacks support token-wise streaming
# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 42  # The number of layers to put on the GPU. The rest will be on the CPU. If you don't know how many layers there are, you can use -1 to move all to GPU.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

# n_gpu_layers = 25
# n_batch = 32
# n_ctx=131072

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    # Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
    # Meta-Llama-3.1-8B-Instruct-Doctor.Q4_K_M.gguf
    model_path="/home/zchenchen1999/thesis/models/Meta-Llama-3.1-8B-Instruct-Doctor.Q4_K_M.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    temperature=0.7,
    # top_k=1,
    # max_tokens=1024, # 1024, 1
    n_ctx=30000,
    # callback_manager=callback_manager,
    # verbose=True,  # Verbose is required to pass to the callback manager
)

# 計算 Token 數量的自訂方法
def calculate_tokens(text):
    # 獲取底層 llama_cpp 模型
    tokens = llm.client.tokenize(text.encode("utf-8"), add_bos=True)  # 將字串轉換為 bytes
    return len(tokens)

def parse_answer(text, choice):
    print(f"LLM: {text}")
    if (text.strip() in choice):
        print(f"The extracted relation is: {text.strip()}")
        return text.strip()
    
    # 有匹配句號
    # match = re.search(r'is ([A-D])\.', text)
    match1 = re.search(r'is \*{0,2}([A-D])\*{0,2}\.', text)

    if match1:
        relation = match1.group(1)
        print(f"The extracted relation is: {relation}")
        return relation
    else:
        pass
        # print("No match found.")
        # return None
    
    # 沒匹配句號
    match2 = re.search(r'is \*{0,2}([A-D])\*{0,2}', text)

    if match2:
        relation = match2.group(1)
        print(f"The extracted relation is: {relation}")
        return relation
    else:
        pass
        # print("No match found.")
        # return None
    
    match3 = re.search(r'\*{0,2}Relation\*{0,2}:\*{0,2} ([A-Z])', text)
    if match3:
        relation = match3.group(1)
        print(f"The extracted relation is: {relation}")
        return relation
    else:
        pass
        # print("No match found.")
        # return None

    match4 = re.search(r'\*{0,2}Relation Category\*{0,2}:\*{0,2} ([A-Z])', text)
    if match4:
        relation = match4.group(1)
        print(f"The extracted relation is: {relation}")
        return relation
    else:
        pass
        # print("No match found.")
        # return None

    match5 = re.search(r'is category \*{0,2}([A-D])\*{0,2}\.', text)
    if match5:
        relation = match5.group(1)
        print(f"The extracted relation is: {relation}")
        return relation
    else:
        pass
        # print("No match found.")
        # return None
    
    match6 = re.search(r'is: \*{0,2}([A-D])\*{0,2}', text)
    if match6:
        relation = match6.group(1)
        print(f"The extracted relation is: {relation}")
        return relation
    else:
        pass
        # print("No match found.")
        # return None

def get_total_token_num(system_prompt, evidence_user_prompt, var_dict, print_prompt=False):
    template = begin_of_text + "\n" + system_header + "\n\n" + system_prompt + eot_id + "\n" + user_header + "\n\n" + evidence_user_prompt + eot_id + "\n" + assistant_header + "\n\n"
    prompt = PromptTemplate.from_template(template)
    if (print_prompt == True):
        print(prompt.format(**var_dict))
    prompt_token_num = calculate_tokens(prompt.format(**var_dict))
    return prompt_token_num

def get_open_llm_result(system_prompt, evidence_user_prompt, var_dict, data="webmd", parse=False):
    if (data == "webmd"):
        choice = ["A", "B", "C", "D"]
    else:
        choice = ["A", "B"]

    # template = begin_of_text + "\n" + system_header + "\n\n" + system_prompt + eot_id + "\n" + user_header + "\n\n" + evidence_user_prompt + eot_id + "\n" + assistant_header + "\n\n"
    template =  "\n" + system_header + "\n\n" + system_prompt + eot_id + "\n" + user_header + "\n\n" + evidence_user_prompt + eot_id + "\n" + assistant_header + "\n\n"

    prompt = PromptTemplate.from_template(template)
    # print(prompt.format(**var_dict))
    # prompt_token_num = calculate_tokens(prompt.format(**var_dict))
    # print(f'總 Token 數量：{prompt_token_num}')

    llm_chain = prompt | llm

    ans = ""
    count = 0

    # parse: True, False, None
    if (parse == True):
        while ans not in choice:
            if (count != 0):
                print(f"執行第{count}次")
            ans = parse_answer(llm_chain.invoke(var_dict), choice)
            count += 1
            if (count == 30):
                return "no_answer"
    elif (parse == False):
        while ans not in choice:
            if (count != 0):
                print(f"執行第{count}次")
            ans = llm_chain.invoke(var_dict)
            print(ans)
            count += 1
            if (count == 100):
                return "no_answer"
    else:
        ans = llm_chain.invoke(var_dict)

    return ans