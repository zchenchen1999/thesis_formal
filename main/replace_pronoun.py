# %%
import json
from tqdm import tqdm
tqdm.pandas(desc="Processing rows")
import re

from typing import Dict, Any, List


from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

from RE_prompt import replace_pronoun_prompt_system, replace_pronoun_prompt_user

# 讀取環境變數
from dotenv import load_dotenv
import os
# 加載 .env 文件中的環境變數
load_dotenv('./enviroment.env')
# 讀取環境變數
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# %%

class CustomHandler(BaseCallbackHandler):
    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        
        formatted_prompts = "\n".join(prompts)
        # _log.info(f"\n\n==========================Prompt==========================\n{formatted_prompts}\n=========================================================\n\n")
        print((f"==========================Prompt==========================\n{formatted_prompts}\n=========================================================\n\n"))

# chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o", 
#     temperature=0.7, callbacks=[CustomHandler()])

chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model="gpt-4o", 
    temperature=0.7)

# %%
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

# 執行 LLM 替換代名詞
def llm_replace(sents, system_prompt, user_prompt):

    prompt = ChatPromptTemplate.from_messages(
        messages = [
            ("system", system_prompt),
            ("human", user_prompt),
        ]
    )
    
    chain = prompt | chat
    response = chain.invoke({"input_text": str(sents)})
    return response.content

def get_replace_pronoun(sents):
    try_num=0

    # 替換代名詞
    # parse 結果
    replace_result = relative_sentence_parse_code_string_to_ans(
        llm_replace(
            sents, 
            replace_pronoun_prompt_system, 
            replace_pronoun_prompt_user
        )
    )

    while (len(replace_result) != len(sents) and try_num<3):
        try_num += 1
        print("try again!")
        print(sents)
        replace_result = relative_sentence_parse_code_string_to_ans(
            llm_replace(
                sents, 
                replace_pronoun_prompt_system, 
                replace_pronoun_prompt_user
            )
        )
    
    return replace_result