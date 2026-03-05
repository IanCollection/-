import json
from dotenv import load_dotenv
from ian_evolution.client_manager import qwen_client
load_dotenv()

# 黑名单（直接删除）需要扩充,之后考虑单独维护一个数据库/文件
blacklists = ['单位营收', '单位产值', '是否', '披露']
# 专有名词（不需要拆分)需要扩充,之后考虑单独维护一个数据库/文件
specialwords = ['范畴一温室气体', '范畴二温室气体', '温室气体', '生物多样性']

JUDGE_CATEGORY_PROMPT_TEMPLATE = """
You are a precise JSON-generating word classifier.
Your task is to determine the category of the following Chinese word or phrase:
INPUT_WORD_OR_PHRASE_HERE

The output MUST be a SINGLE JSON object with two keys: "类别序号" (category serial number) and "词语" (the input word/phrase).
The value for "类别序号" must be a string: "1", "2", or "3".

**Categories:**
1.  **名词性量词短语 (Category "1"):** Example: "汽油消耗量", "总用水量".
2.  **动词或动词短语 (Category "2"):** Example: "节能措施", "无害废物管理".
3.  **其他 (Category "3"):** Words/phrases not fitting above. Example: "环境管理体系认证".

**Output Format Example:**
For the input "总用水量", the exact JSON output should be:
{"类别序号": "1", "词语": "总用水量"}

Ensure your response is ONLY this single JSON object, with no extra text, explanations, or list wrappers.
If the input word/phrase contains multiple distinct parts (e.g., separated by '/'), classify ONLY THE FIRST part and use that first part as the value for the "词语" key in your JSON output.
"""

SPLIT_WORDS_PROMPT_TEMPLATE = f"""
You are a precise JSON-generating keyword extractor and scorer.
Your task is to split the following input Chinese word/phrase into keywords and assign an importance score (1 or 2) to each keyword:
INPUT_WORD_OR_PHRASE_HERE

The output MUST be a SINGLE JSON object. The keys of this JSON object are the extracted keywords (strings), and the values are their corresponding scores (integers 1 or 2).

**Rules for Splitting and Scoring:**
0.  **Priority to Special Nouns:** Special nouns list: {specialwords}. Identify and extract these first, assigning them a score of 2. Process the remaining parts according to subsequent rules.
1.  **Initial Split:** Split based on word independence.
2.  **Core Concept Identification:** Differentiate core concepts (e.g., "电", "煤", "温室气体") from non-core concepts (e.g., "管理", "体系", "排放量").
3.  **Core Concept Refinement:** Further split core concepts to their minimal entities. If a core word has multiple modifiers, extract them together.

**Scoring:**
*   Core/Important Components: Score 2.
*   Non-core/Auxiliary Components: Score 1.
*   There must be at least one keyword with a score of 2.

**Output Format Examples:**
Input: "企业节用电量"
Expected JSON Output: {{"企业": 1, "节用电量": 2, "电": 2}}

Input: "天然气消耗量" (assuming "天然气" is in the special nouns list)
Expected JSON Output: {{"天然气": 2, "消耗量": 1}}

Input: "节煤措施"
Expected JSON Output: {{"节煤": 2, "措施": 1}}

Ensure your response is ONLY the single JSON object mapping keywords to scores. Do not include the input phrase in the output, and do not wrap the JSON object in a list or any other structure.
"""

CHECK_SINGLE_PROMPT_TEMPLATE = """
You are a precise JSON-generating single-character concept checker.
Your task is to determine if EACH Chinese character in the following input list represents a "concrete physical entity":
INPUT_CHARACTER_LIST_HERE

The output MUST be a SINGLE JSON object. The keys of this JSON object are the input characters (strings from the list), and the values are their corresponding judgment results (integers 0 or 1).

**"Concrete Physical Entity" (判断结果 1):** Refers to specific substances or entities, e.g., '水', '电', '煤', '气', '油'.
**Not a "Concrete Physical Entity" (判断结果 0):** Quantifiers, descriptors, or suffixes, e.g., '量', '向', '比', '率', '法'.

**Output Format Example:**
Input: ['水', '量', '电']
Expected JSON Output: {{"水": 1, "量": 0, "电": 1}}

Ensure your response is ONLY this single JSON object. Do not wrap it in a list or any other structure.
"""



def remove_elements(text, elements):
    for element in elements:
        text = text.replace(element, "")
    return text


def get_nested_dict_value(data):
    if isinstance(data, dict):
        # If the dict has a single key and its value is also a dict,
        # assume it's an unnecessary wrapper and return the inner dict.
        if len(data) == 1:
            first_value = next(iter(data.values()))
            if isinstance(first_value, dict):
                print(f"Info: Unwrapping a single-key dictionary. Using inner dict: {first_value}")
                return first_value # Return the inner dictionary
        return data # Otherwise, return the dictionary as is
    # If data is a list containing a single dictionary, extract and process that dictionary.
    elif isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
        print(f"Info: LLM returned a list with one dict. Processing the dict: {data[0]}")
        # Recursively call to handle potential nesting within this extracted dict
        return get_nested_dict_value(data[0])
    else:
        # If it's a list with multiple items, or not a dict/list, the format is unexpected.
        print(f"Warning: Unexpected data structure for get_nested_dict_value. Type: {type(data)}, Data (first 100 chars): {str(data)[:100]}")
        return None


def get_gpt_requests(prompt):
    try_times = 0
    total_retry_times = 2
    while try_times < total_retry_times:
        try:
            # Outer try for the API call itself and subsequent processing
            try:
                response = qwen_client.chat.completions.create(
                    model="qwen-max",
                    # model="deepseek-v3",
                    response_format={"type": "json_object"},
                    messages=[{"role": "user", "content": prompt}],
                    timeout=120  # 设置超时（例如3秒）
                )
                response_content = response.choices[0].message.content
            except Exception as api_call_exception: # Catch exceptions from the API call directly
                print(f"Error during API call - {api_call_exception}")
                # Optional: add a short sleep before retrying API calls
                # import time
                # time.sleep(1) 
                try_times +=1
                if try_times >= total_retry_times:
                    return None # Exhausted retries for API call
                continue # Retry the API call by restarting the while loop

            # Inner try for JSON parsing and processing via get_nested_dict_value
            try:
                parsed_json = json.loads(response_content)
                return get_nested_dict_value(parsed_json) # Process with the new function
            except json.JSONDecodeError as json_e:
                print(f"JSONDecodeError: {json_e} for response: {response_content}")
                # This JSON error is likely not recoverable by simple retry with the same prompt
                return None # Directly return None for JSON decode errors
            except Exception as processing_e: # Catch other exceptions (e.g., from get_nested_dict_value if it raises one)
                print(f"Error processing LLM response: {processing_e}. Response was: {response_content}")
                return None # Return None for other processing errors

        except Exception as outer_e: # Catch-all for any other unexpected error in the loop
            print(f"Outer loop error - {outer_e}")
            try_times += 1 # Still increment retry for outer loop errors
            # If this point is reached, it's likely an issue not covered by specific API or JSON errors
            # but we still respect the retry mechanism.
    return None # Return None if all retries are exhausted



def get_category(indicator):
    prompt = JUDGE_CATEGORY_PROMPT_TEMPLATE.replace("INPUT_WORD_OR_PHRASE_HERE", indicator)
    return get_gpt_requests(prompt=prompt)


def get_words(indicator): # Removed 'category' parameter
    prompt = SPLIT_WORDS_PROMPT_TEMPLATE.replace("INPUT_WORD_OR_PHRASE_HERE", indicator)
    return get_gpt_requests(prompt=prompt)


def del_single_word(key_words_dict): # Changed parameter name for clarity
    char_list_str = str(list(key_words_dict.keys()))
    prompt = CHECK_SINGLE_PROMPT_TEMPLATE.replace("INPUT_CHARACTER_LIST_HERE", char_list_str)
    return get_gpt_requests(prompt=prompt)



def get_key_words_no_thread(indicator):
    category = get_category(indicator)
    if category and isinstance(category, dict): # Check if category is a dict
        # Correctly extract the category value (e.g., "1", "2", "3")
        # category.values() could be ['1', 'some word'], so list(category.values())[0] is '1'
        category_value = list(category.values())[0] if category.values() else None
        if category_value is None:
            print(f"Warning: Could not determine category value for indicator: {indicator}")
            return None

        words = get_words(indicator) # Call updated get_words
        if words is None: # Check if words is None
            print(f"Warning: get_words returned None for indicator: {indicator}")
            return None
        
        # Ensure words is a dictionary before calling .items()
        if not isinstance(words, dict):
            print(f"Warning: get_words did not return a dictionary for indicator: {indicator}. Got: {type(words)}")
            return None

        single_key_words = {key: value for key, value in words.items() if len(key) == 1}
        if len(single_key_words):
            del_sign = del_single_word(single_key_words)
            if del_sign and isinstance(del_sign, dict): # Check if del_sign is a dict
                for key in del_sign.keys():
                    # Ensure key exists in words before attempting to delete
                    if key in words and (del_sign[key] == '0' or del_sign[key] == 0):
                        del words[key]
            elif del_sign is None:
                 print(f"Warning: del_single_word returned None for single_key_words: {single_key_words}")
            elif not isinstance(del_sign, dict):
                 print(f"Warning: del_single_word did not return a dictionary for single_key_words: {single_key_words}. Got: {type(del_sign)}")
        return words
    elif category is None:
        print(f"Warning: get_category returned None for indicator: {indicator}")
    else:
        print(f"Warning: get_category did not return a dictionary for indicator: {indicator}. Got: {type(category)}")
    return None


def get_key_words_thread(executor, indicator, idx):
    category = get_category(indicator)

    if category and isinstance(category, dict): # Check if category is a dict
        # Correctly extract the category value (e.g., "1", "2", "3")
        category_value = list(category.values())[0] if category.values() else None
        if category_value is None:
            print(f"Warning (thread): Could not determine category value for indicator: {indicator}")
            return idx, [None, None]

        words = get_words(indicator) # Call updated get_words
        if words is None: # Check if words is None
            print(f"Warning (thread): get_words returned None for indicator: {indicator}")
            return idx, [None, None]

        # Ensure words is a dictionary before calling .items()
        if not isinstance(words, dict):
            print(f"Warning (thread): get_words did not return a dictionary for indicator: {indicator}. Got: {type(words)}")
            return idx, [None, None]
            
        single_key_words = {key: value for key, value in words.items() if len(key) == 1}

        if len(single_key_words):
            del_sign = del_single_word(single_key_words)
            if del_sign and isinstance(del_sign, dict): # Check if del_sign is a dict
                for key in del_sign.keys():
                    # Ensure key exists in words before attempting to delete
                    if key in words and (del_sign[key] == '0' or del_sign[key] == 0):
                        del words[key]
            elif del_sign is None:
                print(f"Warning (thread): del_single_word returned None for single_key_words: {single_key_words}")
            elif not isinstance(del_sign, dict):
                print(f"Warning (thread): del_single_word did not return a dictionary for single_key_words: {single_key_words}. Got: {type(del_sign)}")
        return idx, [words, category_value] # Return category_value (from get_category)

    elif category is None:
        print(f"Warning (thread): get_category returned None for indicator: {indicator}")
    else:
        print(f"Warning (thread): get_category did not return a dictionary for indicator: {indicator}. Got: {type(category)}")
    return idx, [None, None]


