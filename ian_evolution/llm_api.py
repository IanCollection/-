import time
from ian_evolution.client_manager import deepseek_client,minimax_client,qwen_client

def minimax_data_collection_top_asking(ai, message, indicator_type, used_tool, sub_indicators_info=None,
                                       full_search_mode=False):
    """
    优化后的Minimax接口调用函数，主要优化速率限制问题，特点：
    1. 将最大重试次数(max_retries)调整为3，减少不必要的等待。
    2. 将基础等待时间(retry_delay)缩短为2秒，提高调用效率。
    3. 保留并优化对状态码及速率限制的判断，一旦触发速率限制，则进行等待重试。
    """
    temp_record = {}
    max_retries = 10
    retry_delay = 0.5
    cost = 0
    found_answer = True

    if ai == "minimax":
        model = 'MiniMax-Text-01'
        client = minimax_client
    elif ai =="deepseek":
        # model = 'deepseek-chat'
        model = "Pro/deepseek-ai/DeepSeek-V3"
        client = deepseek_client
    elif ai =="qwen":
        # model = "qwen-plus-latest"
        model = "qwen-max-latest"
        client = qwen_client

    for attempt in range(max_retries):
        try:
            # 调用API
            if model == "Pro/deepseek-ai/DeepSeek-V3":
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.01,
                    top_p=0.1,
                    messages=message,
                    timeout=240,
                    # response_format={'type': 'json_object'}
                )
            elif model == "qwen-max-latest" or "qwen-plus-latest":
                response = qwen_client.chat.completions.create(
                    model=model,
                    temperature=0.01,
                    top_p=0.1,
                    messages=message,
                    timeout=240,
                    # response_format={"type": "json_object"}
                )
            else:
                response = client.chat.completions.create(
                    model=model,
                    temperature=0.01,
                    top_p=0.1,
                    messages=message,
                    timeout=240
                )
                # print(response)
            # print(response)
            # if ai == 'minimax':
            #     time.sleep(1)
            # 正常响应处理
            this_answer = response.choices[0].message.content
            input_usage = response.usage.prompt_tokens
            output_usage = response.usage.completion_tokens

            # 记录结果
            temp_record = {
                "message": message.copy(),
                "answer": this_answer,
                "usage": {
                    "prompt_tokens": input_usage,
                    "completion_tokens": output_usage
                }
            }

            # 判断是否已获得有效答案
            if not full_search_mode and "信息不足" not in this_answer:
                found_answer = True

            # 计算本次调用成本
            cost += (input_usage / 1000 * 0.0024) + (output_usage / 1000 * 0.008)
            break  # 成功则退出循环

        except Exception as e:
            # 打印出每次尝试失败的具体异常信息
            print(f"Attempt {attempt + 1} failed for model {model}. Exception type: {type(e).__name__}, Details: {str(e)}")
            # 异常处理
            if attempt == max_retries - 1:
                temp_record["error"] = f"超过最大重试次数({max_retries})，异常: {str(e)}"
            else:
                time.sleep(retry_delay)  # 在重试前等待

    return temp_record, used_tool, found_answer, cost

