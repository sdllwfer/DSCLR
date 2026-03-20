import requests
import typing

def deepseek_request_chat(api_key: str, 
                          model: str, 
                          messages: typing.List[dict], 
                          timeout: int = 30, # 提纯长文本建议稍微加长超时时间
                          *, 
                          max_tokens: int = 1024, 
                          stop: typing.Union[typing.List[str], None] = None, 
                          temperature: float = 1.0, 
                          top_p: float = 1.0, 
                          frequency_penalty: float = 0.0, 
                          presence_penalty: float = 0.0, 
                          n: int = 1,
                          response_format: dict = None, # 新增：用于强制 JSON 输出
                          **kargs
                          )-> dict:
    """
    访问deepseek平台的对话API的基础函数。
    """
    url = "https://api.deepseek.com/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "stop": stop,
        "temperature": temperature,
        "top_p": top_p,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "n": n
    }
    
    if response_format:
        payload["response_format"] = response_format

    response = requests.request("POST", url, json=payload, headers=headers, timeout=timeout)
    if response.status_code != 200:
        response.raise_for_status()

    return response.json()


def call_deepseek(api_key: str, system_prompt: str, user_prompt: str, is_json: bool = False, temperature: float = 0.1):
    """
    高度封装的业务调用接口。
    参数:
        api_key: 你的秘钥
        system_prompt: 系统提示词 (定义规则)
        user_prompt: 用户提示词 (传入数据)
        is_json: 是否强制要求模型输出 JSON 格式
        temperature: 温度 (默认 0.1 保证提纯稳定性)
    """
    model = "deepseek-chat"
    
    # 构建双层消息结构
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # DeepSeek 要求启用 JSON 模式时必须有此参数
    response_format = {"type": "json_object"} if is_json else None

    try:
        response = deepseek_request_chat(
            api_key=api_key, 
            model=model, 
            messages=messages, 
            temperature=temperature,
            response_format=response_format
        )
        return response["choices"][0]["message"]["content"]
        
    except requests.exceptions.Timeout:
        print("❌ 请求超时！")
    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP错误: {http_err}")
    except Exception as e:
        print(f"❌ 请求发生未知错误: {e}")
        
    return None