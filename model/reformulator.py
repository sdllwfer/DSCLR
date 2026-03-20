"""
DSCLR Query Reformulator Module
调用 LLM API 对查询进行解耦，返回 Q_plus 和 Q_minus
支持缓存和复用机制
"""

import os
import json
import logging
import hashlib
from typing import Tuple, Optional, Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-e76614db92844a68bb36956b1e445c42")

SYSTEM_PROMPT = """You are a query decoupling expert. Your task is to decompose a given Query and Instruction into two parts:

1. Q_plus (Positive Semantic Features): Describes the core needs and intents that the document should satisfy
2. Q_minus (Negative Rejection Features): Describes features that the document should avoid (or "[NONE]" if none)

Output ONLY in the following JSON format, no other content:
```json
{
  "Reasoning_Steps": "Your reasoning process",
  "Q_plus": "The positive feature extracted",
  "Q_minus": "The negative feature extracted or [NONE]"
}
```

Important:
- If the Instruction has no explicit exclusion requirement, Q_minus should be "[NONE]"
- Q_plus should clearly describe what the document should contain
- Q_minus should explicitly describe what the document should avoid
- Output in English"""

USER_PROMPT_TEMPLATE = """Please analyze and decouple the following Query and Instruction:

Query: "{query}"
Instruction: "{instruction}"
"""


DEFAULT_MAX_RETRIES = 3
DEFAULT_INITIAL_DELAY = 1.0
DEFAULT_MAX_DELAY = 30.0
DEFAULT_BACKOFF_FACTOR = 2.0


class APIRetryError(Exception):
    """API 重试次数耗尽异常"""
    def __init__(self, message: str, attempts: int, last_error: str):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


def call_llm_api(
    query: str,
    instruction: str,
    api_key: str = API_KEY,
    max_retries: int = DEFAULT_MAX_RETRIES,
    initial_delay: float = DEFAULT_INITIAL_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    on_retry: Optional[callable] = None
) -> Tuple[str, str]:
    """
    调用 LLM API 进行查询解耦
    
    Args:
        query: 查询文本
        instruction: 指令文本
        api_key: API 密钥
        max_retries: 最大重试次数
        initial_delay: 初始重试延迟（秒）
        max_delay: 最大重试延迟（秒）
        backoff_factor: 退避因子
        on_retry: 重试回调函数 (attempt: int, error: str) -> None
        
    Returns:
        (Q_plus, Q_minus) 元组
        
    Raises:
        APIRetryError: 当重试次数耗尽时抛出
    """
    from utils.call_llm.call_deepseek import call_deepseek
    import time
    
    user_prompt = USER_PROMPT_TEMPLATE.format(
        query=query,
        instruction=instruction
    )
    
    last_error = "未知错误"
    delay = initial_delay
    
    for attempt in range(1, max_retries + 1):
        try:
            result_text = call_deepseek(
                api_key=api_key,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                is_json=True,
                temperature=0.1
            )
            
            if not result_text:
                last_error = "API 返回为空"
                logger.warning(f"尝试 {attempt}/{max_retries}: API 返回为空")
                
                if attempt < max_retries:
                    if on_retry:
                        on_retry(attempt, last_error)
                    time.sleep(delay)
                    delay = min(delay * backoff_factor, max_delay)
                    continue
                else:
                    raise APIRetryError(
                        f"API 重试 {max_retries} 次后仍失败",
                        attempts=max_retries,
                        last_error=last_error
                    )
            
            parsed_json = json.loads(result_text)
            q_plus = parsed_json.get("Q_plus", "")
            q_minus = parsed_json.get("Q_minus", "[NONE]")
            
            if attempt > 1:
                logger.info(f"API 调用成功 (重试 {attempt - 1} 次后)")
            
            return (q_plus, q_minus)
            
        except json.JSONDecodeError as e:
            last_error = f"JSON 解析失败: {e}"
            logger.warning(f"尝试 {attempt}/{max_retries}: {last_error}")
            
            if attempt < max_retries:
                if on_retry:
                    on_retry(attempt, last_error)
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                raise APIRetryError(
                    f"API 重试 {max_retries} 次后仍失败",
                    attempts=max_retries,
                    last_error=last_error
                )
                
        except Exception as e:
            last_error = f"API 调用失败: {e}"
            logger.warning(f"尝试 {attempt}/{max_retries}: {last_error}")
            
            if attempt < max_retries:
                if on_retry:
                    on_retry(attempt, last_error)
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_delay)
            else:
                raise APIRetryError(
                    f"API 重试 {max_retries} 次后仍失败",
                    attempts=max_retries,
                    last_error=last_error
                )
    
    raise APIRetryError(
        f"API 重试 {max_retries} 次后仍失败",
        attempts=max_retries,
        last_error=last_error
    )


class DualQueryCache:
    """
    双流查询缓存管理器
    支持多任务区分、版本控制、高效查询
    """
    
    def __init__(self, cache_dir: str = "dataset/FollowIR_test/dual_queries"):
        self.cache_dir = cache_dir
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """确保缓存目录存在"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _get_cache_file(self, task_name: str) -> str:
        """获取指定任务的缓存文件路径"""
        return os.path.join(self.cache_dir, f"dual_queries_{task_name}.jsonl")
    
    def _get_version_file(self, task_name: str) -> str:
        """获取版本信息文件路径"""
        return os.path.join(self.cache_dir, f"version_{task_name}.json")
    
    def load_cache(self, task_name: str) -> Dict[str, Dict]:
        """加载指定任务的缓存数据"""
        cache_file = self._get_cache_file(task_name)
        cache_data = {}
        
        if not os.path.exists(cache_file):
            return cache_data
        
        with open(cache_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    qid = item.get('qid')
                    query_type = item.get('query_type', 'og')
                    if qid:
                        # 使用与查找时一致的键格式: f"{qid}_{query_type}"
                        cache_key = f"{qid}_{query_type}"
                        cache_data[cache_key] = item
        
        logger.info(f"✅ 已加载 {len(cache_data)} 条缓存 (task: {task_name})")
        return cache_data
    
    def save_record(self, task_name: str, record: Dict):
        """保存单条记录"""
        cache_file = self._get_cache_file(task_name)
        
        with open(cache_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
    
    def save_batch(self, task_name: str, records: List[Dict]):
        """批量保存记录"""
        if not records:
            return
        
        cache_file = self._get_cache_file(task_name)
        
        with open(cache_file, 'a', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            f.flush()
        
        logger.info(f"💾 已保存 {len(records)} 条记录 (task: {task_name})")
    
    def get_record(self, task_name: str, qid: str) -> Optional[Dict]:
        """获取单条缓存记录"""
        cache = self.load_cache(task_name)
        return cache.get(qid)
    
    def get_all_records(self, task_name: str) -> Dict[str, Dict]:
        """获取所有缓存记录"""
        return self.load_cache(task_name)
    
    def clear_cache(self, task_name: str):
        """清空指定任务的缓存"""
        cache_file = self._get_cache_file(task_name)
        if os.path.exists(cache_file):
            os.remove(cache_file)
            logger.info(f"🗑️ 已清空缓存 (task: {task_name})")
    
    def get_version_info(self, task_name: str) -> Dict:
        """获取版本信息"""
        version_file = self._get_version_file(task_name)
        if os.path.exists(version_file):
            with open(version_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"version": "1.0", "created_at": None, "updated_at": None}
    
    def update_version(self, task_name: str, info: Dict):
        """更新版本信息"""
        version_file = self._get_version_file(task_name)
        info['updated_at'] = datetime.now().isoformat()
        
        with open(version_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)


class QueryReformulator:
    """
    查询重构器 - 封装 LLM API 调用
    支持缓存、失败记录和可配置的重试机制
    """
    
    def __init__(
        self,
        task_name: str = "Core17InstructionRetrieval",
        api_key: str = API_KEY,
        use_cache: bool = True,
        cache_dir: str = "dataset/FollowIR_test/dual_queries",
        max_retries: int = DEFAULT_MAX_RETRIES,
        initial_delay: float = DEFAULT_INITIAL_DELAY,
        max_delay: float = DEFAULT_MAX_DELAY,
        backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
        log_failed_dir: str = "dataset/FollowIR_test/failed_queries"
    ):
        """
        初始化查询重构器
        
        Args:
            task_name: 任务名称 (用于区分不同数据集)
            api_key: DeepSeek API 密钥
            use_cache: 是否使用缓存
            cache_dir: 缓存目录
            max_retries: 最大重试次数
            initial_delay: 初始重试延迟（秒）
            max_delay: 最大重试延迟（秒）
            backoff_factor: 退避因子
            log_failed_dir: 失败记录目录
        """
        self.task_name = task_name
        self.api_key = api_key
        self.use_cache = use_cache
        self.cache = DualQueryCache(cache_dir)
        self._cache_data: Dict[str, Dict] = {}
        
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        
        self.log_failed_dir = log_failed_dir
        self._failed_records: List[Dict] = []
        self._retry_callback_attempts: Dict[str, int] = {}
        
        self._init_failed_logger()
        
        if self.use_cache:
            self._cache_data = self.cache.get_all_records(task_name)
    
    def _init_failed_logger(self):
        """初始化失败记录日志"""
        os.makedirs(self.log_failed_dir, exist_ok=True)
        self._failed_log_file = os.path.join(
            self.log_failed_dir,
            f"failed_{self.task_name}.jsonl"
        )
    
    def _on_retry(self, qid_key: str, attempt: int, error: str):
        """重试回调函数"""
        self._retry_callback_attempts[qid_key] = attempt
    
    def _log_failed_query(
        self,
        qid: str,
        idx: int,
        query: str,
        instruction: str,
        query_type: str,
        attempts: int,
        error: str
    ):
        """记录失败查询详情"""
        failed_record = {
            "task_name": self.task_name,
            "qid": qid,
            "idx": idx,
            "query": query,
            "instruction": instruction,
            "query_type": query_type,
            "attempts": attempts,
            "error": error,
            "timestamp": datetime.now().isoformat()
        }
        
        self._failed_records.append(failed_record)
        
        with open(self._failed_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(failed_record, ensure_ascii=False) + '\n')
        
        logger.error(
            f"❌ 记录失败查询: {qid} ({query_type}), "
            f"尝试 {attempts} 次, 错误: {error}"
        )
    
    def get_failed_summary(self) -> Dict:
        """获取失败统计摘要"""
        return {
            "total_failed": len(self._failed_records),
            "task_name": self.task_name,
            "log_file": self._failed_log_file
        }
    
    def _create_record(
        self,
        qid: str,
        idx: int,
        query: str,
        instruction: str,
        query_type: str,
        q_plus: str,
        q_minus: str
    ) -> Dict:
        """创建缓存记录"""
        return {
            "task_name": self.task_name,
            "qid": qid,
            "idx": idx,
            "query": query,
            "instruction": instruction,
            "query_type": query_type,
            "q_plus": q_plus,
            "q_minus": q_minus,
            "created_at": datetime.now().isoformat()
        }
    
    def reformulate(
        self,
        qid: str,
        idx: int,
        query: str,
        instruction: str,
        query_type: str = "og"
    ) -> Tuple[str, str]:
        """
        对查询进行解耦重构
        
        Args:
            qid: 查询ID
            idx: 查询索引 (用于关联原数据集)
            query: 查询文本
            instruction: 指令文本
            query_type: 查询类型 (og/changed)
            
        Returns:
            (Q_plus, Q_minus) 元组
        """
        cache_key = f"{qid}_{query_type}"
        
        if self.use_cache and cache_key in self._cache_data:
            record = self._cache_data[cache_key]
            logger.debug(f"📥 使用缓存: {qid} ({query_type})")
            return (record['q_plus'], record['q_minus'])
        
        retry_callback = lambda attempt, error: self._on_retry(cache_key, attempt, error)
        
        try:
            q_plus, q_minus = call_llm_api(
                query, instruction, self.api_key,
                max_retries=self.max_retries,
                initial_delay=self.initial_delay,
                max_delay=self.max_delay,
                backoff_factor=self.backoff_factor,
                on_retry=retry_callback
            )
        except APIRetryError as e:
            logger.warning(f"API 重试耗尽，记录失败查询: {qid} ({query_type})")
            self._log_failed_query(
                qid=qid,
                idx=idx,
                query=query,
                instruction=instruction,
                query_type=query_type,
                attempts=e.attempts,
                error=e.last_error
            )
            return (query, "[NONE]")
        
        if self.use_cache:
            record = self._create_record(qid, idx, query, instruction, query_type, q_plus, q_minus)
            self._cache_data[cache_key] = record
            self.cache.save_record(self.task_name, record)
        
        return (q_plus, q_minus)
    
    def reformulate_batch(
        self,
        queries: List[Tuple[str, int, str, str, str]]
    ) -> Dict[str, Tuple[str, str]]:
        """
        批量解耦查询
        
        Args:
            queries: [(qid, idx, query, instruction, query_type), ...]
            
        Returns:
            {f"{qid}_{query_type}": (Q_plus, Q_minus), ...}
        """
        results = {}
        to_api_call = []
        
        for qid, idx, query, instruction, query_type in queries:
            cache_key = f"{qid}_{query_type}"
            
            if self.use_cache and cache_key in self._cache_data:
                record = self._cache_data[cache_key]
                results[cache_key] = (record['q_plus'], record['q_minus'])
            else:
                to_api_call.append((qid, idx, query, instruction, query_type))
        
        if to_api_call:
            logger.info(f"🔄 需要调用 API: {len(to_api_call)} 条")
            
            new_records = []
            for qid, idx, query, instruction, query_type in to_api_call:
                retry_callback = lambda attempt, error: self._on_retry(f"{qid}_{query_type}", attempt, error)
                
                try:
                    q_plus, q_minus = call_llm_api(
                        query, instruction, self.api_key,
                        max_retries=self.max_retries,
                        initial_delay=self.initial_delay,
                        max_delay=self.max_delay,
                        backoff_factor=self.backoff_factor,
                        on_retry=retry_callback
                    )
                except APIRetryError as e:
                    logger.warning(f"API 重试耗尽，记录失败查询: {qid} ({query_type})")
                    self._log_failed_query(
                        qid=qid,
                        idx=idx,
                        query=query,
                        instruction=instruction,
                        query_type=query_type,
                        attempts=e.attempts,
                        error=e.last_error
                    )
                    q_plus, q_minus = query, "[NONE]"
                
                cache_key = f"{qid}_{query_type}"
                results[cache_key] = (q_plus, q_minus)
                
                if self.use_cache:
                    record = self._create_record(qid, idx, query, instruction, query_type, q_plus, q_minus)
                    new_records.append(record)
            
            if self.use_cache and new_records:
                self.cache.save_batch(self.task_name, new_records)
                for record in new_records:
                    cache_key = f"{record['qid']}_{record['query_type']}"
                    self._cache_data[cache_key] = record
        
        return results
    
    def get_cached(self, qid: str, query_type: str = "og") -> Optional[Tuple[str, str]]:
        """获取缓存的查询结果"""
        cache_key = f"{qid}_{query_type}"
        if cache_key in self._cache_data:
            record = self._cache_data[cache_key]
            return (record['q_plus'], record['q_minus'])
        return None
    
    def has_cache(self, qid: str, query_type: str = "og") -> bool:
        """检查是否有缓存"""
        cache_key = f"{qid}_{query_type}"
        return cache_key in self._cache_data
    
    def get_cache_stats(self) -> Dict[str, int]:
        """获取缓存统计信息"""
        return {
            "total": len(self._cache_data),
            "task": self.task_name
        }
    
    def clear_cache(self):
        """清空当前任务的缓存"""
        self.cache.clear_cache(self.task_name)
        self._cache_data = {}


def get_reformulator(
    task_name: str = "Core17InstructionRetrieval",
    api_key: str = API_KEY,
    use_cache: bool = True
) -> QueryReformulator:
    """
    获取查询重构器的便捷函数
    """
    return QueryReformulator(
        task_name=task_name,
        api_key=api_key,
        use_cache=use_cache
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    reformulator = get_reformulator(task_name="Core17InstructionRetrieval", use_cache=True)
    
    test_qid = "1-og"
    test_idx = 1
    test_query = "What are the causes of railway accidents throughout history?"
    test_instruction = "A relevant document should discuss the various factors contributing to railway accidents."
    
    print("测试 reformulate:")
    q_plus, q_minus = reformulator.reformulate(test_qid, test_idx, test_query, test_instruction, "og")
    print(f"  Q_plus: {q_plus}")
    print(f"  Q_minus: {q_minus}")
    
    print(f"\n缓存统计: {reformulator.get_cache_stats()}")
