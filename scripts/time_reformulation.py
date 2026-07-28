"""
Timing script for query decomposition with Qwen3-4B

Measures per-query decomposition time and reports mean/std statistics.
"""

import os
import sys
import json
import time
import logging
import argparse
import statistics
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# TSC_BALANCED prompt (best performing)
SYSTEM_PROMPT = """Your task: Decide whether the instruction contains any exclusion information, then extract Q_plus and Q_minus.

STEP 1 — Binary decision: Does the instruction explicitly state that something is NOT relevant, irrelevant, or outside scope?

YES signals (exclusion EXISTS):
  - "X is not relevant" / "X is irrelevant" / "X are not relevant"
  - "outside of [scope]" / "not directly attributable to"
  - "Discussions of X are not relevant"
  - "only in [region]" (implies other regions excluded)

NO signals (NO exclusion — these are all RELEVANT content descriptions):
  - "A relevant document will include/provide/contain X" → X is what to FIND
  - "Particularly sought are X" → X is what to FIND
  - "Relevant documents may identify X" → X is RELEVANT
  - "must include X" / "must quote X" / "must discuss X" → X is a REQUIREMENT, not exclusion
  - "Documents must include X" → X is REQUIRED content
  - The instruction provides background context or examples → ALL context is RELEVANT
  - The instruction lists subtopics or details → ALL are RELEVANT
  - "X are also relevant" / "X is relevant as well" → X is RELEVANT

CRITICAL DISTINCTION:
  - "X is not relevant" → X goes in Q_minus (excluded)
  - "X is relevant" / "X must be included" → X goes in Q_plus (required)
  - Instruction describes background → background is RELEVANT context
  - Instruction says "must include X" → X is a requirement, NOT an exclusion

If NO → Q_minus = [NONE]. Put ALL instruction content into Q_plus.
If YES → Extract ONLY the explicitly excluded topics into Q_minus as short keywords.

Output JSON: {"Q_plus": "...", "Q_minus": "[NONE]"} or {"Q_plus": "...", "Q_minus": "keyword1, keyword2"}

FORMAT RULES:
- Q_minus must be exactly [NONE] (with brackets) when no exclusion exists
- Never write "none" or "NONE" without brackets
- Q_minus uses only short keywords (2-5 words per item)
- Q_minus must NOT contain anything that is also in Q_plus"""

USER_TEMPLATE = """Query: {query}
Instruction: {instruction}

Analyze and output JSON:"""


def load_followir_queries(task_name: str):
    """Load FollowIR queries from HuggingFace datasets."""
    import datasets
    
    path_map = {
        "Core17InstructionRetrieval": "jhu-clsp/core17-instructions-mteb",
        "Robust04InstructionRetrieval": "jhu-clsp/robust04-instructions-mteb",
        "News21InstructionRetrieval": "jhu-clsp/news21-instructions-mteb",
    }
    
    dataset_path = path_map.get(task_name, "")
    if not dataset_path:
        raise ValueError(f"Unknown task: {task_name}")
    
    logger.info(f"Loading dataset: {dataset_path}")
    ds_q = datasets.load_dataset(dataset_path, 'queries')
    ds_inst = datasets.load_dataset(dataset_path, 'instruction')
    
    q_split = 'queries' if 'queries' in ds_q else list(ds_q.keys())[0]
    i_split = 'instruction' if 'instruction' in ds_inst else list(ds_inst.keys())[0]
    
    instruction_dict = {}
    for item in ds_inst[i_split]:
        qid = str(item.get('query-id', ''))
        instruction_dict[qid] = str(item.get('instruction', ''))
    
    q_og, q_changed = {}, {}
    for q in ds_q[q_split]:
        full_qid = str(q.get('_id', q.get('id', '')))
        query_text = q.get('text', '')
        inst = instruction_dict.get(full_qid, "")
        if full_qid.endswith('-og'):
            q_og[full_qid] = (query_text, inst)
        elif full_qid.endswith('-changed'):
            q_changed[full_qid] = (query_text, inst)
    
    logger.info(f"Loaded {len(q_og)} OG queries, {len(q_changed)} changed queries")
    return q_og, q_changed


class TimerReformulator:
    """Qwen3-4B reformulator with timing capability."""
    
    def __init__(self, model_path: str, device: str = "cuda", max_new_tokens: int = 512):
        logger.info(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.max_new_tokens = max_new_tokens
        
        if device == "cuda":
            try:
                torch.cuda._lazy_init()
                if not torch.cuda.is_available():
                    device = "cpu"
            except Exception:
                device = "cpu"
        
        if device == "cpu":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float32, trust_remote_code=True
            ).to("cpu")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
            )
        self.model.eval()
        logger.info(f"Model loaded on {device}")
    
    def reformulate(self, query: str, instruction: str, temperature: float = 0.1) -> Tuple[str, str, float]:
        """Reformulate query and return (q_plus, q_minus, elapsed_time)."""
        user_prompt = USER_TEMPLATE.format(query=query, instruction=instruction)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        
        start_time = time.perf_counter()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
            )
        elapsed = time.perf_counter() - start_time
        
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        result_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        
        q_plus, q_minus = self._parse_result(result_text, query)
        return q_plus, q_minus, elapsed
    
    def _parse_result(self, result_text: str, original_query: str) -> Tuple[str, str]:
        """Parse model output to extract Q_plus and Q_minus."""
        try:
            json_start = result_text.find('{')
            json_end = result_text.rfind('}') + 1
            if json_start != -1 and json_end > json_start:
                result = json.loads(result_text[json_start:json_end])
                q_plus = result.get('Q_plus', result.get('q_plus', '')).strip()
                q_minus = result.get('Q_minus', result.get('q_minus', '')).strip()
                if q_plus:
                    if q_minus.lower() in ['none', '[none]']:
                        q_minus = '[NONE]'
                    return q_plus, (q_minus if q_minus else '[NONE]')
        except json.JSONDecodeError:
            pass
        
        # Fallback parsing
        q_plus, q_minus = original_query, '[NONE]'
        for line in result_text.split('\n'):
            line = line.strip()
            if 'Q_plus' in line or 'q_plus' in line:
                parts = line.split(':', 1)
                if len(parts) > 1 and parts[1].strip().strip('",'):
                    q_plus = parts[1].strip().strip('",')
            elif 'Q_minus' in line or 'q_minus' in line:
                parts = line.split(':', 1)
                if len(parts) > 1 and parts[1].strip().strip('",'):
                    q_minus = parts[1].strip().strip('",')
                    if q_minus.lower() in ['none', '[none]']:
                        q_minus = '[NONE]'
        return q_plus, q_minus


def main():
    parser = argparse.ArgumentParser(description="Time query decomposition with Qwen3-4B")
    parser.add_argument("--task_name", type=str, default="Core17InstructionRetrieval",
                        choices=["Core17InstructionRetrieval", "Robust04InstructionRetrieval", "News21InstructionRetrieval"])
    parser.add_argument("--model_path", type=str, default="/home/luwa/Documents/models/Qwen3-4B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_file", type=str, default="logs/reformulation_timing.json")
    args = parser.parse_args()
    
    # Load queries
    q_og, q_changed = load_followir_queries(args.task_name)
    
    # Combine all queries
    all_queries = []
    for qid, (query_text, instruction) in q_og.items():
        all_queries.append((qid, query_text, instruction, "og"))
    for qid, (query_text, instruction) in q_changed.items():
        all_queries.append((qid, query_text, instruction, "changed"))
    
    logger.info(f"Total queries to process: {len(all_queries)}")
    
    # Initialize model
    reformulator = TimerReformulator(args.model_path, args.device)
    
    # Warmup (first few queries may be slower due to CUDA initialization)
    logger.info("Running warmup (3 queries)...")
    warmup_queries = all_queries[:3]
    for qid, query, instruction, qtype in warmup_queries:
        reformulator.reformulate(query, instruction)
    logger.info("Warmup complete")
    
    # Time all queries
    timings: List[float] = []
    results = []
    
    logger.info("Starting timing experiment...")
    for i, (qid, query, instruction, qtype) in enumerate(all_queries):
        q_plus, q_minus, elapsed = reformulator.reformulate(query, instruction)
        timings.append(elapsed)
        results.append({
            "qid": qid,
            "query_type": qtype,
            "elapsed_seconds": elapsed,
        })
        
        if (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(all_queries)} queries")
    
    # Compute statistics
    mean_time = statistics.mean(timings)
    std_time = statistics.stdev(timings)
    min_time = min(timings)
    max_time = max(timings)
    total_time = sum(timings)
    
    stats = {
        "task_name": args.task_name,
        "model_path": args.model_path,
        "num_queries": len(timings),
        "mean_seconds": mean_time,
        "std_seconds": std_time,
        "min_seconds": min_time,
        "max_seconds": max_time,
        "total_seconds": total_time,
        "queries_per_second": len(timings) / total_time,
    }
    
    # Print results
    print("\n" + "="*60)
    print("QUERY DECOMPOSITION TIMING RESULTS")
    print("="*60)
    print(f"Task: {args.task_name}")
    print(f"Model: {args.model_path}")
    print(f"Number of queries: {len(timings)}")
    print("-"*60)
    print(f"Mean time per query: {mean_time:.4f} seconds")
    print(f"Std deviation: {std_time:.4f} seconds")
    print(f"Min time: {min_time:.4f} seconds")
    print(f"Max time: {max_time:.4f} seconds")
    print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Throughput: {len(timings) / total_time:.2f} queries/second")
    print("="*60)
    
    # Save results
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    output = {
        "statistics": stats,
        "per_query_timings": results,
    }
    with open(args.output_file, 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()