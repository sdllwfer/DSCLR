"""
FollowIR 评测脚本 - 单向量稠密检索模型版本
评估 BAAI/bge-large-en-v1.5 模型在 FollowIR 基准上的性能
"""

import os
import sys
import json
import time
import argparse
import logging
from datetime import datetime
from collections import defaultdict

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import mteb
import datasets

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """设置随机种子以确保可复现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True


class FollowIREvaluator:
    """FollowIR 评测器 - 单向量稠密检索模型版本"""

    def __init__(self, model_name='BAAI/bge-large-en-v1.5', device='cuda', batch_size=64):
        self.device = device
        self.batch_size = batch_size
        self.model_name = model_name
        
        logger.info(f"📥 正在加载模型: {model_name}")
        start_time = time.time()
        self.encoder = SentenceTransformer(model_name, device=device)
        logger.info(f"✅ 模型加载完成，耗时: {time.time() - start_time:.1f}s")

    def load_data(self, task_name):
        """加载 FollowIR 评测数据"""
        logger.info(f"📂 加载数据集: {task_name}")
        task = mteb.get_task(task_name)
        dataset_path = task.metadata.dataset["path"]

        corpus, q_og, q_changed, candidates = {}, {}, {}, {}

        try:
            logger.info("   加载 corpus...")
            ds_c = datasets.load_dataset(dataset_path, 'corpus', trust_remote_code=True)
            c_split = 'corpus' if 'corpus' in ds_c else 'train'
            for d in ds_c[c_split]:
                corpus[str(d.get('_id', d.get('id')))] = {'text': str(d.get('text', ''))}
            logger.info(f"   ✅ 加载 {len(corpus)} 个文档")

            logger.info("   加载 queries...")
            ds_q = datasets.load_dataset(dataset_path, 'queries', trust_remote_code=True)
            q_split = 'queries' if 'queries' in ds_q else 'train'

            logger.info("   加载 instruction...")
            ds_inst = datasets.load_dataset(dataset_path, 'instruction', trust_remote_code=True)
            i_split = 'instruction' if 'instruction' in ds_inst else 'train'

            instruction_dict = {}
            for inst_item in ds_inst[i_split]:
                qid = str(inst_item.get('query-id', ''))
                inst_text = str(inst_item.get('instruction', ''))
                instruction_dict[qid] = inst_text

            for q in ds_q[q_split]:
                full_qid = str(q.get('_id', q.get('id', '')))
                query_text = q.get('text', '')
                inst = instruction_dict.get(full_qid, "")

                if full_qid.endswith('-og'):
                    q_og[full_qid] = f"{query_text} {inst}".strip()
                elif full_qid.endswith('-changed'):
                    q_changed[full_qid] = f"{query_text} {inst}".strip()

            logger.info(f"   ✅ 加载 {len(q_og)} 个 og 查询, {len(q_changed)} 个 changed 查询")

            try:
                logger.info("   加载 top_ranked...")
                ds_top = datasets.load_dataset(dataset_path, 'top_ranked', trust_remote_code=True)
                available_splits = list(ds_top.keys())
                t_split = available_splits[0] if available_splits else None
                if t_split:
                    for item in ds_top[t_split]:
                        full_qid = str(item.get('query-id', item.get('query_id', item.get('qid', ''))))
                        base_qid = full_qid.replace('-og', '').replace('-changed', '')
                        results_list = item.get('corpus-ids', item.get('results', []))

                        if base_qid not in candidates:
                            candidates[base_qid] = [str(did) for did in results_list]

                    if candidates:
                        avg_cand = sum(len(v) for v in candidates.values()) / len(candidates)
                        logger.info(f"   ✅ 加载 {len(candidates)} 个查询的候选文档, 平均 {avg_cand:.0f} 个/查询")
            except Exception as e:
                logger.warning(f"   ⚠️ 无法加载 top_ranked: {e}")

        except Exception as e:
            logger.error(f"加载数据出错: {e}")
            import traceback
            traceback.print_exc()

        logger.info(f"📂 数据加载完成")
        return corpus, q_og, q_changed, candidates

    def encode_queries(self, queries_dict):
        """编码查询文本"""
        queries_list = list(queries_dict.keys())
        query_texts = [queries_dict[qid] for qid in queries_list]
        
        logger.info(f"📝 编码 {len(queries_list)} 个查询 (batch_size={self.batch_size})...")
        embeddings = self.encoder.encode(
            query_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        query_embeddings = {qid: embeddings[idx] for idx, qid in enumerate(queries_list)}
        logger.info(f"✅ 查询编码完成")
        return query_embeddings

    def encode_documents(self, corpus, doc_ids):
        """编码文档集合"""
        doc_texts = [corpus[did].get('text', '') for did in doc_ids]
        
        logger.info(f"📚 编码 {len(doc_ids)} 个文档 (batch_size={self.batch_size})...")
        embeddings = self.encoder.encode(
            doc_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        
        doc_embeddings = {did: embeddings[idx] for idx, did in enumerate(doc_ids)}
        logger.info(f"✅ 文档编码完成")
        return doc_embeddings

    def compute_scores(self, query_embeddings, doc_embeddings, doc_ids_list):
        """计算查询与文档的点积分数（因为已经归一化，点积等于余弦相似度）"""
        doc_emb = torch.stack([doc_embeddings[did] for did in doc_ids_list]).to(self.device)
        query_emb = query_embeddings.to(self.device)
        
        scores = torch.matmul(query_emb, doc_emb.T).squeeze(0)
        return scores.cpu().tolist()

    def rerank(self, queries_dict, corpus, candidates):
        """执行重排检索"""
        results = {}

        all_doc_ids_set = set()
        for qid, doc_ids in candidates.items():
            all_doc_ids_set.update(doc_ids)
        all_doc_ids = list(all_doc_ids_set)
        
        logger.info(f"🔧 预编码文档集合，共 {len(all_doc_ids)} 个唯一文档")
        doc_embeddings = self.encode_documents(corpus, all_doc_ids)

        logger.info(f"🔧 编码 og 查询集合...")
        q_og_filtered = {k: v for k, v in queries_dict.items() if k.endswith('-og')}
        og_query_embeddings = self.encode_queries(q_og_filtered)
        
        logger.info(f"🔧 编码 changed 查询集合...")
        q_changed_filtered = {k: v for k, v in queries_dict.items() if k.endswith('-changed')}
        changed_query_embeddings = self.encode_queries(q_changed_filtered)

        query_embeddings_combined = {**og_query_embeddings, **changed_query_embeddings}

        logger.info("🔍 开始计算检索分数...")
        for qid in tqdm(query_embeddings_combined.keys(), desc="检索"):
            base_qid = qid.replace('-og', '').replace('-changed', '')
            if base_qid not in candidates or not candidates[base_qid]:
                logger.warning(f"⚠️ 查询 {qid} 没有候选文档")
                continue

            doc_ids = candidates[base_qid]
            scores = self.compute_scores(query_embeddings_combined[qid], doc_embeddings, doc_ids)
            
            q_results = {doc_id: score for doc_id, score in zip(doc_ids, scores)}
            results[qid] = q_results

        return results


def save_to_trec_format(results, output_path, run_name="bge_followir"):
    """保存为 TREC 格式"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for q_id, doc_scores in results.items():
            sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
            for rank_idx, (doc_id, score) in enumerate(sorted_docs, start=1):
                f.write(f"{q_id} Q0 {doc_id} {rank_idx} {score:.6f} {run_name}\n")
    logger.info(f"💾 TREC 文件已保存至: {output_path}")


def get_rank_from_dict(rank_dict, doc_id):
    """从排名字典中获取文档的排名"""
    if doc_id not in rank_dict:
        return -1, None
    
    sorted_docs = sorted(rank_dict.items(), key=lambda x: x[1], reverse=True)
    for rank, (did, score) in enumerate(sorted_docs, start=1):
        if did == doc_id:
            return rank, score
    return -1, None


def rank_score(og_rank, new_rank):
    """计算单个文档的 rank score"""
    if og_rank <= 0 or new_rank <= 0:
        return 0.0
    
    if og_rank >= new_rank:
        result = (1 / og_rank) / (1 / new_rank) - 1
    else:
        result = 1 - ((1 / new_rank) / (1 / og_rank))
    
    return result


def calculate_pmrr(results_og, results_changed, changed_qrels):
    """计算 p-MRR"""
    qid_pmrr = {}
    
    for qid in changed_qrels.keys():
        og_key = qid + '-og'
        changed_key = qid + '-changed'
        
        if og_key not in results_og or changed_key not in results_changed:
            continue
        
        original_qid_run = results_og[og_key]
        new_qid_run = results_changed[changed_key]
        
        query_scores = []
        for changed_doc in changed_qrels[qid]:
            original_rank, original_score = get_rank_from_dict(original_qid_run, changed_doc)
            new_rank, new_score = get_rank_from_dict(new_qid_run, changed_doc)
            
            if original_rank < 0 or new_rank < 0:
                continue
            
            score = rank_score(original_rank, new_rank)
            query_scores.append(score)
        
        if query_scores:
            qid_pmrr[qid] = sum(query_scores) / len(query_scores)
    
    return qid_pmrr


def evaluate_followir_metrics(task_name, trec_dir, output_dir):
    """调用 FollowIR 评测接口计算指标"""
    logger.info("📊 计算 FollowIR 评测指标...")
    
    from mteb._evaluators.retrieval_metrics import evaluate_p_mrr_change
    
    dataset_path_map = {
        "Core17InstructionRetrieval": "jhu-clsp/core17-instructions-mteb",
        "Robust04InstructionRetrieval": "jhu-clsp/robust04-instructions-mteb",
        "News21InstructionRetrieval": "jhu-clsp/news21-instructions-mteb",
    }
    
    dataset_path = dataset_path_map.get(task_name)
    if not dataset_path:
        logger.error(f"未知任务: {task_name}")
        return None
    
    ds_top = datasets.load_dataset(dataset_path, 'top_ranked', trust_remote_code=True)
    available_splits = list(ds_top.keys())
    t_split = available_splits[0] if available_splits else None
    
    changed_qrels = {}
    if t_split:
        for item in ds_top[t_split]:
            qid = str(item.get('query-id', item.get('query_id', item.get('qid', ''))))
            if qid.endswith('-changed'):
                base_qid = qid.replace('-changed', '')
                results_list = item.get('corpus-ids', item.get('results', []))
                changed_qrels[base_qid] = [str(did) for did in results_list]
    
    run_og_path = os.path.join(trec_dir, f"run_{task_name}_og.trec")
    run_changed_path = os.path.join(trec_dir, f"run_{task_name}_changed.trec")
    
    if not os.path.exists(run_og_path) or not os.path.exists(run_changed_path):
        logger.error("TREC 文件不存在")
        return None
    
    results_og = {}
    with open(run_og_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid = parts[0]
                docid = parts[2]
                score = float(parts[4])
                if qid not in results_og:
                    results_og[qid] = {}
                results_og[qid][docid] = score
    
    results_changed = {}
    with open(run_changed_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid = parts[0]
                docid = parts[2]
                score = float(parts[4])
                if qid not in results_changed:
                    results_changed[qid] = {}
                results_changed[qid][docid] = score
    
    qid_pmrr = calculate_pmrr(results_og, results_changed, changed_qrels)
    
    if qid_pmrr:
        pmrr_value = sum(qid_pmrr.values()) / len(qid_pmrr)
    else:
        pmrr_value = 0.0
    
    from mteb import MTEB
    task = MTEB().tasks_dict[task_name]
    
    evaluation = mteb.get_evaluator(task)
    
    ndcg_scores = {}
    try:
        scores_og = evaluation.evaluate(results_og)
        if 'og' in scores_og:
            ndcg_scores['og'] = scores_og['og'].get('ndcg_at_5', 0.0)
    except Exception as e:
        logger.warning(f"OG 评测失败: {e}")
        ndcg_scores['og'] = 0.0
    
    try:
        scores_changed = evaluation.evaluate(results_changed)
        if 'changed' in scores_changed:
            ndcg_scores['changed'] = scores_changed['changed'].get('ndcg_at_5', 0.0)
    except Exception as e:
        logger.warning(f"Changed 评测失败: {e}")
        ndcg_scores['changed'] = 0.0
    
    final_results = {
        'p-MRR': pmrr_value,
        'original': {
            'ndcg_at_5': ndcg_scores.get('og', 0.0)
        },
        'changed': {
            'ndcg_at_5': ndcg_scores.get('changed', 0.0)
        }
    }
    
    return final_results


def main():
    parser = argparse.ArgumentParser(description='FollowIR 评测脚本 - 单向量稠密检索模型')
    parser.add_argument('--model_name', type=str, default='BAAI/bge-large-en-v1.5', help='模型名称或路径')
    parser.add_argument('--task', type=str, default='Core17InstructionRetrieval', 
                        help='评测任务 (Core17InstructionRetrieval/Robust04InstructionRetrieval/News21InstructionRetrieval)')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--batch_size', type=int, default=64, help='批处理大小')
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        args.output_dir = f"/home/luwa/Documents/DSCLR/evaluation/bge_followir_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("🚀 FollowIR 评测开始")
    logger.info(f"   模型: {args.model_name}")
    logger.info(f"   任务: {args.task}")
    logger.info(f"   设备: {args.device}")
    logger.info(f"   批处理大小: {args.batch_size}")
    logger.info(f"   输出目录: {args.output_dir}")
    logger.info("=" * 60)
    
    evaluator = FollowIREvaluator(
        model_name=args.model_name,
        device=args.device,
        batch_size=args.batch_size
    )
    
    corpus, q_og, q_changed, candidates = evaluator.load_data(args.task)
    
    if not corpus or not q_og:
        logger.error("❌ 数据加载失败")
        return
    
    avg_cand = sum(len(v) for v in candidates.values()) / len(candidates) if candidates else 0
    logger.info(f"📊 共 {len(q_og)} og 查询 + {len(q_changed)} changed 查询, 平均 {avg_cand:.0f} 候选文档")
    
    trec_dir = os.path.join(args.output_dir, "trec")
    os.makedirs(trec_dir, exist_ok=True)
    
    all_queries = {**q_og, **q_changed}
    
    logger.info("--- 开始检索: og queries ---")
    q_og_results = evaluator.rerank(q_og, corpus, candidates)
    run_og_path = os.path.join(trec_dir, f"run_{args.task}_og.trec")
    save_to_trec_format(q_og_results, run_og_path)
    
    logger.info("--- 开始检索: changed queries ---")
    q_changed_results = evaluator.rerank(q_changed, corpus, candidates)
    run_changed_path = os.path.join(trec_dir, f"run_{args.task}_changed.trec")
    save_to_trec_format(q_changed_results, run_changed_path)
    
    logger.info("--- 计算 FollowIR 评测指标 ---")
    results = evaluate_followir_metrics(args.task, trec_dir, args.output_dir)
    
    if results:
        result_file = os.path.join(args.output_dir, f"results_{args.task}.json")
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info("=" * 60)
        logger.info("📊 评测结果:")
        logger.info(f"   p-MRR: {results.get('p-MRR', 0):.4f}")
        logger.info(f"   og nDCG@5: {results.get('original', {}).get('ndcg_at_5', 0):.4f}")
        logger.info(f"   changed nDCG@5: {results.get('changed', {}).get('ndcg_at_5', 0):.4f}")
        logger.info("=" * 60)
        logger.info(f"💾 结果已保存至: {result_file}")
    
    logger.info("✅ 评测完成!")


if __name__ == "__main__":
    main()
