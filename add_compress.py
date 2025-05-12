from utils import (
    get_ssm_secret,
    query_pinecone,
    show_pinecone_results,
    query_opensearch,
    show_opensearch_results,
    batch_query_pinecone,
    # 嵌入函数
    embed_query,\
    batch_embed_queries,    
    # Pinecone 函数
    get_pinecone_index,
    query_pinecone,
    batch_query_pinecone,
    show_pinecone_results,
    # OpenSearch 函数
    get_opensearch_client,
    query_opensearch,
    batch_query_opensearch,
    show_opensearch_results,
    
)
from tqdm import tqdm

# 添加生成部分的代码
# from langchain.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder


# 初始化 Rerank 模型
def get_reranker(model_name: str = "/data/sangshuailong/sigir_liveRAG/local_model/ms-marco-MiniLM-L-6-v2"):
    return CrossEncoder(model_name)


def context_compress(query: str, context, generator) -> str:
    """
    压缩上下文
    """
    messages = [
        SystemMessage(content="""您是一个有丰富经验的文档压缩专家。根据以下问题压缩参考文档的上下文，保留关键信息："""),
        HumanMessage(content=f"""原始问题: {query}
        参考文档: {context}  # 限制输入长度
                        
请生成压缩后的文档上下文:""")
    ]
    
    try:
        response = generator.invoke(messages)
        return response.content
    except Exception as e:
        print(f"文档压缩失败: {str(e)}")
        return context  # 失败时回退到初始答案


# 批量重写查询
def rewrite_query(query: str, generator) -> list:
    messages = [
        SystemMessage(content="You are a natural language processing expert who helps users enhance the query performance of search engines. Given a problem, generate multiple query variants that are semantically similar but express different expressions. \
                            Each query can contain different keywords, synonyms, grammatical changes, etc. Ensure that these variations increase search recall and cover more relevant content. Rewrite the user's original query into three items to make it more suitable for retrieval, and output the rewritten query in the form of a list."),
        HumanMessage(content=f"Original Query: {query}"),
    ]
    # print("查询重写函数中的query和generator参数：", query)
    response = generator.invoke(messages)
    # print("查询重写后的输出：\n", response.content)

    # 将多行字符串转换为列表
    try:
        # 按行分割
        lines = response.content.strip().splitlines()
        # 跳过第一行（"Rewritten Query:"）
        lines = lines if lines else []
        # 去除每行的序号和多余空格
        rewritten_queries = [line.split(". ", 1)[1].strip() for line in lines if line.strip()]
        rewritten_queries = [query]+rewritten_queries
        # print("重写后的查询:", rewritten_queries)
        return rewritten_queries
    except Exception as e:
        print(f"解析重写后的查询时出错: {e}")
        return []

# 初始化生成模型
def get_generator(api_key: str, base_url: str, model: str = "tiiuae/Falcon3-10B-Instruct"):
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        streaming=True,
        max_tokens=10
    )


def refine_answer_with_top_doc(
    query: str,
    first_answer: str,
    top_doc: dict,
    generator
) -> str:
    """
    使用第一轮答案和Top1文档生成优化后的最终答案
    :param first_answer: 第一轮生成的初始答案
    :param top_doc: 第一轮检索得分最高的文档（包含text和metadata）
    :return: 优化后的最终答案
    """
    messages = [
        SystemMessage(content="""You are a professional fact checking assistant. Please improve the initial answer based on the following information:
1. If the reference document can supplement important details, integrate them into the answer
2. If the initial answer conflicts with the document, the document shall prevail
3. Keep the answer concise (within 40 words)"""),
        HumanMessage(content=f"""原始问题: {query}
                        
initial answer: {first_answer}
                        
reference document: {top_doc['text'][:1000]}  # 限制输入长度
                        
Provide the optimized final answer directly:""")
    ]
    
    try:
        response = generator.invoke(messages)
        return response.content
    except Exception as e:
        print(f"答案优化失败: {str(e)}")
        return first_answer  # 失败时回退到初始答案
    

def verify_answer(
    query: str,
    answer: str,
    top_doc: dict,  # top1文档内容
    generator
) -> str:
    """
    执行答案验证并返回修正结果
    """
    messages = [
        SystemMessage(content="""You are a rigorous fact checker. Please execute:
1. Compare the factual consistency between [Answer] and [Reference Document]
2. Identify specific points of contradiction (if any)
3. Generate the revised final answer"""),
        HumanMessage(content=f"""question: {query}
        
当前答案: {answer}
        
参考文档: {top_doc['text'][:1000]}  # 控制输入长度
        
请生成修正后的最终答案,只输出答案即可！:""")
    ]
    
    try:
        response = generator.invoke(messages)
        return response.content
    except Exception as e:
        print(f"验证失败: {str(e)}")
        return answer  # 失败时回退到初始答案
    

def generate_hyde_documents(query: str, generator, num_hypotheses: int = 1) -> list[str]:
    """
    生成假设性文档（HYDE的核心功能）
    :param query: 原始查询
    :param generator: 生成模型
    :param num_hypotheses: 生成假设性答案的数量
    :return: 假设性文档列表
    """
    messages = [
        SystemMessage(content="You are a helpful assistant. Given a question, generate hypothetical answers that might contain the information needed to answer it."),
        HumanMessage(content=f"Question: {query}\n\nGenerate {num_hypotheses} possible answers to this question, each on a new line.")
    ]
    
    try:
        response = generator.invoke(messages)
        # 解析生成的假设性答案
        hyde_docs = [doc.strip() for doc in response.content.split('\n') if doc.strip()]
        return hyde_docs[:num_hypotheses]
    except Exception as e:
        print(f"HYDE生成失败: {str(e)}")
        return []
    

def rrf_fusion(rankings: list[list[dict]], k: int = 60) -> list[dict]:
    """
    原始RRF算法实现（无权重）
    :param rankings: 多个排序结果列表（每个列表已按score降序排列）
    :param k: 平滑常数（建议60）
    :return: 按RRF分数排序的文档列表
    """
    doc_scores = defaultdict(float)
    
    # 计算每个排序中的RRF贡献（所有排序平等对待）
    for ranking in rankings:
        for rank, doc in enumerate(ranking, 1):  # 排名从1开始
            doc_scores[doc['doc_id']] += 1 / (k + rank)
    
    # 合并文档元数据
    all_docs = {doc['doc_id']: doc for ranking in rankings for doc in ranking}
    
    # 按RRF分数降序排序
    return sorted(all_docs.values(), key=lambda x: -doc_scores[x['doc_id']])



def weighted_rrf_fusion(rankings: list[list[dict]], 
                        weights: list[float], 
                        k: int = 60) -> list[dict]:
    """
    加权RRF算法实现
    :param rankings: 多个排序结果列表（顺序必须对应权重顺序）
    :param weights: 各排序列表的权重（如[0.5, 0.3, 0.2]）
    :param k: 平滑常数（建议60）
    :return: 按加权RRF分数排序的文档列表
    """
    # 验证输入
    if len(rankings) != len(weights):
        raise ValueError("rankings和weights的长度必须相同")
    if not all(0 <= w <= 1 for w in weights):
        raise ValueError("权重必须在0到1之间")
    
    doc_scores = defaultdict(float)
    
    # 计算加权RRF分数（pinecone权重0.5，opensearch 0.3，hyde 0.2）
    for ranking, weight in zip(rankings, weights):
        for rank, doc in enumerate(ranking, 1):  # 排名从1开始
            doc_scores[doc['doc_id']] += weight * (1 / (k + rank))
    
    # 合并文档元数据（保留所有字段）
    all_docs = {}
    for ranking in rankings:
        for doc in ranking:
            doc_id = doc['doc_id']
            if doc_id not in all_docs:
                all_docs[doc_id] = doc
            else:
                # 合并字段（以第一个出现的为准）
                all_docs[doc_id].update({k: v for k, v in doc.items() 
                                       if k not in all_docs[doc_id]})
    
    # 按加权RRF分数降序排序
    return sorted(all_docs.values(), 
                 key=lambda x: -doc_scores[x['doc_id']])


def get_final_prompt(query: str, context: str) -> str:
    final_prompt = f"You are a helpful assistant.You can refer to the content of relevant context to answer the questions.Provide a concise answer of about 30 words.Context: {context}\n\nQuestion: {query}"
    return final_prompt

# 生成答案
def generate_answer(query: str, context: str, generator) -> str:
    messages = [
        SystemMessage(content="You are a helpful assistant.You can refer to the content of relevant context to answer the questions.Provide a concise answer of about 30 words."),
        HumanMessage(content=f"Context: {context}\n\nQuestion: {query}"),
    ]
    response = generator.invoke(messages)
    return response.content


def extract_docs_from_opensearch(response):
    """从OpenSearch返回带元数据的文档列表"""
    docs = []
    for resp in response['responses']:
        if 'hits' in resp and 'hits' in resp['hits']:
            for hit in resp['hits']['hits']:
                docs.append({
                    'text': hit['_source']['text'],
                    'doc_id': hit['_source']['doc_id'],
                    'score': hit.get('_score', 0)
                })
    return docs

def extract_docs_from_pinecone(response):
    """从Pinecone返回带元数据的文档列表""" 
    docs = []
    for resp in response:
        if 'matches' in resp:
            for match in resp['matches']:
                docs.append({
                    'text': match['metadata']['text'],
                    'doc_id': match['metadata']['doc_id'],
                    'score': match['score']
                })
    return docs


def rerank_results(query: str, documents: list[dict], reranker, top_k: int = 10) -> list[dict]:
    """支持带元数据文档的重排序"""
    if not documents:
        return []
    
    # 提取纯文本用于重排序
    texts = [doc['text'] for doc in documents]
    pairs = [[query, text] for text in texts]
    
    # 获取排序分数
    scores = reranker.predict(pairs)
    
    # 对文档按分数排序
    ranked = sorted(zip(documents, scores), key=lambda x: -x[1])
    
    # 返回排序后的文档（保留元数据）
    return [doc for doc, _ in ranked[:top_k]]
 

def detect_query_type(query: str) -> float:
    """启发式判断问题类型"""
    if any(word in query for word in [
        'what', 'What', 'WHAT',
        'when', 'When', 'WHEN',
        'where', 'Where', 'WHERE',
        'who', 'Who', 'WHO',
        'why', 'Why', 'WHY',
        # 'how', 'How', 'HOW'
    ]):
        return 0.7  # 事实型问题，偏向OpenSearch
    else:
        return 0.3  # 语义型问题，偏向Pinecone


def hybrid_retrieve_with_hyde(
    query: str, 
    generator, 
    top_k: int = 5,
    hyde_weight: float = 0.3,  # HYDE向量的权重
    use_hyde: bool = True,  # 是否启用HYDE
    use_rrf: bool = True,
    rrf_k: int = 60,
    use_two_stage: bool = True,  # 新增参数控制是否启用两阶段
    enable_verification: bool = True  # 新增参数控制是否启用验证
) -> tuple:
    """
    带HYDE的混合检索
    :param hyde_weight: HYDE向量的权重 (0-1)
    :param use_hyde: 是否启用HYDE功能
    """
    try:
        # 1. 查询重写
        rewritten_queries = rewrite_query(query, generator)
        
        # 2. 生成HYDE文档
        hyde_docs = []
        if use_hyde:
            hyde_docs = generate_hyde_documents(query, generator)
            print(f"生成的HYDE文档: {hyde_docs}")
        
        # 3. 并行检索
        # 原始查询的检索
        opensearch_results = batch_query_opensearch(rewritten_queries, top_k=10)
        os_docs = [dict(doc, source='opensearch') for doc in extract_docs_from_opensearch(opensearch_results)]
        
        pinecone_results = batch_query_pinecone(rewritten_queries, top_k=10)
        pc_docs = [dict(doc, source='pinecone') for doc in extract_docs_from_pinecone(pinecone_results)]
        
        # HYDE文档的检索（仅向量搜索）
        hyde_pc_docs = []
        if hyde_docs:
            hyde_pinecone_results = batch_query_pinecone(hyde_docs, top_k=5)
            hyde_pc_docs = [dict(doc, source='hyde_pinecone') for doc in extract_docs_from_pinecone(hyde_pinecone_results)]
        
        # 4. 文档去重
        doc_map = defaultdict(list)
        for doc in os_docs + pc_docs + hyde_pc_docs:
            doc_map[doc['doc_id']].append(doc)
        unique_docs = [max(docs, key=lambda x: x['score']) for docs in doc_map.values()]

        # 构建各来源的独立排序（仅包含去重后文档）
        def get_ranked(source):
            return sorted(
                [d for d in unique_docs if d['source'] == source],
                key=lambda x: -x['score']
            )
        
        ranked_os = get_ranked('opensearch')
        ranked_pc = get_ranked('pinecone')
        ranked_hyde = get_ranked('hyde_pinecone') if use_hyde else []

        # 若使用rrf排序：
        if use_rrf:
            print("使用原始RRF融合======================")
            # 使用原始RRF（不传weights参数）
            final_docs = rrf_fusion(
                rankings=[ranked_os, ranked_pc, ranked_hyde],
                k=rrf_k
            )[:top_k]

        # 8. 重排序
        reranker = get_reranker()
        final_docs = rerank_results(query, final_docs, reranker)
        
        # 9. 生成答案
        context = "\n".join(doc['text'] for doc in final_docs)
        # 压缩
        context = context_compress(query, context, generator)
        print(f"压缩完成======================")
        final_prompt = f"You are a helpful assistant.You can refer to the content of relevant context to answer the questions.Provide a concise answer of about 30 words.Context: {context}\n\nQuestion: {query}"
    
        answer = generate_answer(query, context, generator)

        # 10. 两阶段答案生成（新增）
        if use_two_stage and final_docs:
            top1_doc = final_docs[0]  # 取分数最高的文档
            answer1 = refine_answer_with_top_doc(
                query=query,
                first_answer=answer,
                top_doc=top1_doc,  # 取分数最高的文档
                generator=generator
            )
            final_answer = answer1
            # print(f"选择的最终答案: {final_answer}")
        else:
            final_answer = answer

        # 新增第三阶段验证
        verification_result = None
        if enable_verification and final_docs:
            verification_result = verify_answer(
                query=query,
                answer=final_answer,  # 第二轮生成的答案
                top_doc=final_docs[0],  # 取分数最高的文档
                generator=generator
            )
            # print(f"验证结果: {verification_result}")
            final_answer = verification_result
        
        return rewritten_queries, final_answer, final_docs, final_prompt
        
    except Exception as e:
        logging.error(f"带HYDE的混合检索失败: {str(e)}", exc_info=True)
        return [], f"生成答案时出错: {str(e)}", [], [] 

from collections import defaultdict
import logging
import json
  

import json

def convert_to_jsonl_entry(index, query, answer, contexts, final_prompt):
    passages = []
    for ctx in contexts:
        passages.append({
            "passage": ctx["text"][:100],
            "doc_IDs": [ctx["doc_id"]]
        })
    
    return {
        "id": index + 1,
        "question": query,
        "passages": passages,
        "final_prompt": final_prompt,
        "answer": answer
    }

def save_to_jsonl(data, output_file="output.jsonl"):
    with open(output_file, "w", encoding="utf-8") as f:
        for i in range(len(data["question"])):
            entry = convert_to_jsonl_entry(
                i,
                data["question"][i],
                data["answer"][i],
                data["contexts"][i],
                data["final_prompt"][i]
            )
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            # json.dump(data, file, ensure_ascii=False, indent=4)  # 写入时保留中文


import json
def main_batch():
    data = {
        "question": [],
        "rewritten_query": [],
        "answer": [],
        "contexts": [],
        "hyde_docs": [],  # 新增HYDE文档字段
        "ground_truth": [],
        "final_prompt": []
    }

    import pandas as pd
    file_path = "/data/sangshuailong/sigir_liveRAG/LiveRAG_LCD_Session1_Question_file.jsonl"
    # 读取 JSONL 文件
    df = pd.read_json(file_path, lines=True)

    # 提取 id 和 question 列
    ids = df['id'].tolist()[:250]
    questions = df['question'].tolist()[:250]

    # 打印前 5 条
    # print(df.head(5))
    for query,i in zip(questions, ids):
        query = query.strip()
        if not query:
            continue
            
        rewritten_query, answer, retrieved_docs, final_prompt = hybrid_retrieve_with_hyde(
            query, 
            generator, 
            top_k=5, 
            hyde_weight=0.3,  # 可调整
            use_hyde=True,  # 启用HYDE
            use_rrf=True,
            rrf_k=60,
            use_two_stage=True,
            enable_verification=True  # 启用验证
        )
        
        contexts = [doc for doc in retrieved_docs] if retrieved_docs else []
        
        data["question"].append(query)
        data["rewritten_query"].append(rewritten_query)
        data["answer"].append(answer)
        data["contexts"].append(contexts)
        # data["hyde_docs"].append(hyde_docs)  # 记录HYDE文档
        data["final_prompt"].append(final_prompt)
        

        print(f"Question(问题) {i}: {query}")
        # print(f"HYDE Docs: {hyde_docs}")
        print(f"Answer: {answer}")
        print('=' * 50)
        save_to_jsonl(data, output_file="/data/sangshuailong/sigir_liveRAG/DryTest/前250.jsonl")


if __name__ == "__main__":
    AI71_API_KEY = "ai71-api-e141e495-2f9f-4cd2-b100-1e28f0784928"
    AI71_BASE_URL = "https://api.ai71.ai/v1/"
    generator = get_generator(AI71_API_KEY, AI71_BASE_URL)
    from datetime import datetime

    # 记录开始时间（带日期）
    start_time = datetime.now()
    print(f"程序开始时间: {start_time}")

    main_batch()

    end_time = datetime.now()
    print(f"程序结束时间: {end_time}")
    # 新调用（混合检索）
        # 计算时间差
    elapsed = end_time - start_time
    print(f"总运行时间: {elapsed}")
