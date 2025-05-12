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
from collections import defaultdict
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from sentence_transformers import CrossEncoder
from concurrent.futures import ThreadPoolExecutor
import json
import pandas as pd
from datetime import datetime


# 初始化 Rerank 模型
def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    return CrossEncoder(model_name)


def context_compress(query: str, context, generator) -> str:
    """
    压缩上下文
    """
    messages = [
        SystemMessage(content="""You are an experienced document compression expert. Compress the context of the reference document and preserve key information based on the following questions:"""),
        HumanMessage(content=f"""questions: {query}
        reference document: {context}  # 限制输入长度
                        
Directly output compressed document:""")
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
    response = generator.invoke(messages)
    # 将多行字符串转换为列表
    try:
        # 按行分割
        lines = response.content.strip().splitlines()
        # 跳过第一行（"Rewritten Query:"）
        lines = lines if lines else []
        # 去除每行的序号和多余空格
        rewritten_queries = [line.split(". ", 1)[1].strip() for line in lines if line.strip()]
        rewritten_queries = [query]+rewritten_queries
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
        max_tokens=100
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


def normalize_scores(docs, source):
    if not docs:
        return docs
    scores = [doc['score'] for doc in docs]
    min_score, max_score = min(scores), max(scores)
    # OpenSearch分数范围大，需压缩到0-1
    if source == 'opensearch' and max_score > 1:
        for doc in docs:
            doc['normalized_score'] = (doc['score'] - min_score) / (max_score - min_score)
    else:  # Pinecone/HyDE分数已在0-1范围内
        for doc in docs:
            doc['normalized_score'] = doc['score']
    return docs


def deduplicate_with_boost(os_docs, pc_docs, hyde_docs, boost_factor=0.1):
    # 合并所有文档并记录来源
    all_docs = []
    for doc in os_docs:
        doc['sources'] = ['opensearch']
        all_docs.append(doc)
    for doc in pc_docs:
        doc['sources'] = ['pinecone']
        all_docs.append(doc)
    for doc in hyde_docs:
        doc['sources'] = ['hyde_pinecone']
        all_docs.append(doc)
    
    # 按doc_id分组
    doc_map = defaultdict(list)
    for doc in all_docs:
        doc_map[doc['doc_id']].append(doc)
    
    # 合并来源并加分
    unique_docs = []
    for doc_id, docs in doc_map.items():
        best_doc = max(docs, key=lambda x: x['normalized_score'])
        # 计算重复次数和加分
        repeat_count = len(docs)
        if repeat_count > 1:
            best_doc['boosted_score'] = min(
                best_doc['normalized_score'] * (1 + boost_factor * (repeat_count - 1)),
                1.0  # 确保不超过1.0
            )
        else:
            best_doc['boosted_score'] = best_doc['normalized_score']
        unique_docs.append(best_doc)
    
    return sorted(unique_docs, key=lambda x: -x['boosted_score'])


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
    use_hyde: bool = True,
    use_rrf: bool = True,
    rrf_k: int = 60,
    use_two_stage: bool = True,
    enable_verification: bool = True,
    boost_factor: float = 0.15  # 新增重复文档加分系数
) -> tuple:
    """
    优化后的混合检索流程
    """
    try:
        # 1. 并行执行查询重写和HYDE生成
        rewritten_queries = rewrite_query(query, generator)
        hyde_docs = generate_hyde_documents(query, generator) if use_hyde else []
        
        # 2. 并行检索（原始查询 + HYDE查询）
        with ThreadPoolExecutor() as executor:
            # 提交所有检索任务
            os_future = executor.submit(batch_query_opensearch, rewritten_queries, 10)
            pc_future = executor.submit(batch_query_pinecone, rewritten_queries, 10)
            hyde_pc_future = executor.submit(batch_query_pinecone, hyde_docs, 5) if hyde_docs else None
            
            # 获取结果并添加来源标记
            os_docs = [dict(doc, source='opensearch') 
                      for doc in extract_docs_from_opensearch(os_future.result())]
            pc_docs = [dict(doc, source='pinecone') 
                      for doc in extract_docs_from_pinecone(pc_future.result())]
            hyde_pc_docs = [dict(doc, source='hyde_pinecone') 
                           for doc in extract_docs_from_pinecone(hyde_pc_future.result())] if hyde_pc_future else []

        # 3. 归一化分数并去重加分
        def normalize(docs, source):
            if not docs: return docs
            scores = [d['score'] for d in docs]
            min_s, max_s = min(scores), max(scores)
            for doc in docs:
                doc['normalized_score'] = (doc['score']-min_s)/(max_s-min_s) if max_s>min_s else 0.5
            return docs

        ranked_os = normalize(os_docs, 'opensearch')
        ranked_pc = normalize(pc_docs, 'pinecone')
        ranked_hyde = normalize(hyde_pc_docs, 'hyde_pinecone') if hyde_docs else []

        # 去重并加分（优化版）
        doc_map = defaultdict(list)
        for doc in ranked_os + ranked_pc + ranked_hyde:
            doc_map[doc['doc_id']].append(doc)
        
        unique_docs = []
        for doc_id, docs in doc_map.items():
            best_doc = max(docs, key=lambda x: x['normalized_score'])
            repeat_count = len(docs)
            if repeat_count > 1:
                best_doc['boosted_score'] = min(
                    best_doc['normalized_score'] * (1 + boost_factor * (repeat_count - 1)),
                    1.0
                )
            else:
                best_doc['boosted_score'] = best_doc['normalized_score']
            unique_docs.append(best_doc)

        # 4. RRF融合（使用boosted_score排序）
        if use_rrf:
            source_groups = {
                'opensearch': [d for d in unique_docs if d['source'] == 'opensearch'],
                'pinecone': [d for d in unique_docs if d['source'] == 'pinecone'],
                'hyde_pinecone': [d for d in unique_docs if d['source'] == 'hyde_pinecone']
            }
            final_docs = rrf_fusion(
                rankings=[
                    sorted(source_groups['opensearch'], key=lambda x: -x['boosted_score']),
                    sorted(source_groups['pinecone'], key=lambda x: -x['boosted_score']),
                    sorted(source_groups['hyde_pinecone'], key=lambda x: -x['boosted_score'])
                ],
                k=rrf_k
            )[:top_k]
        else:
            final_docs = sorted(unique_docs, key=lambda x: -x['boosted_score'])[:top_k]

        # 5. rerank
        if len(final_docs) > 0:
            reranker = get_reranker()
            final_docs = rerank_results(query, final_docs, reranker)

        # 6. 答案生成流程优化
        context = "\n".join(doc['text'] for doc in final_docs)  # 仅用前3个文档
        compressed_context = context_compress(query, context, generator)
        final_prompt = get_final_prompt(query, compressed_context)
        answer = generate_answer(query, compressed_context, generator)

        # 7. 两阶段优化（可选）
        if use_two_stage and final_docs:
            answer = refine_answer_with_top_doc(
                query=query,
                first_answer=answer,
                top_doc=final_docs[0],
                generator=generator
            )

        # 8. 验证（可选）
        if enable_verification and final_docs:
            answer = verify_answer(
                query=query,
                answer=answer,
                top_doc=final_docs[0],
                generator=generator
            )

        return rewritten_queries, answer, final_docs, final_prompt

    except Exception as e:
        return [], f"生成答案时出错: {str(e)}", [], []


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
            

def main_batch():
    data = {
        "question": [],
        "rewritten_query": [],
        "answer": [],
        "contexts": [],
        "final_prompt": []
    }
    file_path = "LiveRAG_LCD_Session1_Question_file.jsonl"
    # 读取 JSONL 文件
    df = pd.read_json(file_path, lines=True)

    # 提取 id 和 question 列
    ids = df['id'].tolist()
    questions = df['question'].tolist()

    for query,i in zip(questions, ids):
        query = query.strip()
        if not query:
            continue
            
        # alpha = detect_query_type(query)
        rewritten_query, answer, retrieved_docs, final_prompt = hybrid_retrieve_with_hyde(
            query, 
            generator, 
            top_k=5, 
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
        data["final_prompt"].append(final_prompt)
        

        print(f"Question(问题) {i}: {query}")
        print(f"Answer: {answer}")
        print('=' * 50)
        save_to_jsonl(data, output_file="answer.jsonl")


if __name__ == "__main__":
    AI71_API_KEY = "ai71-api-e141e495-2f9f-4cd2-b100-1e28f0784928"
    AI71_BASE_URL = "https://api.ai71.ai/v1/"
    generator = get_generator(AI71_API_KEY, AI71_BASE_URL)


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
