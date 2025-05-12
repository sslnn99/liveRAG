# # AWS utilities
# import boto3
# import json


# AWS_PROFILE_NAME = "sigir-participant"
# AWS_REGION_NAME = "us-east-1"

# def get_ssm_value(key: str, profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME) -> str:
#     """Get a cleartext value from AWS SSM."""
#     session = boto3.Session(profile_name=profile, region_name=region)
#     ssm = session.client("ssm")
#     return ssm.get_parameter(Name=key)["Parameter"]["Value"]

# def get_ssm_secret(key: str, profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME):
#     session = boto3.Session(profile_name=profile, region_name=region)
#     ssm = session.client("ssm")
#     return ssm.get_parameter(Name=key, WithDecryption=True)["Parameter"]["Value"]
# # Pinecone sample

# print(get_ssm_secret("/pinecone/ro_token"))
# print(get_ssm_value("/opensearch/endpoint"))

# from typing import List, Literal, Tuple
# from multiprocessing.pool import ThreadPool
# import boto3
# from pinecone import Pinecone
# import torch
# from functools import cache
# from transformers import AutoModel, AutoTokenizer

# PINECONE_INDEX_NAME = "fineweb10bt-512-0w-e5-base-v2"
# PINECONE_NAMESPACE="default"

# @cache
# def has_mps():
#     return torch.backends.mps.is_available()

# @cache
# def has_cuda():
#     return torch.cuda.is_available()

# @cache
# def get_tokenizer(model_name: str = "/data/sangshuailong/sigir_liveRAG/local_model/e5-base-v2"):
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     return tokenizer

# @cache
# def get_model(model_name: str = "/data/sangshuailong/sigir_liveRAG/local_model/e5-base-v2"):
#     model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
#     if has_mps():
#         model = model.to("mps")
#     elif has_cuda():
#         model = model.to("cuda")
#     else:
#         model = model.to("cpu")
#     return model

# def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
#     last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
#     return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# def embed_query(query: str,
#                 query_prefix: str = "query: ",
#                 model_name: str = "/data/sangshuailong/sigir_liveRAG/local_model/e5-base-v2",
#                 pooling: Literal["cls", "avg"] = "avg",
#                 normalize: bool =True) -> list[float]:
#     return batch_embed_queries([query], query_prefix, model_name, pooling, normalize)[0]

# def batch_embed_queries(queries: List[str], query_prefix: str = "query: ", model_name: str = "/data/sangshuailong/sigir_liveRAG/local_model/e5-base-v2", pooling: Literal["cls", "avg"] = "avg", normalize: bool =True) -> List[List[float]]:
#     with_prefixes = [" ".join([query_prefix, query]) for query in queries]
#     tokenizer = get_tokenizer(model_name)
#     model = get_model(model_name)
#     with torch.no_grad():
#         encoded = tokenizer(with_prefixes, padding=True, return_tensors="pt", truncation="longest_first")
#         encoded = encoded.to(model.device)
#         model_out = model(**encoded)
#         match pooling:
#             case "cls":
#                 embeddings = model_out.last_hidden_state[:, 0]
#             case "avg":
#                 embeddings = average_pool(model_out.last_hidden_state, encoded["attention_mask"])
#         if normalize:
#             embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
#     return embeddings.tolist()

# @cache
# def get_pinecone_index(index_name: str = PINECONE_INDEX_NAME):
#     pc = Pinecone(api_key=get_ssm_secret("/pinecone/ro_token"))
#     index = pc.Index(name=index_name)
#     return index

# def query_pinecone(query: str, top_k: int = 10, namespace: str = PINECONE_NAMESPACE) -> dict:
#     index = get_pinecone_index()
#     results = index.query(
#         vector=embed_query(query),
#         top_k=top_k,
#         include_values=False,
#         namespace=namespace,
#         include_metadata=True
#     )

#     return results

# def batch_query_pinecone(queries: list[str], top_k: int = 10, namespace: str = PINECONE_NAMESPACE, n_parallel: int = 10) -> list[dict]:
#     """Batch query a Pinecone index and return the results.

#     Internally uses a ThreadPool to parallelize the queries.
#     """
#     index = get_pinecone_index()
#     embeds = batch_embed_queries(queries)
#     pool = ThreadPool(n_parallel)
#     results = pool.map(lambda x: index.query(vector=x, top_k=top_k, include_values=False, namespace=namespace, include_metadata=True), embeds)
#     return results

# def show_pinecone_results(results):
#     for match in results["matches"]:
#         print("chunk:", match["id"], "score:", match["score"])
#         print(match["metadata"]["text"])
#         print()

# # OpenSearch sample
# from functools import cache
# from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection

# OPENSEARCH_INDEX_NAME = "fineweb10bt-512-0w-e5-base-v2"

# @cache
# def get_client(profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME):
#     credentials = boto3.Session(profile_name=profile).get_credentials()
#     auth = AWSV4SignerAuth(credentials, region=region)
#     host_name = get_ssm_value("/opensearch/endpoint", profile=profile, region=region)
#     aos_client = OpenSearch(
#         hosts=[{"host": host_name, "port": 443}],
#         http_auth=auth,
#         use_ssl=True,
#         verify_certs=True,
#         connection_class=RequestsHttpConnection,
#     )
#     return aos_client

# def query_opensearch(query: str, top_k: int = 1) -> dict:
#     """Query an OpenSearch index and return the results."""
#     client = get_client()
#     results = client.search(index=OPENSEARCH_INDEX_NAME, body={"query": {"match": {"text": query}}, "size": top_k})
#     return results

# def batch_query_opensearch(queries: list[str], top_k: int = 10, n_parallel: int = 10) -> list[dict]:
#     """Sends a list of queries to OpenSearch and returns the results. Configuration of Connection Timeout might be needed for serving large batches of queries"""
#     client = get_client()
#     request = []
#     for query in queries:
#         req_head = {"index": OPENSEARCH_INDEX_NAME}
#         req_body = {
#             "query": {
#                 "multi_match": {
#                     "query": query,
#                     "fields": ["text"],
#                 }
#             },
#             "size": top_k,
#         }
#         request.extend([req_head, req_body])

#     return client.msearch(body=request)



# def show_opensearch_results(results: dict):
#     for match in results["hits"]["hits"]:
#         print("chunk:", match["_id"], "score:", match["_score"])
#         print(match["_source"]["text"])
#         print()


# rag_utils.py
import boto3
from typing import List, Literal
from multiprocessing.pool import ThreadPool
from pinecone import Pinecone
import torch
from functools import cache
from transformers import AutoModel, AutoTokenizer
from opensearchpy import OpenSearch, AWSV4SignerAuth, RequestsHttpConnection

# AWS 配置常量
AWS_PROFILE_NAME = "sigir-participant"
AWS_REGION_NAME = "us-east-1"
PINECONE_INDEX_NAME = "fineweb10bt-512-0w-e5-base-v2"
PINECONE_NAMESPACE = "default"
OPENSEARCH_INDEX_NAME = "fineweb10bt-512-0w-e5-base-v2"

# AWS 工具函数
def get_ssm_value(key: str, profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME) -> str:
    session = boto3.Session(profile_name=profile, region_name=region)
    ssm = session.client("ssm")
    return ssm.get_parameter(Name=key)["Parameter"]["Value"]

def get_ssm_secret(key: str, profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME):
    session = boto3.Session(profile_name=profile, region_name=region)
    ssm = session.client("ssm")
    return ssm.get_parameter(Name=key, WithDecryption=True)["Parameter"]["Value"]

# 设备检测函数
@cache
def has_mps():
    return torch.backends.mps.is_available()

@cache
def has_cuda():
    return torch.cuda.is_available()

# 模型加载函数
@cache
def get_tokenizer(model_name: str = "/data/sangshuailong/sigir_liveRAG/local_model/e5-base-v2"):
    return AutoTokenizer.from_pretrained(model_name)

@cache
def get_model(model_name: str = "/data/sangshuailong/sigir_liveRAG/local_model/e5-base-v2"):
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
    device = "cuda" if has_cuda() else "mps" if has_mps() else "cpu"
    return model.to(device)

# 嵌入相关函数
def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def embed_query(query: str, query_prefix: str = "query: ",
               model_name: str = "/data/sangshuailong/sigir_liveRAG/local_model/e5-base-v2",
               pooling: Literal["cls", "avg"] = "avg", normalize: bool = True) -> list[float]:
    return batch_embed_queries([query], query_prefix, model_name, pooling, normalize)[0]

def batch_embed_queries(queries: List[str], query_prefix: str = "query: ",
                      model_name: str = "/data/sangshuailong/sigir_liveRAG/local_model/e5-base-v2",
                      pooling: Literal["cls", "avg"] = "avg", normalize: bool = True) -> List[List[float]]:
    with_prefixes = [" ".join([query_prefix, query]) for query in queries]
    tokenizer = get_tokenizer(model_name)
    model = get_model(model_name)
    with torch.no_grad():
        encoded = tokenizer(with_prefixes, padding=True, return_tensors="pt", truncation="longest_first")
        encoded = encoded.to(model.device)
        model_out = model(**encoded)
        if pooling == "cls":
            embeddings = model_out.last_hidden_state[:, 0]
        else:
            embeddings = average_pool(model_out.last_hidden_state, encoded["attention_mask"])
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.tolist()

# Pinecone 相关函数
@cache
def get_pinecone_index(index_name: str = PINECONE_INDEX_NAME):
    pc = Pinecone(api_key=get_ssm_secret("/pinecone/ro_token"))
    return pc.Index(name=index_name)

def query_pinecone(query: str, top_k: int = 10, namespace: str = PINECONE_NAMESPACE) -> dict:
    index = get_pinecone_index()
    return index.query(
        vector=embed_query(query),
        top_k=top_k,
        include_values=False,
        namespace=namespace,
        include_metadata=True
    )

def batch_query_pinecone(queries: list[str], top_k: int = 10,
                       namespace: str = PINECONE_NAMESPACE, n_parallel: int = 10) -> list[dict]:
    index = get_pinecone_index()
    embeds = batch_embed_queries(queries)
    pool = ThreadPool(n_parallel)
    return pool.map(
        lambda x: index.query(vector=x, top_k=top_k, include_values=False, 
                            namespace=namespace, include_metadata=True), 
        embeds
    )

def show_pinecone_results(results):
    for match in results["matches"]:
        print("chunk:", match["id"], "score:", match["score"])
        print(match["metadata"]["text"])
        print()

# OpenSearch 相关函数
@cache
def get_opensearch_client(profile: str = AWS_PROFILE_NAME, region: str = AWS_REGION_NAME):
    credentials = boto3.Session(profile_name=profile).get_credentials()
    auth = AWSV4SignerAuth(credentials, region=region)
    host_name = get_ssm_value("/opensearch/endpoint", profile=profile, region=region)
    return OpenSearch(
        hosts=[{"host": host_name, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
    )

def query_opensearch(query: str, top_k: int = 5) -> dict:
    client = get_opensearch_client()
    return client.search(
        index=OPENSEARCH_INDEX_NAME,
        body={"query": {"match": {"text": query}}, "size": top_k}
    )

def batch_query_opensearch(queries: list[str], top_k: int = 10, n_parallel: int = 10) -> list[dict]:
    client = get_opensearch_client()
    request = []
    for query in queries:
        req_head = {"index": OPENSEARCH_INDEX_NAME}
        req_body = {
            "query": {"multi_match": {"query": query, "fields": ["text"]}},
            "size": top_k
        }
        request.extend([req_head, req_body])
    return client.msearch(body=request)

def show_opensearch_results(results: dict):
    for match in results["hits"]["hits"]:
        print("chunk:", match["_id"], "score:", match["_score"])
        print(match["_source"]["text"])
        print()

# 显式导出列表
__all__ = [
    # AWS 函数
    'get_ssm_value', 'get_ssm_secret',
    
    # 模型函数
    'get_tokenizer', 'get_model', 'has_mps', 'has_cuda',
    
    # 嵌入函数
    'average_pool', 'embed_query', 'batch_embed_queries',
    
    # Pinecone 函数
    'get_pinecone_index', 'query_pinecone', 'batch_query_pinecone', 'show_pinecone_results',
    
    # OpenSearch 函数
    'get_opensearch_client', 'query_opensearch', 'batch_query_opensearch', 'show_opensearch_results',
    
    # 常量
    'AWS_PROFILE_NAME', 'AWS_REGION_NAME', 
    'PINECONE_INDEX_NAME', 'PINECONE_NAMESPACE',
    'OPENSEARCH_INDEX_NAME'
]