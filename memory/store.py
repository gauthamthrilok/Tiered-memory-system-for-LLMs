from typing import Any
import chromadb
from chromadb.utils import embedding_functions
from observability.logger import log_stage
from memory.conflict import detect_conflict
from datetime import datetime
from rank_bm25 import BM25Okapi
import uuid
import math

#load the embedding model at import time
embed_fn = embedding_functions.DefaultEmbeddingFunction()

#init chromadb with a local persistent directory
client = chromadb.PersistentClient(path = "./chroma_db")

#create a 3 tier memory system
#semantic tier for storing facts and knowledge
sem_collection = client.get_or_create_collection(
    name = "semantic",
    embedding_function=embed_fn
)

#episodic tier to store events
ep_collection = client.get_or_create_collection(
    name = "episodic",
    embedding_function=embed_fn
)

#procedural tier stores the preferences of the user
proc_collection = client.get_or_create_collection(
    name = "procedural",
    embedding_function=embed_fn
)

COLLECTIONS = {
    "semantic" : sem_collection,
    "episodic" : ep_collection,
    "procedural" : proc_collection
}

SIMILARITY_THRESHOLD = {
    "semantic" : 0.3,
    "episodic" : 0.15,
    "procedural" : 0.3
}

SIMILARITY_WEIGHT = 0.6
RECENCY_WEIGHT = 0.25
FREQUENCY_WEIGHT = 0.15

RECECNCY_DECAY = 14

#returns true if a semantically similar text exists in vector store, false otherwise
def is_duplicate(text : str, tier : str) -> bool:
    #vector store is empty
    if not COLLECTIONS[tier].count():
        return False

    result = COLLECTIONS[tier].query(
        query_texts = [text],
        n_results = 1
    )

    distance = result["distances"][0][0]
    return distance < SIMILARITY_THRESHOLD[tier]

def store_to_memory(text : str,tier : str,metadata : dict = {}) -> bool:

    with log_stage("storage",fact=text,tier=tier) as log:
        if tier not in COLLECTIONS:
            log["result"] = "invalid tier"
            return False

        """if is_duplicate(text,tier):
            return False"""

        collection = COLLECTIONS[tier]
        similar_ids = []

        if collection.count() > 0:
            results = collection.query(
                query_texts = [text],
                n_results = min(3,collection.count()),
                include = ["documents","metadatas","distances"]
            )

            documents = results["documents"][0]
            metadatas = results["metadatas"][0]
            distances = results["distances"][0]
            ids = results["ids"][0]

            for doc,meta,id_,dist in zip(documents,metadatas,ids,distances):
                if dist < 0.6 and not meta.get("superseded",False):
                    similar_ids.append((id_,meta,doc))
            
            if similar_ids:
                conflict_result = detect_conflict(text,[doc for _,_,doc in similar_ids])
                classification = conflict_result.get("classification","NEW")
            else:
                classification = "NEW"

            log["conflict_classification"] = classification

            if classification == "REDUNDANT":
                log["result"] = "duplicate_skipped"
                return False

            if classification == "CONTRADICTS":
                conflicting_text = conflict_result.get("conflicting_memory","")

                for id_,meta,doc in similar_ids:
                    if doc.lower() in conflicting_text.lower() or conflicting_text.lower() in doc.lower():
                        collection.update(
                            ids = [id_],
                            metadatas = [{**meta, "superseded" : True,"superseding_text" : text}]
                        )
                        log["superseded_memory"] = doc
                        break

        #create a unique identifier for the memory entry
        memory_id = str(uuid.uuid4())
        timeStamp = datetime.now().isoformat()

        collection.add(
            documents = [text],
            metadatas = [{"timestamp" : timeStamp, "tier" : tier, "access_count" : 0, "superseded" : False, **metadata}],
            ids = [memory_id]
        )

        log["result"] = "stored"
        log["memory_id"] = memory_id 

    return True

#compute the score of retrieved context to choose which ones to select
def compute_score(distance : float,metadata : dict) -> float:
    similarity_score = 1/(1+distance)

    timestamp = datetime.fromisoformat(metadata.get("timestamp",datetime.now().isoformat()))
    age_days = (datetime.now()-timestamp).total_seconds()/86400
    recency_score = math.exp(-age_days/RECECNCY_DECAY)

    access_count = metadata.get("access_count",0)
    frequency_score = access_count/(access_count+1)

    return (SIMILARITY_WEIGHT*similarity_score + RECENCY_WEIGHT*recency_score + FREQUENCY_WEIGHT*frequency_score)

def _bm25_search(query : str, documents : list[str], ids : list[str], n_results : int) -> list[tuple[str,str]]:
    if not documents:
        return []

    tokenised_docs = [doc.lower().split() for doc in documents]
    tokenised_query = query.lower().split()

    bm25 = BM25Okapi(tokenised_docs)
    scores = bm25.get_scores(tokenised_query)

    scored = sorted(
        zip(ids,documents,scores),
        key = lambda x : x[2],
        reverse = True
    )

    return [(mem_id,doc) for mem_id,doc,score in scored[:n_results] if score > 0]

def rrf(vector_scores : list[tuple[str,str,float,dict]], bm_scores : list[tuple[str,str]], k : int = 60) -> list[str]:

    rrf_ranks : dict[str,float] = {}
    for rank, (mem_id, doc, dist, meta) in enumerate(vector_scores):
        rrf_ranks[mem_id] = rrf_ranks.get(mem_id,0) + (1 / (1 + rank + k))

    for rank, (mem_id, doc) in enumerate(bm_scores):
        rrf_ranks[mem_id] = rrf_ranks.get(mem_id,0) + (1 / (1 + rank + k))

    return sorted(rrf_ranks.keys(),key = lambda x : rrf_ranks[x], reverse = True)

def retrieve_memory(query : str,n_result : int = 2,pool : int = 5) -> dict:

    with log_stage("retrieval",query=query,n_result=n_result) as log:
        retrieved = {}

        for tier,collection in COLLECTIONS.items():
            if not collection.count():
                retrieved[tier] = []
                continue

            #number of memory storage to return
            n = min(pool,collection.count())

            vector_results = collection.query(
                query_texts = [query],
                n_results = n,
                include = ["documents","metadatas","distances"]
            )

            documents = vector_results["documents"][0]
            metadatas = vector_results["metadatas"][0]
            distances = vector_results["distances"][0]
            ids = vector_results["ids"][0]

            vector_candidates = [(mem_id,doc,dist,meta) for mem_id,doc,dist,meta in zip(ids,documents,distances,metadatas) if not meta.get("superseded", False)]

            #bm25 search
            all_documents = collection.get(include = ["documents","metadatas"])
            all_docs = all_documents["documents"]
            all_ids = all_documents["ids"]
            all_metas = all_documents["metadatas"]

            active_docs = [(mem_id,doc) for mem_id,doc,meta in zip(all_ids,all_docs,all_metas) if not meta.get("superseded",False)]

            active_ids = [x[0] for x in active_docs]
            active_documents = [x[1] for x in active_docs]

            bm25_candidates = _bm25_search(query,active_documents,active_ids,n)

            merged_ids = rrf(vector_candidates,bm25_candidates)

            id_to_doc : dict[str,(str,float,dict)] = {}

            for mem_id,doc,dist,meta in vector_candidates:
                id_to_doc[mem_id] = (doc,dist,meta)

            id_to_meta = {mem_id : meta for mem_id,meta in zip(all_ids,all_metas)}
            for mem_id,doc in bm25_candidates:
                id_to_doc[mem_id] = (doc,0.5,id_to_meta.get(mem_id,{}))

            top_merged = merged_ids[:n]
            scored = []
            for mem_id in top_merged:
                if mem_id not in id_to_doc:
                    continue

                doc,dist,meta = id_to_doc[mem_id]
                score = compute_score(dist,meta)
                scored.append((score,doc,meta,mem_id))

            scored.sort(key = lambda x : x[0],reverse = True)
            top = scored[:min(n_result,len(scored))]

            for _,_,meta,mem_id in top:
                new_count = meta.get("access_count",0) + 1
                collection.update(
                    ids = [mem_id],
                    metadatas = [{**meta, "access_count" : new_count}]
                )

            retrieved[tier] = [doc for _,doc,_,_ in top]

            log["retrieved"] = retrieved
            log["total_returned"] = sum(len(v) for v in retrieved.values())

    return retrieved