import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chromadb
from chromadb.utils import embedding_functions
from memory.store import retrieve_memory,store_to_memory,COLLECTIONS
from llm.client import chat
from testcases import TEST_CASES

embedding_fn = embedding_functions.DefaultEmbeddingFunction()

PROMPT = """You are an evaluation assistant.
Your job is to assess whether an AI response is faithful to the provided memory context.

A response is faithful if every factual claim it makes is supported by the memories.
A response is unfaithful if it introduces facts not present in the memories (hallucination).

Score from 0.0 to 1.0:
- 1.0: every claim is supported by the memories
- 0.5: some claims supported, some hallucinated
- 0.0: response ignores memories entirely or contradicts them

Output ONLY a JSON object in this exact format, nothing else:
{"score": <float>, "reason": "<one sentence explanation>"}
"""

def faith_score(retrieved : list[str],response : str) -> dict:
    memory_block = "\n".join(m for m in retrieved)
    user_content = (f"Retrieved memories : {memory_block}"
                    f"Response to evaluate : {response}")

    message = [{"role":"system","content":PROMPT},
                {"role":"user","content":user_content}]

    raw = chat(message)

    import json
    try:
        clean = raw.strip().replace("```json","").replace("```","").strip()
        return json.loads(clean)
    except:
        return {"score": 0.0, "reason": f"Failed to parse scorer output: {raw}"}

def recall_score(results : list[str],expected : list[str]) -> float:
    if not expected:
        return 1

    hits = sum(1 for e in expected if any(e.lower() in r.lower() for r in results))
    return hits/len(expected)

#delete and re initialise all three collections
def clear_memory():
    client = chromadb.PersistentClient("./chroma_db")
    for collection in ["semantic","episodic","procedural"]:
        try:
            client.delete_collection(collection)
        except:
            pass
    
    #re initialise store module in cache
    import importlib
    import memory.store as store_module
    importlib.reload(store_module)

#add context to vectore store
def seed_memories(memories : list[dict]):
    from memory.store import store_to_memory
    for entry in memories:
        store_to_memory(entry["text"],entry["tier"])

def run_eval():
    print("="*60)
    print("Memory pipeline evalutaion")
    print("="*60)

    recall_scores = []
    faith_scores = []

    for tc in TEST_CASES:
        print(f"\nID : {tc['id']} Description : {tc['description']}")
        print("-"*40)

        clear_memory()

        from memory.store import retrieve_memory,store_to_memory
        seed_memories(tc["seed_memories"])

        retrieved = retrieve_memory(tc["query"])
        all_retrieved = []
        for tier_results in retrieved.values():
            all_retrieved.extend(tier_results)

        recall_scores.append(recall_score(all_retrieved,tc["expected_retrievals"]))
        
        print(f"query : {tc['query']}")
        print(f"retrieved : {all_retrieved}")
        print(f"Expected : {tc['expected_retrievals']}")
        print(f"Recall Score : {recall_scores[-1]:.2f}")

        if all_retrieved:
            from chat import build_prompt
            from llm.client import chat

            messages = build_prompt(tc["query"])
            response = chat(messages)
            resp = faith_score(all_retrieved,response)
            faith_scores.append(resp["score"])
            print(f"Response : {response}")
            print(f"Faithfulness : {resp['score']:.2f} - {resp['reason']}")
        else:
            print("Faithfulness : skipped (nothing retrieved)")


    print("\n"+"="*60)
    print("Summary")
    print("="*60)
    avg_recall = sum(recall_scores)/len(recall_scores)
    print(f"Average recall score : {avg_recall:.2f}")
    if faith_scores:
        avg_faith = sum(faith_scores)/len(faith_scores)
        print(f"Average faithfulness score : {avg_faith:.2f}")
    print(f"Testcases run : {len(TEST_CASES)}")
    print("="*60)

if __name__ == "__main__":
    run_eval()