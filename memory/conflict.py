from llm.client import chat

PROMPT = """You are a memory conflict resolution assistant.
You will be given a new fact and a list of existing memories that are semantically similar to it.

Your job is to classify the relationship between the new fact and the existing memories.

Classifications:
- CONTRADICTS: the new fact directly updates or contradicts one of the existing memories. 
  Example: existing = "User is in semester 6", new = "User is in semester 7"
- REDUNDANT: the new fact says essentially the same thing as an existing memory, just worded differently.
  Example: existing = "User's name is Gautham", new = "The user is called Gautham"
- NEW: the new fact adds information not present in any existing memory.
  Example: existing = "User's name is Gautham", new = "User uses Neovim as their editor"

Rules:
- Output ONLY a JSON object, nothing else, no markdown fences
- If CONTRADICTS, include the exact text of the memory being contradicted in "conflicting_memory"
- Be strict: only classify as CONTRADICTS if there is a clear factual conflict, not just topic overlap

Output format:
{"classification": "CONTRADICTS", "conflicting_memory": "<exact text of contradicted memory>"}
{"classification": "REDUNDANT"}
{"classification": "NEW"}
"""

def detect_conflict(new_fact : str, existing_mem : list[str]) -> dict:
    if not existing_mem:
        return {"classification" : "NEW"}

    memory_block = "\n".join(f"- {m}" for m in existing_mem)

    messages = [
        {"role" : "system","content" : PROMPT},
        {"role" : "user","content" : f"Existing memories : \n{memory_block}\n\nNew fact : {new_fact}"}
    ]

    raw = chat(messages)

    import json
    try:
        clean = raw.strip().replace("```json","").replace("```","").strip()
        return json.loads(clean)
    except Exception:
        return {"classification" : "NEW"}
