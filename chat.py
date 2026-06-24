from memory.store import store_to_memory, retrieve_memory
from memory.extractor import extract_memories
from llm.client import chat
from observability.logger import log_stage

def build_prompt(user_msg : str) -> list[dict]:
    memory = retrieve_memory(user_msg)

    semantic = memory.get("semantic",[])
    episodic = memory.get("episodic",[])
    proc = memory.get("procedural",[])

    memory_sections = []
    if semantic:
        section = "Context about the user :\n" + "\n".join(f"- {m}" for m in semantic)
        memory_sections.append(section)
    if episodic:
        section = "Context from recent conversations :\n" + "\n".join(f"- {m}" for m in episodic)
        memory_sections.append(section)
    if proc:
        section = "User preferences :\n" + "\n".join(f"- {m}" for m in proc)
        memory_sections.append(section)

    if memory_sections:
        memory_block = "\n\n".join(memory_sections)

        system_content = ("You are a helpful assistant with memory of the past conversations.\n\n"
                        "Relevant memories from the past conversations : \n"
                        f"{memory_block}\n\n"
                        "Use this naturally in your response if relevant."
                        "Do not explicitly say anything like \"according to my memory\".")
    else:
        system_content = ("You are a helpful assistant with memory of the past conversations.\n\n"
                        "No relevant memories available from the past conversation for this query."
                        "Do not explicitly say anything like \"according to my memory\".")

    return [
        {"role":"system","content":system_content},
        {"role":"user","content":user_msg}
    ]

def run():
    print("Enter 'quit' to exit\n\n")

    while True:
        user_ip = input("You : ").strip()

        if not user_ip:
            continue
        elif user_ip.lower() == "quit":
            break
        
        messages = build_prompt(user_ip)

        with log_stage("generation",user_prompt=user_ip) as log:
            response = chat(messages)
            log["response_length"] = len(response)
            
        print(f"\nAssistant : {response}")

        facts = extract_memories(user_ip,response)
        if facts:
            stored = skipped = 0
            for fact,tier in facts:
                if store_to_memory(fact,tier):
                    print(f"    -> [{tier}] {fact}")
                    stored += 1
                else:
                    print(f"    ->[skipped] ({tier}) {fact}")
                    skipped += 1
            print(f"    {stored} stored and {skipped} skipped.")
        else:
            print(f"    No new memories extracted")
if __name__ == "__main__":
    run()