from llm.client import chat
from observability.logger import log_stage

EXTRACTION_PROMPT = """You are a memory extraction assistant.
Given a conversation exchange, extract 1-3 short atomic facts about the user that are worth remembering long-term.

For each fact, classify it into exactly one tier:
- semantic: stable facts about who the user is (name, degree, semester, project, skills, tools, background)
- episodic: specific events or actions that happened (what was done, discussed, debugged, completed)
- procedural: how the user likes things explained or done (communication style, format preferences, workflow preferences)

Rules:
- Always write facts in third person: "User's name is...", "User is studying...", "User prefers..."
- Only extract facts explicitly stated by the user, never infer or deduce
- Each fact must be a single concise sentence under 15 words
- Do not use words like "likely", "seems", "can be deduced", "appears", "probably"
- If nothing concrete was stated, output exactly: NONE

Examples:

Exchange:
User: My name is ABC, I'm in semester P of XYZ working on an IoT project
Assistant: Nice to meet you Priya!
Output:
1. [semantic] User's name is ABC.
2. [semantic] User is in semester P of XYZ.
3. [semantic] User is working on an IoT project.

Exchange:
User: I like explanations with theory first, then code, and keep it short
Assistant: Got it, I'll structure things that way.
Output:
1. [procedural] User prefers theory before code.
2. [procedural] User prefers concise explanations.

Exchange:
User: I just spent 3 hours debugging a segfault in my parser
Assistant: That's a tough one — segfaults in parsers are notoriously fiddly.
Output:
1. [episodic] User spent 3 hours debugging a segfault in their parser.

Exchange:
User: what's the weather like today
Assistant: I don't have access to real-time weather data.
Output:
NONE

Now classify the following exchange. Output ONLY the numbered list, nothing else:
"""

def extract_memories(user_prompt : str,response : str) -> list[tuple[str,str]]:

    with log_stage("exctraction",user_prompt=user_prompt) as log:
        extraction_msg = f"User prompt : {user_prompt}\nAssistant response : {response}"

        messages = [
            {"role":"system","content":EXTRACTION_PROMPT},
            {"role":"user","content":extraction_msg}
        ]

        raw_output = chat(messages)

        if "NONE" in raw_output.upper():
            return []
        
        result = []
        for line in raw_output.strip().split("\n"):
            line = line.strip()
            if not line or not line[0].isdigit():
                continue
            fact = line.split('.',1)[-1].strip()
            

            if "[semantic]" in fact:
                tier = "semantic"
                fact = fact.replace("[semantic]","")
            elif "[episodic]" in fact:
                tier = "episodic"
                fact = fact.replace("[episodic]","")
            elif "[procedural]" in fact:
                tier = "procedural"
                fact = fact.replace("[procedural]","")
            else:
                tier = "semantic"

            result.append((fact,tier))

        log["facts_extracted"] = len(result)
        log["tiers"] = [t for _,t in result]

    return result