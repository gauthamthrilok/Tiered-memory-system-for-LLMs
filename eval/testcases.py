TEST_CASES = [
    {
        "id": "tc_001",
        "description": "Basic identity retrieval",
        "seed_memories": [
            {"text": "User's name is Gautham.", "tier": "semantic"},
            {"text": "User is in semester 6 of CSE.", "tier": "semantic"},
            {"text": "User likes playing chess.", "tier": "semantic"},
        ],
        "query": "What is the user's name?",
        "expected_retrievals": ["User's name is Gautham."],
    },
    {
        "id": "tc_002",
        "description": "Project retrieval",
        "seed_memories": [
            {"text": "User is building a RAG based memory pipeline for LLMs.", "tier": "semantic"},
            {"text": "User uses Python and ChromaDB for their project.", "tier": "semantic"},
            {"text": "User's name is Gautham.", "tier": "semantic"},
        ],
        "query": "What project is the user working on?",
        "expected_retrievals": [
            "User is building a RAG based memory pipeline for LLMs.",
            "User uses Python and ChromaDB for their project.",
        ],
    },
    {
        "id": "tc_003",
        "description": "Procedural preference retrieval",
        "seed_memories": [
            {"text": "User prefers theory explained before code.", "tier": "procedural"},
            {"text": "User prefers concise responses.", "tier": "procedural"},
            {"text": "User is in semester 6 of CSE.", "tier": "semantic"},
        ],
        "query": "How does the user like explanations structured?",
        "expected_retrievals": [
            "User prefers theory explained before code.",
            "User prefers concise responses.",
        ],
    },
    {
        "id": "tc_004",
        "description": "Episodic retrieval",
        "seed_memories": [
            {"text": "User spent 3 hours debugging a segfault in their parser.", "tier": "episodic"},
            {"text": "User completed the TCP server phase of their KV store project.", "tier": "episodic"},
            {"text": "User's name is Gautham.", "tier": "semantic"},
        ],
        "query": "What has the user been working on recently?",
        "expected_retrievals": [
            "User spent 3 hours debugging a segfault in their parser.",
            "User completed the TCP server phase of their KV store project.",
        ],
    },
    {
        "id": "tc_005",
        "description": "Irrelevant query — low recall expected",
        "seed_memories": [
            {"text": "User's name is Gautham.", "tier": "semantic"},
            {"text": "User is in semester 6 of CSE.", "tier": "semantic"},
        ],
        "query": "What is the weather like today?",
        "expected_retrievals": [], 
    },
]