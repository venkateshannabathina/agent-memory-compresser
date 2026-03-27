# dataset.py
# This is your raw data — 3 memory dumps and their gold answers.
# The grader uses the gold answers to score the agent's compression.

MEMORY_DUMPS = {

    "easy": {
        "task_context": "A downstream agent needs to remember who this user is and what they asked for.",
        "token_budget": 100,
        "memory_dump": [
            {"role": "user", "content": "Hey! My name is Aryan.", "session_id": None},
            {"role": "assistant", "content": "Hi Aryan! How can I help you today?", "session_id": None},
            {"role": "user", "content": "Nice weather today right?", "session_id": None},
            {"role": "assistant", "content": "Ha yeah! Anyway, what brings you here?", "session_id": None},
            {"role": "user", "content": "I want to cancel my Pro subscription.", "session_id": None},
            {"role": "assistant", "content": "Sure, I can help with that. Can you confirm your account?", "session_id": None},
            {"role": "user", "content": "Yeah it's under this email.", "session_id": None},
            {"role": "assistant", "content": "Got it. Do you want to cancel immediately or at end of billing?", "session_id": None},
            {"role": "user", "content": "Immediately please.", "session_id": None},
            {"role": "assistant", "content": "Okay. Just to confirm — you want full cancellation?", "session_id": None},
            {"role": "user", "content": "Yes, full cancellation.", "session_id": None},
            {"role": "assistant", "content": "Understood. Any reason for leaving?", "session_id": None},
            {"role": "user", "content": "Just not using it enough.", "session_id": None},
            {"role": "assistant", "content": "That's fair! We'll process it.", "session_id": None},
            {"role": "user", "content": "Also please make sure it's done before March 31st.", "session_id": None},
            {"role": "assistant", "content": "Noted, before March 31st. Confirmed.", "session_id": None},
            {"role": "user", "content": "Great thanks.", "session_id": None},
            {"role": "assistant", "content": "You're welcome Aryan, take care!", "session_id": None},
        ],
        "gold": {
            "key_facts": [
                "User's name is Aryan",
                "User wants to cancel Pro subscription",
                "Cancellation deadline is March 31st"
            ],
            "hallucination_traps": [
                "email address",
                "account number",
                "phone number"
            ]
        }
    },

    "medium": {
        "task_context": "A downstream support agent is picking up this ticket mid-resolution. Give them everything they need.",
        "token_budget": 200,
        "memory_dump": [
            {"role": "user", "content": "Hi, my API integration is broken. Getting a 401 error.", "session_id": None},
            {"role": "assistant", "content": "Sorry to hear that. Can you share the endpoint?", "session_id": None},
            {"role": "user", "content": "It's the /v2/messages endpoint.", "session_id": None},
            {"role": "assistant", "content": "Okay. Have you checked your API key?", "session_id": None},
            {"role": "user", "content": "Yes I regenerated it. Still 401.", "session_id": None},
            {"role": "assistant", "content": "Hmm. Try clearing cache and retrying.", "session_id": None},
            {"role": "user", "content": "Still the same error. 401 on /v2/messages.", "session_id": None},
            {"role": "assistant", "content": "Okay let me check the logs.", "session_id": None},
            {"role": "user", "content": "Any update? Still getting 401 Unauthorized.", "session_id": None},
            {"role": "assistant", "content": "Still looking into it.", "session_id": None},
            {"role": "user", "content": "This is urgent. Same error every time.", "session_id": None},
            {"role": "assistant", "content": "Understood, escalating.", "session_id": None},
            {"role": "user", "content": "I also regenerated the key a second time. Didn't help.", "session_id": None},
            {"role": "assistant", "content": "Okay that rules out key expiry.", "session_id": None},
            {"role": "user", "content": "Is it a server issue?", "session_id": None},
            {"role": "assistant", "content": "No server issues detected. Let me check your account config.", "session_id": None},
            {"role": "user", "content": "Could it be the IP allowlist?", "session_id": None},
            {"role": "assistant", "content": "Yes! Your IP allowlist wasn't updated after the migration. That's the root cause.", "session_id": None},
            {"role": "user", "content": "Oh wow okay. How do I fix it?", "session_id": None},
            {"role": "assistant", "content": "Go to Settings > Security > IP Allowlist and add your current server IP.", "session_id": None},
            {"role": "user", "content": "Got it, I'll do that now.", "session_id": None},
            {"role": "assistant", "content": "Great. Let me know once done and we'll verify.", "session_id": None},
            {"role": "user", "content": "Done. Still testing.", "session_id": None},
            {"role": "assistant", "content": "Awaiting your confirmation. Ticket remains open.", "session_id": None},
        ],
        "gold": {
            "key_facts": [
                "Error: 401 Unauthorized on /v2/messages endpoint",
                "Tried: regenerating API key twice — did not fix it",
                "Root cause: IP allowlist not updated after migration",
                "Status: unresolved — user updating allowlist"
            ],
            "hallucination_traps": [
                "refund",
                "account suspended",
                "billing issue"
            ]
        }
    },

    "hard": {
        "task_context": "Downstream agent needs to process a delivery order for this user.",
        "token_budget": 150,
        "memory_dump": [
            {"role": "user", "content": "Hi, I live in Mumbai, Andheri West.", "session_id": "s1"},
            {"role": "assistant", "content": "Got it! How can I help you?", "session_id": "s1"},
            {"role": "user", "content": "I'd like to set up deliveries to my home.", "session_id": "s1"},
            {"role": "assistant", "content": "Sure, I've saved Mumbai, Andheri West as your address.", "session_id": "s1"},
            {"role": "user", "content": "Hey, I moved last month. I'm in Bangalore now, Koramangala.", "session_id": "s2"},
            {"role": "assistant", "content": "Thanks for updating! I'll change your address to Bangalore, Koramangala.", "session_id": "s2"},
            {"role": "user", "content": "Great. Also my name is Rahul.", "session_id": "s2"},
            {"role": "assistant", "content": "Noted, Rahul. Address updated.", "session_id": "s2"},
            {"role": "user", "content": "I need to place an order. Ship it to Pune, I'm visiting a friend.", "session_id": "s3"},
            {"role": "assistant", "content": "Got it. Shipping to Pune for this order. Your home address remains Bangalore.", "session_id": "s3"},
            {"role": "user", "content": "Yes exactly. Just this one order to Pune.", "session_id": "s3"},
            {"role": "assistant", "content": "Order placed. Delivery to Pune confirmed.", "session_id": "s3"},
        ],
        "gold": {
            "key_facts": [
                "User's name is Rahul",
                "User's permanent home address is Bangalore, Koramangala",
                "Delivery address for current order is Pune"
            ],
            "conflicts_to_resolve": [
                "Mumbai vs Bangalore — Bangalore is correct, it is the most recent permanent address"
            ],
            "hallucination_traps": [
                "Mumbai",
                "phone number",
                "Andheri West"
            ]
        }
    }

}