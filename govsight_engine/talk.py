# talk.py

from govsight.parser.parser import parse_intent_and_facts
from govsight.memory.memory import Memory
from govsight.web_reasoner.web_reasoner import web_search_and_summarize
from govsight.config.settings import settings


memory = Memory(settings)

def main():
    print("🧠 GovSight Chat CLI (R0) – invoking legacy engine...")
    print("Welcome to GovSight\nType 'exit' to quit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            break

        print(f"🧠 Parsing input: {user_input}")
        try:
            intent, facts = parse_intent_and_facts(user_input)

            for fact in facts:
                subject = fact.get("subject")
                attribute = fact.get("attribute")
                value = fact.get("value")

                print(f"🔍 Intent: {intent} | Subject: {subject} | Attribute: {attribute} | Value: {value}")

                if intent == "state_fact" and subject and attribute and value:
                    memory.store_fact(subject, attribute, value)
                    print(f"✅ Stored: {subject} – {attribute} – {value}")

                elif intent == "ask_question" and subject and attribute:
                    # Step 1: Try memory
                    print("🔎 Looking up in local memory...")
                    answer = memory.lookup_fact(subject, attribute)
                    if answer:
                        print(f"GovSight (Memory): {answer}")
                        continue

                    # Step 2: Skip Pinecone, go directly to web
                    print("🌐 Not found in memory. Falling back to web...")

                    # Step 3: Build context from memory
                    context_facts = memory.search(subject)
                    context = ". ".join([f"{s} {a} is {v}" for s, a, v in context_facts]) if context_facts else None

                    # Step 4: Web search
                    result = web_search_and_summarize(query=user_input, context=context)
                    print(f"GovSight (Web): {result}")

                else:
                    print("🤖 Please state a fact or ask a question.")
        except Exception as e:
            import traceback
            print(f"❌ Error: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    main()
