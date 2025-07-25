import sys
import re
from govsight.config import load_settings
from govsight.memory import Memory
from govsight.parser import parse_fact_from_text
from govsight.retrieval.planner import retrieve

# Load settings and initialize memory
settings = load_settings(profile="dev")
mem = Memory(settings)
session_id = mem.start_session(profile=settings.profile)

print("\n🧠 GovSight Chat CLI (R2) — with retrieval cascade!")
print(f"(Memory session #{session_id} started)\n")

turn = 0
while True:
    try:
        user_input = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[exit]")
        break

    if not user_input:
        continue
    if user_input.lower() in {"exit", "quit"}:
        print("GovSight: Exiting session.")
        break

    # Log user input
    mem.log_message(session_id, "user", user_input, turn_index=turn)

    response = None

    # Step 1: Check for parsed fact (declarative or interrogative)
    fact = parse_fact_from_text(user_input)
    if isinstance(fact, dict) and fact.get("value") is not None:
        # Declarative assertion: store the fact
        subject_type = fact["subject_type"]
        subject_name = fact["subject_name"]
        state        = fact["state"]
        attr         = fact["attr"]
        value        = fact["value"]

        slug = mem.subject_slug_city(subject_name, state)
        mem.remember_fact(
            subject_type=subject_type,
            subject_slug=slug,
            attr=attr,
            value=value,
            source="user",
        )
        response = (
            f"Noted — updated {attr} of {subject_name.title()}, "
            f"{state.upper()} to {value.title().rstrip('.')}"
        )

    else:
        # Step 2: Either an interrogative fact-request or no fact parsed
        # Try memory recall first
        recalled = None
        if isinstance(fact, dict) and fact.get("value") is None:
            # It's a question parse (e.g. "What is the population of X?")
            recalled = mem.recall_fact_from_text(user_input)
        else:
            # No parse or irrelevant parse, still try recall
            recalled = mem.recall_fact_from_text(user_input)

        if recalled:
            # Format memory recall into a sentence
            slug_parts = recalled.subject_slug.split("_")
            if slug_parts[-1].isalpha() and len(slug_parts[-1]) == 2:
                city = " ".join(slug_parts[:-1]).title()
                st   = slug_parts[-1].upper()
            else:
                city = " ".join(slug_parts).title()
                st   = ""
            attr = recalled.attr.replace("_", " ")
            val  = recalled.value.rstrip(".").title()
            if st:
                response = f"{val} is the {attr} of {city}, {st}."
            else:
                response = f"{val} is the {attr} of {city}."

        else:
            # Step 3: Full retrieval cascade (DB -> Pinecone -> Web)
            fetched = retrieve(user_input)
            if fetched:
                slug_parts = fetched.subject_slug.split("_")
                if slug_parts[-1].isalpha() and len(slug_parts[-1]) == 2:
                    city = " ".join(slug_parts[:-1]).title()
                    st   = slug_parts[-1].upper()
                else:
                    city = " ".join(slug_parts).title()
                    st   = ""
                attr = fetched.attr.replace("_", " ")
                val  = fetched.value.rstrip(".").title()
                if st:
                    response = f"{val} is the {attr} of {city}, {st}."
                else:
                    response = f"{val} is the {attr} of {city}."
            else:
                # Final fallback echo
                response = f"[echo] You said: {user_input}"

    # Print and log assistant response
    print(f"GovSight: {response}")
    mem.log_message(session_id, "assistant", response, turn_index=turn + 1)
    turn += 2
