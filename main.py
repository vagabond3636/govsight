import argparse

AGENTS = {
    "bill_resolver": "agents.bill_resolver",
    "tagger": "agents.tagger_agent",
    "vote_tracker": "ingest.vote_tracker_agent",
    "press": "ingest.press_agent",
    "news": "ingest.news_agent",
    "documents": "ingest.ingest_documents",
    "comm": "ingest.communication_agent",
    "bulk_bills": "ingest.congress_bulk_ingest"
}

def run_agent(name):
    import importlib
    if name in AGENTS:
        module = importlib.import_module(AGENTS[name])
        if hasattr(module, "main"):
            module.main()
        else:
            print("⚠️ No main() function in that module.")
    else:
        print("❌ Agent not found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", help="Run specific agent by name")
    args = parser.parse_args()

    if args.agent:
        run_agent(args.agent)
    else:
        print("✅ Available agents:")
        for key in AGENTS:
            print(f" - {key}")