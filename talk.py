import os
import sys

# ✅ Safe import for readline on Windows
try:
    import readline
except ImportError:
    pass  # 'readline' is not available on Windows

from rich import print
from govsight.memory import Memory
from govsight.vector.search import search_pinecone
from govsight.web_reasoner import query_web_and_summarize
from govsight.db.core import search_local_facts
from govsight.parser import parse_intent_and_facts
from govsight.config import settings

# Initialize memory object
memory = Memory(settings=settings)

def print_welcome():
    print("[bold green]Welcome to GovSight[/bold green]")
    print("Type 'exit' to quit.\n")

def main():
    print_welcome()

    while True:
        try:
            user_input = input("[bold blue]You:[/bold blue] ").strip()
            if user_input.lower() in ["exit", "quit"]:
                print("[bold yellow]Goodbye![/bold yellow]")
                break

            # Check local database first
            try:
                answer = memory.search(user_input)
                if answer:
                    print(f"[bold green]GovSight (Local):[/bold green] {answer}")
                    continue
            except Exception as e:
                print(f"[Fact Search Error] {e}")

            # Check Pinecone (vector search)
            try:
                pinecone_answer = search_pinecone(user_input)
                if pinecone_answer:
                    print(f"[bold green]GovSight (Pinecone):[/bold green] {pinecone_answer}")
                    continue
            except Exception as e:
                print(f"[Embedding Error] {e}")

            # Final fallback: Web search
            try:
                print(f"[web_reasoner] Querying the web for: {user_input}")
                web_answer = query_web_and_summarize(user_input)
                print(f"[bold green]GovSight (Web):[/bold green] {web_answer}")
            except Exception as e:
                print(f"GovSight (Web):  Error during web query: {e}")

        except KeyboardInterrupt:
            print("\n[bold yellow]Session interrupted. Exiting.[/bold yellow]")
            sys.exit(0)
        except EOFError:
            print("\n[bold yellow]Session ended.[/bold yellow]")
            break

if __name__ == "__main__":
    main()
