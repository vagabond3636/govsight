import logging
from govsight.config.settings import settings
from govsight.memory import Memory
from govsight.db.search_local import search_local_facts
from govsight.parser.parser import parse_intent_and_facts
from govsight.web_reasoner import query_web_and_summarize
from rich import print
from rich.prompt import Prompt

memory = Memory(settings=settings)

def main():
    print("[bold green]🧠 GovSight Chat CLI (R0) – invoking legacy engine...[/bold green]")
    print("[yellow]Welcome to GovSight[/yellow]")
    print("[dim]Type 'exit' to quit.[/dim]\n")

    while True:
        user_input = Prompt.ask("[bold blue]You[/bold blue]")
        if user_input.lower() in ["exit", "quit"]:
            break

        try:
            fact = parse_intent_and_facts(user_input)
            subject = fact.get("subject")
            attribute = fact.get("attribute")
            value = fact.get("value")
            intent = fact.get("intent")

            if intent == "provide_fact" and subject and attribute and value:
                memory.store_fact(subject, attribute, value)
                print(f"[green]✅ Stored:[/green] {subject} – {attribute} – {value}")
                continue
            elif intent == "ask_question" and subject and attribute:
                answer = search_local_facts(settings.db_path, subject, attribute)
                if answer:
                    print(f"[green]GovSight (Memory):[/green] {answer}")
                    continue

            # If no memory hit or fallback, search the web
            print("[yellow][web_reasoner] Querying the web for:[/yellow]", user_input)
            summary = query_web_and_summarize(user_input)
            print(f"[green]GovSight (Web):[/green] {summary}")

        except Exception as e:
            logging.exception("Unexpected error")
            print(f"[red]❌ Error:[/red] {e}")

if __name__ == "__main__":
    main()
