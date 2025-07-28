from typing import Optional
from govsight.config import load_settings
from govsight.memory import Memory
from govsight.retrieval.constraints import QueryConstraints
from govsight.memory.records import FactRecord

def get_structured_fact(
    constraints: QueryConstraints
) -> Optional[FactRecord]:
    if not (constraints.subject_name and constraints.attribute):
        return None

    # slugify
    if constraints.subject_type == "city" and constraints.state:
        slug = Memory.subject_slug_city(
            constraints.subject_name, constraints.state
        )
    else:
        slug = Memory.subject_slug_generic(constraints.subject_name)

    settings = load_settings(profile="dev")
    mem = Memory(settings)
    return mem.get_fact(slug, constraints.attribute)
