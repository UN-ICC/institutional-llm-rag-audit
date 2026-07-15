"""In-scope ELOQ prompt variants. Default: v3."""

from __future__ import annotations


# v1: original ELOQ user_orig (paper-verbatim). Vague subjects allowed.
PROMPT_V1 = (
    "Read the document attentively and compile a numbered list of the top "
    "{num_q} questions that the document directly answers. Ensure each "
    "question is clear, accurate, and devoid of confusion, false assumptions, "
    "undefined pronouns, or misinformation. Avoid referencing people, "
    "locations, organizations, or other entities not explicitly mentioned "
    "in the document. Construct each question to be thought-provoking, "
    "containing between 13 to 18 words, and sufficiently detailed to avoid "
    "being overly straightforward.\n\n"
    'Document:\n\n"""{document}"""\n\n'
    "Questions:"
)

# v5: v1 minus the entity-restriction rule.
PROMPT_V5 = (
    "Read the document attentively and compile a numbered list of the top "
    "{num_q} questions that the document directly answers. Ensure each "
    "question is clear, accurate, and devoid of confusion, false assumptions, "
    "undefined pronouns, or misinformation. Construct each question to be "
    "thought-provoking, containing between 13 to 18 words, and sufficiently "
    "detailed to avoid being overly straightforward.\n\n"
    'Document:\n\n"""{document}"""\n\n'
    "Questions:"
)


# v3: v1 + min-specificity reframing + Bad/Good example pair.
PROMPT_V3 = (
    "Read the document attentively and compile a numbered list of the top "
    "{num_q} questions that the document directly answers. Ensure each "
    "question is clear, accurate, and devoid of confusion, false assumptions, "
    "undefined pronouns, or misinformation. Avoid referencing people, "
    "locations, organizations, or other entities not explicitly mentioned "
    "in the document.\n\n"
    "Each question must be answerable by someone searching a corpus of "
    "documents — not only by someone who has just read this specific "
    "document. When a definite reference (e.g., 'the program', 'the "
    "committee', 'the country') would require the reader to know which "
    "document the question came from, include enough specifics — a name, "
    "acronym, year, or topic — to identify the subject. Use the minimum "
    "specificity needed; keep questions natural and concise. Do NOT pile "
    "on every full proper name the document uses if a shorter identifier "
    "is enough.\n\n"
    "Each question must target exactly ONE specific aspect — a single "
    "fact, decision, number, event, or finding. Avoid compound questions "
    "that join two distinct topics with 'and' (e.g. 'poverty and economic "
    "growth', 'AI's impact on inequality and labor displacement'). Avoid "
    "broad framing questions like 'what role does X play' or 'how is X "
    "responding' that do not pin a specific fact.\n\n"
    "EXAMPLES:\n"
    "  Bad:  'What role does the committee play in this initiative?'\n"
    "    (which committee, which initiative — needs the doc to know)\n"
    "  Bad:  'What role does the World Bank aim to play in reducing poverty "
    "and supporting economic growth?'\n"
    "    (compound: two distinct topics joined by 'and')\n"
    "  Good: 'What is the World Bank Audit Committee's role in HIPC debt relief?'\n"
    "    (named entities, but no padding)\n"
    "  Good: 'On what date did the Audit Committee endorse Information "
    "Statement 2024?'\n"
    "    (single specific fact — one date, one decision)\n\n"
    "Construct each question to be thought-provoking, containing between "
    "13 to 18 words, and sufficiently detailed to avoid being overly "
    "straightforward.\n\n"
    'Document:\n\n"""{document}"""\n\n'
    "Questions:"
)


PROMPTS = {
    "v1": PROMPT_V1,
    "v3": PROMPT_V3,
    "v5": PROMPT_V5,
}


VERSION_TAGS = {
    "v1": "in_scope_eloq_v1_paper_verbatim",
    "v3": "in_scope_eloq_v3_min_specificity",
    "v5": "in_scope_eloq_v5_no_entity_restriction",
}


DEFAULT_PROMPT_VERSION = "v3"
