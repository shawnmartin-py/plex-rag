import time

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from qdrant_client.models import FieldCondition, Filter, MatchValue

from app.models.media_item import MediaItem
from app.services.vector_store import VectorStoreService

BASE_RETRY_DELAY = 10
MAX_RETRY_DELAY = 120
INTER_BATCH_DELAY = 4

SECTIONS = ["craft", "meaning", "context"]

_HUMAN = (
    "human",
    (
        "Title: {title} ({year})\n"
        "Genres: {genres}\n"
        "IMDb Rating: {imdb_rating}\n"
        "Content Rating: {content_rating}\n"
        "Synopsis: {synopsis}"
    ),
)

_CRAFT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a film expert generating a focused profile of a film's craft and identity "
                "for use in a semantic recommendation system.\n\n"
                "Write in dense, continuous prose covering:\n"
                "- Exact subgenre positioning with precision — not 'thriller' but 'paranoid Cold War "
                "conspiracy thriller' or 'slow-burn Scandinavian psychological horror'\n"
                "- The cinematic movement, tradition, or school it belongs to (French New Wave, "
                "Italian neorealism, New Hollywood, J-horror, Dogme 95, Ozploitation, etc.)\n"
                "- Country of origin and how it fits within that national cinema's history\n"
                "- The director's signature style, obsessions, and where this film sits in their career "
                "— debut, peak, late period, or departure\n"
                "- Which directors influenced them, and which directors they in turn influenced\n"
                "- If a known auteur, name their recurring thematic and visual preoccupations across "
                "their body of work\n"
                "- Visual grammar: camera movement (handheld, static, slow zoom), aspect ratio, "
                "depth of field, lighting philosophy (chiaroscuro, naturalistic, neon)\n"
                "- Color palette and what it communicates emotionally\n"
                "- Editing rhythm — fragmented and disorienting, languid and contemplative, or "
                "classically invisible\n"
                "- Score or soundtrack: composer, genre of music, how it functions emotionally\n"
                "- Cinematographer if notable; production design and costume as storytelling\n\n"
                "Be the expert recommender, not a Wikipedia editor. Every sentence should carry "
                "retrieval signal — specific names, subgenre labels, movement names, technique terms."
            ),
        ),
        _HUMAN,
    ]
)

_MEANING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a film expert generating a focused profile of a film's narrative, themes, "
                "and emotional experience for use in a semantic recommendation system.\n\n"
                "Write in dense, continuous prose covering:\n"
                "- How the story is told — linear or non-linear, unreliable narrator, multiple "
                "perspectives, found footage, epistolary, or other formal conceits\n"
                "- How much it withholds versus reveals, and where tension is generated: plot, "
                "character, atmosphere, or ideas\n"
                "- Core themes and recurring motifs — what questions it asks without necessarily "
                "answering\n"
                "- What the film is actually about beneath the surface plot: identity, mortality, "
                "power, grief, memory, capitalism, masculinity, colonialism, faith, etc.\n"
                "- Any literary, mythological, or philosophical traditions it draws from\n"
                "- Tone and emotional register with precision — numbing, exhilarating, suffocating, "
                "melancholy, darkly comedic, tender, alienating, or cathartic\n"
                "- Whether tone shifts across the film (comedy that turns brutal, horror that becomes "
                "tragic) and what the viewer carries out afterward\n"
                "- Acting style: naturalistic, theatrical, minimalist, Method, Brechtian — and how "
                "it serves the film\n"
                "- Ensemble dynamic, character archetypes or anti-archetypes present\n"
                "- How the film ends emotionally — cathartic, ambiguous, devastating, ironic — "
                "without revealing plot specifics\n\n"
                "Be the expert recommender, not a Wikipedia editor. Every sentence should carry "
                "retrieval signal — specific thematic keywords, tone descriptors, narrative terms."
            ),
        ),
        _HUMAN,
    ]
)

_CONTEXT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a film expert generating a focused profile of a film's cultural position, "
                "audience fit, and comparable films for use in a semantic recommendation system.\n\n"
                "Write in dense, continuous prose covering:\n"
                "- Why this film was made when it was — cultural anxieties, political events, or "
                "social movements it responds to, consciously or not\n"
                "- Initial critical reception versus retrospective reassessment — was it "
                "controversial, ahead of its time, or rediscovered later\n"
                "- Awards recognition, cult following, and cultural footprint — has it been remade, "
                "referenced, or parodied in ways that signal its reach\n"
                "- Who this film is for — be honest and specific about the viewer it rewards\n"
                "- What prior film experiences best prepare a viewer for it\n"
                "- Whether it demands patience or is immediately engaging; whether it improves on "
                "rewatch; whether it's best watched alone or with others\n"
                "- At least six films that share meaningful DNA, approached from multiple angles: "
                "same director's other work, same national cinema, same thematic obsession, same "
                "visual style, same emotional register, same cult audience. For each, name the "
                "specific axis of similarity — not just the title\n"
                "- What would surprise a first-time viewer who only knew the genre label\n"
                "- What makes this film unmistakable — the one thing it does that almost no other "
                "film does\n"
                "- End with a dense paragraph of retrieval-optimized descriptors: adjectives, genre "
                "micro-labels, thematic keywords, mood words, director names, actor names, "
                "cinematographer, composer, country, decade, movement names, and any other terms a "
                "knowledgeable person might use to find this film. Include synonyms and adjacent "
                "terms. This paragraph exists purely for search recall.\n\n"
                "Be the expert recommender, not a Wikipedia editor. Every sentence should carry "
                "retrieval signal."
            ),
        ),
        _HUMAN,
    ]
)

_PROMPTS = {"craft": _CRAFT_PROMPT, "meaning": _MEANING_PROMPT, "context": _CONTEXT_PROMPT}


class EnrichmentService:
    def __init__(self, llm: ChatGoogleGenerativeAI, vs_service: VectorStoreService, collection_name: str) -> None:
        self._chains = {section: prompt | llm | StrOutputParser() for section, prompt in _PROMPTS.items()}
        self._vs_service = vs_service
        self._collection_name = collection_name

    def _already_enriched(self, imdb_id: str, section: str) -> bool:
        results, _ = self._vs_service.client.scroll(
            collection_name=self._collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="metadata.imdb_id", match=MatchValue(value=imdb_id)),
                    FieldCondition(key="metadata.embedding_type", match=MatchValue(value="enriched")),
                    FieldCondition(key="metadata.section", match=MatchValue(value=section)),
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(results) > 0

    def _build_input(self, item: MediaItem, synopsis: str | None) -> dict:
        return {
            "title": item.title,
            "year": item.year,
            "genres": ", ".join(item.genres),
            "imdb_rating": item.imdb_rating,
            "content_rating": item.content_rating,
            "synopsis": synopsis or "(synopsis unavailable)",
        }

    def _generate_section(self, item: MediaItem, section: str) -> str | None:
        delay = BASE_RETRY_DELAY
        synopsis = item.synopsis
        while True:
            try:
                result = self._chains[section].invoke(self._build_input(item, synopsis))
                if not result.strip():
                    if synopsis:
                        # Synopsis may have triggered a content policy block — retry without it
                        print(f"  Content policy block for '{item.title}' ({section}) — retrying without synopsis")
                        synopsis = None
                        continue
                    print(f"  Warning: empty response for '{item.title}' ({section}) — section skipped")
                    return None
                return result
            except Exception as e:
                err = str(e)
                if (
                    "429" in err
                    or "RESOURCE_EXHAUSTED" in err
                    or "timeout" in err.lower()
                    or "deadline" in err.lower()
                    or "timed out" in err.lower()
                ):
                    print(f"  Rate limited or timed out, retrying in {delay}s...")
                    time.sleep(delay)
                    delay = min(delay * 2, MAX_RETRY_DELAY)
                else:
                    raise

    def build(self, items: list[MediaItem]) -> None:
        eligible = [item for item in items if item.synopsis]
        print(f"Processing enrichments for {len(eligible)} movies...")

        total_generated = 0
        total_skipped = 0
        total_blocked = 0

        for idx, item in enumerate(eligible, 1):
            new_docs = []
            pending = [s for s in SECTIONS if not self._already_enriched(item.imdb_id, s)]
            total_skipped += len(SECTIONS) - len(pending)

            if pending:
                print(f"  [{idx}/{len(eligible)}] {item.title}: generating {', '.join(pending)}...")
                for section in pending:
                    text = self._generate_section(item, section)
                    if text is not None:
                        new_docs.append(item.to_enriched_document(text, section))
                        total_generated += 1
                        print(f"    {section}: done ({len(text)} chars)")
                    else:
                        total_blocked += 1
                        print(f"    {section}: blocked")

            if new_docs:
                self._vs_service.add_documents_with_retry(new_docs)
                print(f"  [{idx}/{len(eligible)}] {item.title}: added {len(new_docs)} section(s)")
                if idx < len(eligible):
                    time.sleep(INTER_BATCH_DELAY)
            else:
                print(f"  [{idx}/{len(eligible)}] {item.title}: already complete")

        summary = f"Done. Generated {total_generated} section(s), skipped {total_skipped}."
        if total_blocked:
            summary += f" {total_blocked} section(s) blocked by safety filter."
        print(summary)
