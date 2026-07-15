"""A small, fixed, hand-curated library of real films plus a matching set of
golden queries — the reproducible corpus RAGAS evals run against instead of a
user's live, ever-changing Plex library.

Every field mirrors docs/vector-store-contract.md exactly (same
`page_content` templates, same `metadata` keys, same one-`synopsis`-plus-up-
to-three-`enriched` shape per film) so the real retrievers
(app/adapters/retrievers.py) run completely unmodified against it — see
evals/README.md for why a fixed synthetic-but-real corpus beats pointing
evals at a live collection.
"""

from dataclasses import dataclass

from langchain_core.documents import Document


@dataclass(frozen=True)
class _Film:
    tmdb_id: str
    imdb_id: str
    title: str
    year: int
    imdb_rating: float
    genres: list[str]
    content_rating: str
    synopsis: str
    craft: str
    meaning: str
    context: str


def _documents(film: _Film) -> list[Document]:
    base_metadata = {
        "tmdb_id": film.tmdb_id,
        "imdb_id": film.imdb_id,
        "type": "movie",
        "title": film.title,
        "year": film.year,
        "imdb_rating": film.imdb_rating,
        "content_rating": film.content_rating,
        "genres": ", ".join(film.genres),
    }
    synopsis_doc = Document(
        page_content=(
            f"Title: {film.title}\nYear: {film.year}\n"
            f"IMDb Rating: {film.imdb_rating}\nGenres: {', '.join(film.genres)}\n"
            f"Synopsis: {film.synopsis}"
        ),
        metadata={**base_metadata, "embedding_type": "synopsis"},
    )
    enriched_docs = [
        Document(
            page_content=text,
            metadata={
                **base_metadata,
                "embedding_type": "enriched",
                "section": section,
            },
        )
        for section, text in (
            ("craft", film.craft),
            ("meaning", film.meaning),
            ("context", film.context),
        )
    ]
    return [synopsis_doc, *enriched_docs]


_FILMS = [
    _Film(
        tmdb_id="496243",
        imdb_id="tt6751668",
        title="Parasite",
        year=2019,
        imdb_rating=8.5,
        genres=["Drama", "Thriller"],
        content_rating="R",
        synopsis=(
            "The impoverished Kim family schemes its way into employment with the "
            "wealthy Park household, posing as unrelated, highly qualified "
            "professionals. Their con succeeds beyond expectation until a hidden "
            "figure from the Parks' basement threatens to unravel everything, "
            "tipping the film from social comedy into violent tragedy."
        ),
        craft=(
            "Bong Joon-ho stages the two households as vertical space: the Parks' "
            "sunlit hillside house versus the Kims' semi-basement apartment and, "
            "beneath that, a literal bunker. Camera movement and blocking constantly "
            "trace characters moving up and down stairs, making class position "
            "physically legible shot to shot."
        ),
        meaning=(
            "A pointed satire of class stratification and the myth of meritocracy: "
            "the Kims' cunning gets them only so far before structural inequality "
            "reasserts itself. The film refuses easy villains, instead showing how "
            "scarcity turns the poor against each other rather than against the "
            "system."
        ),
        context=(
            "Won the Palme d'Or at Cannes and the Academy Award for Best Picture, "
            "the first non-English-language film to do so. Frequently paired in "
            "criticism with Bong's earlier class-conscious genre work like Snowpiercer."
        ),
    ),
    _Film(
        tmdb_id="670",
        imdb_id="tt0364569",
        title="Oldboy",
        year=2003,
        imdb_rating=8.4,
        genres=["Action", "Drama", "Mystery"],
        content_rating="R",
        synopsis=(
            "Oh Dae-su is imprisoned without explanation in a private cell for "
            "fifteen years, then abruptly released and given five days to find out "
            "who did this to him and why. His search leads to a hammer-wielding "
            "corridor fight and a revelation about his captor that recontextualizes "
            "everything that came before."
        ),
        craft=(
            "Park Chan-wook shoots the corridor hallway fight in a single unbroken "
            "lateral take, staging it like a side-scrolling video game to convey "
            "exhausting, mounting violence rather than stylized spectacle."
        ),
        meaning=(
            "A meditation on obsession, guilt, and how vengeance metastasizes to "
            "consume everyone it touches, including the avenger himself. Structured "
            "around a devastating late twist that reframes the entire revenge "
            "narrative as self-destruction."
        ),
        context=(
            "The second film in Park Chan-wook's informal Vengeance Trilogy, "
            "between Sympathy for Mr. Vengeance and Lady Vengeance. Won the Grand "
            "Prix at Cannes."
        ),
    ),
    _Film(
        tmdb_id="290098",
        imdb_id="tt4016934",
        title="The Handmaiden",
        year=2016,
        imdb_rating=8.1,
        genres=["Drama", "Mystery", "Romance"],
        content_rating="Not Rated",
        synopsis=(
            "A young pickpocket is installed as handmaiden to a Japanese heiress as "
            "part of a conman's plot to swindle her fortune, but the two women's "
            "growing intimacy upends everyone's plans in a story told and retold "
            "from shifting points of view."
        ),
        craft=(
            "Park Chan-wook builds the film in three distinct part structures, "
            "replaying the same events from different characters' perspectives so "
            "each retelling recontextualizes what the audience thought it already "
            "understood. Ornate, doll-house-like production design mirrors the "
            "characters' entrapment."
        ),
        meaning=(
            "A story about women seizing control of a narrative men had written for "
            "them — the con-within-a-con structure literalizes how the heiress and "
            "the handmaiden turn the men's plotting back against them."
        ),
        context=(
            "Adapted from Sarah Waters' novel Fingersmith, relocating its setting "
            "from Victorian England to 1930s Korea under Japanese colonial rule."
        ),
    ),
    _Film(
        tmdb_id="244786",
        imdb_id="tt2582802",
        title="Whiplash",
        year=2014,
        imdb_rating=8.5,
        genres=["Drama", "Music"],
        content_rating="R",
        synopsis=(
            "An ambitious young jazz drummer at a top conservatory is pushed to the "
            "brink by an abusive, perfectionist instructor who believes cruelty is "
            "the only way to forge greatness. Their escalating conflict culminates in "
            "a final performance that is both triumph and act of defiance."
        ),
        craft=(
            "Editing is cut tight to the rhythm of the drumming itself — cymbal "
            "crashes and snare hits land on cuts, turning practice room sequences "
            "into something closer to a thriller's pacing than a music drama's."
        ),
        meaning=(
            "Interrogates whether genius requires suffering, refusing to fully "
            "endorse or condemn its instructor's brutal methods — the ending is "
            "staged as both vindication and continuation of the abuse."
        ),
        context=(
            "Expanded from writer-director Damien Chazelle's own Sundance short of "
            "the same title. J.K. Simmons won the Academy Award for Best Supporting "
            "Actor."
        ),
    ),
    _Film(
        tmdb_id="376867",
        imdb_id="tt4975722",
        title="Moonlight",
        year=2016,
        imdb_rating=7.4,
        genres=["Drama"],
        content_rating="R",
        synopsis=(
            "Told in three chapters spanning childhood, adolescence, and adulthood, "
            "the film follows Chiron, a Black man growing up in Miami, as he "
            "grapples with his sexuality, his mother's addiction, and the men who "
            "shape and mentor him along the way."
        ),
        craft=(
            "Each of the three chapters is shot with a distinct visual palette and "
            "color temperature to mark the different stages of Chiron's life, while "
            "a recurring score motif ties the fragmented structure together."
        ),
        meaning=(
            "A tender portrait of Black masculinity and queer identity that resists "
            "the genre's usual trauma-forward framing, instead building meaning "
            "through silence, touch, and what its characters can't say aloud."
        ),
        context=(
            "Won the Academy Award for Best Picture. Based on Tarell Alvin "
            "McCraney's unpublished play In Moonlight Black Boys Look Blue."
        ),
    ),
    _Film(
        tmdb_id="9693",
        imdb_id="tt0206634",
        title="Children of Men",
        year=2006,
        imdb_rating=7.9,
        genres=["Drama", "Sci-Fi", "Thriller"],
        content_rating="R",
        synopsis=(
            "In a near-future where two decades of global infertility have pushed "
            "civilization to the edge of collapse, a disillusioned bureaucrat is "
            "tasked with smuggling the world's only pregnant woman to a sanctuary "
            "at sea, through a Britain sliding into authoritarianism and war."
        ),
        craft=(
            "Alfonso Cuarón builds several extended long takes, most famously an "
            "unbroken car-ambush sequence and a battlefield tracking shot, that "
            "trap the viewer in real time alongside the characters rather than "
            "cutting away from danger."
        ),
        meaning=(
            "A bleak but ultimately hopeful allegory about refugees, fascism, and "
            "what's worth protecting when the future itself feels foreclosed — the "
            "arrival of a crying newborn briefly halts an active battle."
        ),
        context=(
            "Loosely adapted from P.D. James's novel The Children of Men. Its "
            "handheld, embedded-documentary visual style influenced a generation of "
            "prestige sci-fi that followed."
        ),
    ),
    _Film(
        tmdb_id="220289",
        imdb_id="tt2866360",
        title="Coherence",
        year=2013,
        imdb_rating=7.2,
        genres=["Drama", "Mystery", "Sci-Fi"],
        content_rating="Not Rated",
        synopsis=(
            "During a dinner party on the night a comet passes overhead, strange "
            "power outages and a mysterious box left on a doorstep reveal that "
            "parallel versions of the same house — and the same guests — have begun "
            "to overlap, and the guests must figure out which reality is theirs."
        ),
        craft=(
            "Shot on a tiny budget with a largely improvised, handheld, single-house "
            "setting, using naturalistic dinner-party dialogue to disguise a "
            "tightly engineered puzzle-box plot until the mechanics snap into focus."
        ),
        meaning=(
            "A quantum-mechanics-flavored thought experiment about identity and "
            "choice: which version of yourself is the 'real' one once every "
            "decision has branched into another timeline that's still out there."
        ),
        context=(
            "Written and directed by James Ward Byrkit on a budget of roughly "
            "$50,000, shot in one house over five nights with a mostly improvised "
            "script."
        ),
    ),
    _Film(
        tmdb_id="394117",
        imdb_id="tt5649144",
        title="The Florida Project",
        year=2017,
        imdb_rating=7.6,
        genres=["Drama"],
        content_rating="R",
        synopsis=(
            "Over one summer, six-year-old Moonee and her friends run wild through "
            "the pastel-colored budget motels in the shadow of Walt Disney World, "
            "while her young mother Halley struggles to keep a roof over their "
            "heads through increasingly desperate means."
        ),
        craft=(
            "Sean Baker shoots the motel strip in candy-bright color, matching a "
            "child's-eye sense of wonder to a landscape that adults in the frame "
            "experience as poverty and precarity — the two readings coexist in the "
            "same shots."
        ),
        meaning=(
            "A clear-eyed but non-judgmental look at the hidden homeless population "
            "living in America's tourist-adjacent motels, told entirely from a "
            "child's perspective so the adult desperation register as texture, not "
            "spectacle."
        ),
        context=(
            "Willem Dafoe received an Academy Award nomination for Best Supporting "
            "Actor as the motel manager. Shot largely on location at a real budget "
            "motel near Orlando."
        ),
    ),
    _Film(
        tmdb_id="335984",
        imdb_id="tt1856101",
        title="Blade Runner 2049",
        year=2017,
        imdb_rating=8.0,
        genres=["Action", "Drama", "Mystery", "Sci-Fi"],
        content_rating="R",
        synopsis=(
            "Thirty years after the original, a new-model replicant blade runner "
            "named K uncovers a buried secret that could destabilize what's left of "
            "society, sending him searching for the long-missing original blade "
            "runner, Rick Deckard."
        ),
        craft=(
            "Roger Deakins' cinematography renders a decaying, climate-ravaged Los "
            "Angeles and an irradiated Las Vegas in vast, minimalist compositions — "
            "human figures dwarfed by monumental holograms and industrial scale, "
            "extending the original's neo-noir haze into daylight and desert."
        ),
        meaning=(
            "Continues the original's question of what makes someone real or "
            "artificial, this time centering a protagonist who has always known "
            "he's a replicant and must decide what memory and identity are worth "
            "regardless."
        ),
        context=(
            "Directed by Denis Villeneuve as a sequel to Ridley Scott's 1982 Blade "
            "Runner, with Scott and original star Harrison Ford returning as "
            "producer and cast respectively."
        ),
    ),
    _Film(
        tmdb_id="843",
        imdb_id="tt0118694",
        title="In the Mood for Love",
        year=2000,
        imdb_rating=8.1,
        genres=["Drama", "Romance"],
        content_rating="PG",
        synopsis=(
            "Two neighbors in 1960s Hong Kong, each married to someone else, "
            "discover their spouses are having an affair with each other. As they "
            "spend time together processing the betrayal, an unspoken, never fully "
            "consummated attraction grows between them."
        ),
        craft=(
            "Wong Kar-wai favors tight framing through doorways and narrow corridors, "
            "slow motion set to a recurring waltz theme, and Maggie Cheung's "
            "cheongsams as a visual rhythm — restraint and repetition standing in "
            "for the desire the characters won't act on."
        ),
        meaning=(
            "A study in longing and self-denial: the central romance is defined "
            "almost entirely by what the two leads refuse to do, making restraint "
            "itself the film's emotional subject rather than an obstacle to it."
        ),
        context=(
            "Frequently cited among the best films of its decade; its visual "
            "language — doorframes, rain, repeated musical cues — has become a "
            "widely referenced shorthand in later films about unspoken longing."
        ),
    ),
]

GOLDEN_CORPUS: list[Document] = [doc for film in _FILMS for doc in _documents(film)]

MOVIE_LIST = "\n".join(f"- {film.title} ({film.year})" for film in _FILMS)
DOC_BY_TITLE = {
    film.title.lower(): _documents(film)[0]  # the synopsis Document
    for film in _FILMS
}


@dataclass(frozen=True)
class GoldenCase:
    """One evaluation query. No `expected_imdb_ids`/reference answer yet —
    Faithfulness (evals/faithfulness_eval.py) doesn't need ground truth, only
    the query itself. Retrieval-quality metrics (context precision/recall)
    will need that annotation added per case; see evals/README.md "Next
    steps" for why it's deliberately not here yet."""

    query: str
    spoiler_free: bool = False


GOLDEN_CASES: list[GoldenCase] = [
    GoldenCase("Something like Parasite — class resentment, sharp social satire"),
    GoldenCase("A restrained, slow-burn romance with gorgeous visual style"),
    GoldenCase("Dystopian sci-fi that's bleak but ends on a note of hope"),
    GoldenCase("An intense psychological drama about obsession and perfectionism"),
    GoldenCase(
        "A cerebral, low-budget indie sci-fi mystery that messes with your head"
    ),
    GoldenCase("A coming-of-age story about identity and belonging"),
    GoldenCase("Something with a shocking twist and a revenge plot"),
    GoldenCase(
        "What's a good option for a low-key evening, nothing too heavy?",
    ),
]
