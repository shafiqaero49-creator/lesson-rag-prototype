"""Small, offline retrieval proof of concept for the DBA assignment.

This demonstrates the RETRIEVAL stage only. A production design would replace
TF-IDF with Azure AI Search hybrid keyword + vector retrieval, then pass the
returned chunks to a deployed LLM for grounded generation.
"""
from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Chunk:
    chunk_id: str
    title: str
    source: str
    text: str


KB: List[Chunk] = [
    Chunk(
        "KB01-C01",
        "Lesson planning assumptions",
        "Illustrative school planning brief",
        "Example context: Grade 9 Earth Science, 28 students, mixed prior attainment, "
        "three 50-minute periods, projector, printed maps, paper strips, and no field trip. "
        "Curriculum-standard codes are intentionally left blank for the teacher to add "
        "from the school's approved curriculum.",
    ),
    Chunk(
        "KB02-C01",
        "Understanding plate motions",
        "USGS, This Dynamic Earth",
        "A transform boundary is a zone where two tectonic plates slide horizontally past "
        "one another. The San Andreas Fault zone in California is an example. Along the "
        "San Andreas system, the Pacific Plate moves northwest relative to the North "
        "American Plate. The fault zone is about 1,300 km long and movement averages about "
        "5 cm per year. Transform boundaries commonly have shallow earthquakes.",
    ),
    Chunk(
        "KB03-C01",
        "USGS plate-boundary teaching resource",
        "USGS Educational Resources",
        "The three main plate-boundary interactions are divergent, convergent, and "
        "transform. At transform boundaries plates slide past one another and crust is "
        "neither created nor destroyed. A paper model of the San Andreas Fault system can "
        "help learners model relative movement of the Pacific and North American plates. "
        "The USGS resource is intended for grades 4-12.",
    ),
    Chunk(
        "KB04-C01",
        "BSCS 5E instructional model",
        "Bybee et al. (2006), BSCS",
        "The BSCS 5E instructional model organises learning into five phases: engage, "
        "explore, explain, elaborate, and evaluate. Engage captures interest and surfaces "
        "prior ideas. Explore gives learners a shared hands-on experience before formal "
        "terms are introduced. Explain introduces scientific vocabulary and concepts. "
        "Elaborate applies understanding to a new situation. Evaluate checks understanding "
        "and lets students reflect on their learning.",
    ),
    Chunk(
        "KB05-C01",
        "Universal Design for Learning guidance",
        "CAST UDL Guidelines 3.0",
        "Universal Design for Learning (UDL) recommends multiple means of engagement, "
        "representation, and action and expression. To differentiate, present key "
        "vocabulary both visually and verbally, provide a glossary and labelled diagrams, "
        "and let learners show understanding orally, in writing, or by drawing. Offer "
        "choices that sustain interest and reduce barriers for learners who struggle with "
        "technical language.",
    ),
    Chunk(
        "KB06-C01",
        "Teacher assessment toolkit",
        "Illustrative teacher-authored planning template",
        "Use a diagnostic question before instruction, observe group model-building during "
        "exploration, ask students to explain their reasoning with evidence, and use an "
        "exit ticket to check each learner's understanding. Record common misconceptions "
        "and revise the next lesson. This is a planning template, not an externally "
        "validated test.",
    ),
]


def build_index(chunks: List[Chunk]):
    """Fit a TF-IDF vectoriser on title + text of every chunk."""
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([f"{c.title} {c.text}" for c in chunks])
    return vectorizer, matrix


def retrieve(query: str, vectorizer, matrix, chunks: List[Chunk], k: int = 3):
    """Return the top-k chunks ranked by cosine similarity to the query."""
    q_vec = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, matrix).ravel()
    ranked = sorted(zip(chunks, scores), key=lambda pair: pair[1], reverse=True)
    return ranked[:k]


TEST_QUERIES = {
    "Factual San Andreas query": (
        "What other interesting things can I say about the San Andreas Fault? Explain how "
        "the Pacific Plate and North American Plate move at this transform boundary, and "
        "suggest a Grade 9 teaching model and lesson activity."
    ),
    "Loose lesson-cycle paraphrase": (
        "How should I sequence the lesson so pupils try things themselves before I "
        "lecture?"
    ),
    "Expanded 5E query": (
        "Plan the lesson with the 5E stages: engage, explore, explain, elaborate and "
        "evaluate, including an exit ticket to check understanding."
    ),
    "UDL differentiation query": (
        "How can I differentiate using UDL for learners who struggle with technical "
        "vocabulary?"
    ),
}


if __name__ == "__main__":
    vectorizer, matrix = build_index(KB)
    for name, query in TEST_QUERIES.items():
        print("=" * 78)
        print(f"TEST: {name}")
        print(f"QUERY: {query}")
        print("TOP RETRIEVED CHUNKS (TF-IDF cosine similarity; illustrative ranking)")
        for rank, (chunk, score) in enumerate(retrieve(query, vectorizer, matrix, KB), 1):
            print(f"  {rank}. {chunk.chunk_id} | score={score:.3f} | {chunk.title} | {chunk.source}")
    print("=" * 78)
