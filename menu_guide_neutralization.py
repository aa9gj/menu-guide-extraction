"""
MeNu Guide Knowledge Graph Extraction and Transformation

This module extracts food-compound-disease relationships from the MeNu GUIDE
knowledge graph (RDF/Turtle format) and transforms them into human-readable
sentences suitable for RAG applications.

Usage:
    export MENU_GUIDE_PATH="/path/to/menu_guide.ttl"
    python menu_guide_neutralization.py
"""

import os
import re
from collections import defaultdict

from rdflib import Graph, BNode, URIRef
from rdflib.namespace import RDF, RDFS, OWL, SKOS, DC

# =============================================================================
# Configuration
# =============================================================================

# Path to MeNu GUIDE turtle file (set via environment variable or default)
FILE_PATH = os.environ.get(
    "MENU_GUIDE_PATH",
    "/home/arbya/shared-folder/arby/wrk/lbs_jeff/nutritional_kgs/menu_guide.ttl"
)

# Output file for extracted sentences
OUTPUT_FILE = "final_sentences.txt"

# URI Predicates used in MeNu GUIDE
PREDICATES = {
    "has_compound": URIRef("http://purl.obolibrary.org/obo/FOBI_00423"),
    "is_measurement_of": URIRef("http://MeNuGUIDE.local/isMeasurementOf"),
    "has_measurement": URIRef("http://MeNuGUIDE.local/hasMeasurement"),
    "is_compound_of": URIRef("http://MeNuGUIDE.local/isCompoundOf"),
    "is_content_of": URIRef("http://MeNuGUIDE.local/isContentOf"),
    "has_biomarker": URIRef("http://MeNuGUIDE.local/HasBiomarker"),
    "amount": URIRef("http://MeNuGUIDE.local/amount"),
    "unit": URIRef("http://MeNuGUIDE.local/unit"),
    "cohort": URIRef("http://MeNuGUIDE.local/hasCohort"),
    "sex": URIRef("http://MeNuGUIDE.local/hasSex"),
    "sample_type": URIRef("http://MeNuGUIDE.local/hasSampleType"),
    "reference": URIRef("http://MeNuGUIDE.local/hasReference"),
    "has_compound_local": URIRef("http://MeNuGUIDE.local/hasCompound"),
}

# Terms to filter out from labels
FILTERED_TERMS = ["content_", "measurement_", "reaction_"]


# =============================================================================
# Utility Functions
# =============================================================================

def normalize(text):
    """
    Normalize text for fuzzy matching by removing non-alphanumeric characters.

    Args:
        text: Input string to normalize

    Returns:
        Lowercase string with only alphanumeric characters
    """
    return re.sub(r"[^a-z0-9]", "", text.lower()) if text else ""


def clean_person_description(cohort, sex):
    """
    Create a clean description of a person from cohort and sex data.

    Args:
        cohort: Cohort information (e.g., "adults", "children")
        sex: Sex information (e.g., "male", "female")

    Returns:
        Space-separated string combining cohort and sex if present
    """
    parts = []
    if cohort:
        parts.append(str(cohort))
    if sex:
        parts.append(str(sex))
    return " ".join(parts)


def get_label(graph, entity):
    """
    Get a human-readable label for an RDF entity.

    Attempts to retrieve the RDFS label, falling back to extracting
    the last part of the URI. Filters out uninformative terms.

    Args:
        graph: RDFLib Graph object
        entity: RDF entity (URIRef or BNode)

    Returns:
        Human-readable string label, or None if entity should be skipped
    """
    label = graph.value(subject=entity, predicate=RDFS.label)
    if label:
        return str(label)
    elif isinstance(entity, BNode):
        return None  # Skip blank nodes
    else:
        name = str(entity).split("/")[-1]
        if "reaction" in name.lower() or name.lower().startswith("obo"):
            return None  # Filter out low-informative terms
        return name


def deep_enrich_entity(graph, entity, include_amounts=True):
    """
    Convert an RDF entity to an enriched human-readable format.

    Handles content nodes, measurement nodes, and generic entities,
    extracting relevant contextual information like amounts, units,
    cohorts, and references.

    Args:
        graph: RDFLib Graph object
        entity: RDF entity to enrich
        include_amounts: Whether to include amount/unit information

    Returns:
        Enriched string representation, or None if entity should be skipped
    """
    if entity is None or isinstance(entity, BNode):
        return None

    label = graph.value(subject=entity, predicate=RDFS.label)
    if label and not any(x in str(label).lower() for x in FILTERED_TERMS):
        return str(label)

    uri_str = str(entity)

    # Handle content nodes (food-compound relationships)
    if "content_" in uri_str:
        parts = []
        compound = graph.value(subject=entity, predicate=PREDICATES["has_compound_local"])
        food = graph.value(subject=entity, predicate=PREDICATES["is_content_of"])
        amount = graph.value(subject=entity, predicate=PREDICATES["amount"])
        unit = graph.value(subject=entity, predicate=PREDICATES["unit"])
        pmid = graph.value(subject=entity, predicate=PREDICATES["reference"])

        if include_amounts and amount and unit:
            parts.append(f"{amount} {unit}")
        if compound:
            c_label = graph.value(subject=compound, predicate=RDFS.label)
            if c_label:
                parts.append(str(c_label))
        if food:
            f_label = graph.value(subject=food, predicate=RDFS.label)
            if f_label:
                parts.append(f"in {f_label}")
        if pmid:
            parts.append(f"(PMID:{pmid})")
        return " ".join(filter(None, parts)) if parts else uri_str.split("/")[-1]

    # Handle measurement nodes (compound-disease relationships)
    if "measurement_" in uri_str:
        parts = []
        amount = graph.value(subject=entity, predicate=PREDICATES["amount"])
        unit = graph.value(subject=entity, predicate=PREDICATES["unit"])
        sex = graph.value(subject=entity, predicate=PREDICATES["sex"])
        cohort = graph.value(subject=entity, predicate=PREDICATES["cohort"])
        sample = graph.value(subject=entity, predicate=PREDICATES["sample_type"])
        pmid = graph.value(subject=entity, predicate=PREDICATES["reference"])

        if include_amounts and amount and unit:
            parts.append(f"{amount} {unit}")
        if sex:
            parts.append(str(sex))
        if cohort:
            parts.append(f"({cohort})")
        if sample:
            sample_label = graph.value(subject=sample, predicate=RDFS.label)
            if sample_label:
                parts.append(f"in {sample_label}")
        if pmid:
            parts.append(f"(PMID:{pmid})")
        return " ".join(filter(None, parts)) if parts else uri_str.split("/")[-1]

    # Generic fallback
    return uri_str.split("/")[-1].replace("_", " ")


# =============================================================================
# Extraction Functions
# =============================================================================

def find_compound_uris(graph, compound_name):
    """
    Find all URIs in the graph that match a given compound name.

    Args:
        graph: RDFLib Graph object
        compound_name: Name of the compound to search for (case-insensitive)

    Returns:
        List of URIRefs matching the compound name
    """
    return [
        s for s, p, o in graph.triples((None, RDFS.label, None))
        if str(o).lower() == compound_name.lower()
    ]


def get_compound_triples(graph, compound_uris):
    """
    Get all triples related to a list of compound URIs.

    Args:
        graph: RDFLib Graph object
        compound_uris: List of compound URIRefs

    Returns:
        Set of triples (subject, predicate, object) related to the compounds
    """
    triples = set()
    for uri in compound_uris:
        for triple in graph.triples((uri, None, None)):
            triples.add(triple)
        for triple in graph.triples((None, None, uri)):
            triples.add(triple)
    return triples


def extract_enriched_disease_compound_food_triples(graph):
    """
    Extract food-compound-disease relationships with enriched context.

    Traverses the knowledge graph to find connections between:
    - Foods (what contains the compound)
    - Compounds (the biomarker)
    - Diseases (where the compound is a biomarker)

    Args:
        graph: RDFLib Graph object

    Returns:
        List of tuples (food_label, compound_label, disease_label, sentence)
    """
    results = []

    for measurement, _, compound in graph.triples((None, PREDICATES["has_compound"], None)):
        # Get disease context from measurement
        disease = graph.value(subject=measurement, predicate=PREDICATES["is_measurement_of"])
        disease_label = deep_enrich_entity(graph, disease)
        compound_label = deep_enrich_entity(graph, compound)

        # Find foods containing this compound
        for _, _, content in graph.triples((compound, PREDICATES["is_compound_of"], None)):
            food = graph.value(subject=content, predicate=PREDICATES["is_content_of"])
            food_label = deep_enrich_entity(graph, food)

            if all([food_label, compound_label, disease_label]):
                sentence = (
                    f"{food_label} contains {compound_label}, "
                    f"which is a biomarker in {disease_label}"
                )
                results.append((food_label, compound_label, disease_label, sentence))

    return results


def build_compound_measurements(graph):
    """
    Build a dictionary of compound measurements grouped by reference/disease status.

    Args:
        graph: RDFLib Graph object

    Returns:
        Dictionary mapping compound names to their reference and disease measurements
    """
    compound_measurements = defaultdict(lambda: {"reference": [], "disease": []})

    for measurement, _, compound in graph.triples((None, PREDICATES["has_compound"], None)):
        disease_uri = graph.value(subject=measurement, predicate=PREDICATES["is_measurement_of"])
        disease_label = deep_enrich_entity(graph, disease_uri)

        if not disease_label:
            continue

        # Classify as reference (normal) or disease measurement
        kind = "reference" if "normal" in disease_label.lower() else "disease"

        data = {
            "compound": deep_enrich_entity(graph, compound),
            "disease": disease_label,
            "amount": graph.value(subject=measurement, predicate=PREDICATES["amount"]),
            "unit": graph.value(subject=measurement, predicate=PREDICATES["unit"]),
            "cohort": graph.value(subject=measurement, predicate=PREDICATES["cohort"]),
            "sex": graph.value(subject=measurement, predicate=PREDICATES["sex"]),
            "sample": deep_enrich_entity(
                graph,
                graph.value(subject=measurement, predicate=PREDICATES["sample_type"])
            ),
            "pmid": graph.value(subject=measurement, predicate=PREDICATES["reference"]),
        }

        compound_key = deep_enrich_entity(graph, compound)
        if compound_key:
            compound_measurements[normalize(compound_key)][kind].append(data)

    return compound_measurements


def create_comparison_sentences(enriched_triples, compound_measurements):
    """
    Create sentences comparing disease and reference measurements.

    Args:
        enriched_triples: List of (food, compound, disease, sentence) tuples
        compound_measurements: Dictionary of compound measurements

    Returns:
        List of final sentences with comparison information where available
    """
    final_sentences = []

    for food, compound, disease, base_sentence in enriched_triples:
        c_key = normalize(compound)
        d_key = normalize(disease)

        found = False
        if c_key in compound_measurements:
            for d_entry in compound_measurements[c_key]["disease"]:
                if normalize(d_entry["disease"]) == d_key:
                    for r_entry in compound_measurements[c_key]["reference"]:
                        # Match by unit
                        if (d_entry["unit"] and r_entry["unit"] and
                            str(d_entry["unit"]).lower() == str(r_entry["unit"]).lower()):
                            person_disease = clean_person_description(
                                d_entry["cohort"], d_entry["sex"]
                            )
                            person_ref = clean_person_description(
                                r_entry["cohort"], r_entry["sex"]
                            )
                            comparison = (
                                f"{compound} was measured at {d_entry['amount']} {d_entry['unit']} "
                                f"in {person_disease} with {d_entry['disease']}, "
                                f"compared to {r_entry['amount']} {r_entry['unit']} in healthy {person_ref}."
                            )
                            final_sentences.append(base_sentence.strip() + ". " + comparison.strip())
                            found = True
                            break
                if found:
                    break

        if not found:
            final_sentences.append(base_sentence.strip())

    return final_sentences


def filter_sentences(sentences):
    """
    Filter and deduplicate sentences, removing uninformative entries.

    Args:
        sentences: List of sentences to filter

    Returns:
        Filtered and deduplicated list of sentences
    """
    # Remove duplicates
    sentences = list(set(sentences))

    # Filter out sentences about "normal" as a disease (not informative)
    sentences = [
        s for s in sentences
        if "which is a biomarker in normal" not in s.lower()
    ]

    return sentences


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main execution function."""
    print(f"Loading knowledge graph from: {FILE_PATH}")
    print("Note: This may take 30+ minutes for large turtle files...")

    # Load the RDF graph
    graph = Graph()
    try:
        graph.parse(FILE_PATH, format="turtle")
    except FileNotFoundError:
        print(f"Error: File not found at {FILE_PATH}")
        print("Please set MENU_GUIDE_PATH environment variable to the correct path.")
        return
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    print(f"Graph loaded with {len(graph)} triples.")

    # Example: Find creatinine URIs for testing
    creatinine_uris = find_compound_uris(graph, "creatinine")
    print(f"Found {len(creatinine_uris)} URIs for 'Creatinine'")

    # Extract enriched triples
    print("Extracting food-compound-disease relationships...")
    enriched_triples = extract_enriched_disease_compound_food_triples(graph)
    print(f"Found {len(enriched_triples)} relationships")

    # Build compound measurements for comparison
    print("Building compound measurement database...")
    compound_measurements = build_compound_measurements(graph)

    # Create final sentences with comparisons
    print("Creating comparison sentences...")
    final_sentences = create_comparison_sentences(enriched_triples, compound_measurements)

    # Filter and clean
    final_sentences = filter_sentences(final_sentences)
    print(f"Final sentence count: {len(final_sentences)}")

    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for sentence in final_sentences:
            f.write(sentence.strip() + "\n")

    print(f"Saved {len(final_sentences)} sentences to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
