# Load necessary packages
from rdflib import Graph
import rdflib
import csv
from rdflib import BNode, URIRef, Namespace
from rdflib.namespace import RDF, RDFS, OWL, SKOS, DC
from rdflib.term import URIRef

# this takes over 30 minutes given how large the turtle file is
file_path = "/home/arbya/shared-folder/arby/wrk/lbs_jeff/nutritional_kgs/menu_guide.ttl"

# load graph
g = Graph()
g.parse(file_path, format = "turtle")

print(f"Graph has {len(g)} triples.") #It produces >25 million triplets

# Let's create the minimal viable product for creatinine. 
# The cells below really helped me hoan in on the relationships  

# First find URIs related to Creatinine 
# Get all entities labeled Creatinine
creatinine_uris = [
    s for s, p, o in g.triples((None, RDFS.label, None))
    if str(o).lower() == "creatinine"
]

print(f"Found {len(creatinine_uris)} URIs for 'Creatinine'")
print(creatinine_uris)

# Get the relevant triples 

creatinine_triples = set()

for uri in creatinine_uris:
    for triple in g.triples((uri, None, None)):
        creatinine_triples.add(triple)
    for triple in g.triples((None, None, uri)):
        creatinine_triples.add(triple)
        
print(f"Found {len(creatinine_triples)} triples for 'Creatinine'")
print(creatinine_triples)

# Convert these triples to human readable format, and you can notice the content, 
# measurement, and other nodes that make no sense for a RAG

def get_label(entity):
    label = g.value(subject = entity, predicate = RDFS.label)
    if label:
        return str(label)
    elif isinstance(entity, rdflib.term.BNode):
        return None #skip blank nodes like _:b0
    else:
        name = str(entity).split("/")[-1]
        if "reaction" in name.lower() or name.lower().startswith("obo"):
            return None # filter out low-informative terms
        return name
        
        

human_sentences = []
for s, p, o in creatinine_triples: 
    subj = get_label(s)
    pred = get_label(p)
    obj = get_label(o)
    if subj and pred and obj: # only include fully-readable triples
        sentence = f"{subj} - {pred} - {obj}"
        human_sentences.append(sentence)

# Preview 

for line in human_sentences[:10]:
    print(line)


# at least we now know which URIs are actually important and we know that the RDF labels 
# are not reliable and we need to figure out the connections without them 

def deep_enrich_entity(g, entity):
    from rdflib import URIRef, BNode

    if isinstance(entity, BNode):
        return None

    label = g.value(subject=entity, predicate=RDFS.label)
    if label and not any(x in str(label).lower() for x in ["content_", "measurement_", "reaction_"]):
        return str(label)

    uri_str = str(entity)

    # Content node
    if "content_" in uri_str:
        parts = []
        compound = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/hasCompound"))
        food = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/isContentOf"))
        amount = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/amount"))
        unit = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/unit"))
        pmid = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/hasReference"))

        if amount and unit:
            parts.append(f"{amount} {unit}")
        if compound:
            c_label = g.value(subject=compound, predicate=RDFS.label)
            if c_label:
                parts.append(str(c_label))
        if food:
            f_label = g.value(subject=food, predicate=RDFS.label)
            if f_label:
                parts.append(f"in {f_label}")
        if pmid:
            parts.append(f"(PMID:{pmid})")
        return " ".join(parts) if parts else uri_str.split("/")[-1]

    # Measurement node
    if "measurement_" in uri_str:
        parts = []
        amount = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/amount"))
        unit = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/unit"))
        sex = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/hasSex"))
        cohort = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/hasCohort"))
        sample = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/hasSampleType"))
        pmid = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/hasReference"))

        if amount and unit:
            parts.append(f"{amount} {unit}")
        if sex:
            parts.append(str(sex))
        if cohort:
            parts.append(f"({cohort})")
        if sample:
            sample_label = g.value(subject=sample, predicate=RDFS.label)
            if sample_label:
                parts.append(f"in {sample_label}")
        if pmid:
            parts.append(f"(PMID:{pmid})")

        return " ".join(parts) if parts else uri_str.split("/")[-1]

    # Generic fallback
    return uri_str.split("/")[-1].replace("_", " ")


# this is better but still not very informative
def extract_enriched_disease_compound_food_triples(g):
    results = []

    is_measurement_of = URIRef("http://MeNuGUIDE.local/isMeasurementOf")
    has_measurement = URIRef("http://MeNuGUIDE.local/hasMeasurement")
    has_compound = URIRef("http://purl.obolibrary.org/obo/FOBI_00423")
    is_compound_of = URIRef("http://MeNuGUIDE.local/isCompoundOf")
    is_content_of = URIRef("http://MeNuGUIDE.local/isContentOf")

    for measurement, _, compound in g.triples((None, has_compound, None)):
        # Step 1: Enrich measurement context
        disease = g.value(subject=measurement, predicate=is_measurement_of)
        disease_label = deep_enrich_entity(g, disease)
        compound_label = deep_enrich_entity(g, compound)

        m_amount = g.value(subject=measurement, predicate=URIRef("http://MeNuGUIDE.local/amount"))
        m_unit = g.value(subject=measurement, predicate=URIRef("http://MeNuGUIDE.local/unit"))
        m_sex = g.value(subject=measurement, predicate=URIRef("http://MeNuGUIDE.local/hasSex"))
        m_cohort = g.value(subject=measurement, predicate=URIRef("http://MeNuGUIDE.local/hasCohort"))
        m_sample = g.value(subject=measurement, predicate=URIRef("http://MeNuGUIDE.local/hasSampleType"))
        m_pmid = g.value(subject=measurement, predicate=URIRef("http://MeNuGUIDE.local/hasReference"))

        m_info = []
        if m_amount and m_unit:
            m_info.append(f"{m_amount} {m_unit}")
        if m_sex:
            m_info.append(str(m_sex))
        if m_cohort:
            m_info.append(f"({m_cohort})")
        if m_sample:
            label = g.value(subject=m_sample, predicate=RDFS.label)
            if label:
                m_info.append(f"in {label}")
        if m_pmid:
            m_info.append(f"({m_pmid})")

        m_summary = ", ".join(m_info)

        for _, _, content in g.triples((compound, is_compound_of, None)):
            food = g.value(subject=content, predicate=is_content_of)
            food_label = deep_enrich_entity(g, food)

            c_amount = g.value(subject=content, predicate=URIRef("http://MeNuGUIDE.local/amount"))
            c_unit = g.value(subject=content, predicate=URIRef("http://MeNuGUIDE.local/unit"))
            c_pmid = g.value(subject=content, predicate=URIRef("http://MeNuGUIDE.local/hasReference"))

            c_info = []
            if c_amount and c_unit:
                c_info.append(f"{c_amount} {c_unit}")
            if c_pmid:
                c_info.append(f"({c_pmid})")

            c_summary = ", ".join(c_info)

            if all([food_label, compound_label, disease_label]):
                sentence = (
                    f"{food_label} contains {compound_label}, "
                    f"which is a biomarker in {disease_label}"
                )
                results.append((food_label, compound_label, disease_label, sentence))
    return results
# take a couple of minutes
enriched_triples = extract_enriched_disease_compound_food_triples(g)
for t in enriched_triples[5:20]:
    print(t[3])

# Predicate URIs (confirmed from graph)
has_compound = URIRef("http://purl.obolibrary.org/obo/FOBI_00423")
is_measurement_of = URIRef("http://MeNuGUIDE.local/isMeasurementOf")
amount_pred = URIRef("http://MeNuGUIDE.local/amount")
unit_pred = URIRef("http://MeNuGUIDE.local/unit")
cohort_pred = URIRef("http://MeNuGUIDE.local/hasCohort")
sex_pred = URIRef("http://MeNuGUIDE.local/hasSex")
sample_pred = URIRef("http://MeNuGUIDE.local/hasSampleType")
ref_pred = URIRef("http://MeNuGUIDE.local/hasReference")

# Now we can add information about the amount for disease or normal, the sample type for that disease and compound 
def deep_enrich_entity_amount(g, entity):
    from rdflib import URIRef, BNode

    if entity is None or isinstance(entity, BNode):
        return None

    label = g.value(subject=entity, predicate=RDFS.label)
    if label and not any(x in str(label).lower() for x in ["content_", "measurement_", "reaction_"]):
        return str(label)

    uri_str = str(entity)

    # Content node
    if "content_" in uri_str:
        compound = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/hasCompound"))
        food = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/isContentOf"))
        amount = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/amount"))
        unit = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/unit"))
        pmid = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/hasReference"))

        parts = []
        if amount and unit:
            parts.append(f"{amount} {unit}")
        if compound:
            parts.append(str(g.value(subject=compound, predicate=RDFS.label)))
        if food:
            parts.append(f"in {g.value(subject=food, predicate=RDFS.label)}")
        if pmid:
            parts.append(f"(PMID:{pmid})")
        return " ".join(filter(None, parts))

    # Measurement node
    if "measurement_" in uri_str:
        amount = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/amount"))
        unit = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/unit"))
        sex = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/hasSex"))
        cohort = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/hasCohort"))
        sample = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/hasSampleType"))
        pmid = g.value(subject=entity, predicate=URIRef("http://MeNuGUIDE.local/hasReference"))

        parts = []
        if amount and unit:
            parts.append(f"{amount} {unit}")
        if sex:
            parts.append(str(sex))
        if cohort:
            parts.append(f"({cohort})")
        if sample:
            sample_label = g.value(subject=sample, predicate=RDFS.label)
            if sample_label:
                parts.append(f"in {sample_label}")
        if pmid:
            parts.append(f"(PMID:{pmid})")
        return " ".join(filter(None, parts))

    return uri_str.split("/")[-1].replace("_", " ")

from collections import defaultdict

compound_measurements = defaultdict(lambda: {"reference": [], "disease": []})

for measurement, _, compound in g.triples((None, has_compound, None)):
    # Get the disease or "normal" context
    disease_uri = g.value(subject=measurement, predicate=is_measurement_of)
    disease_label = deep_enrich_entity_amount(g, disease_uri)

    # Skip if no label at all
    if not disease_label:
        continue

    # Classify the measurement
    if "normal" in disease_label.lower():
        kind = "reference"
    else:
        kind = "disease"

    # Collect measurement data
    data = {
        "compound": deep_enrich_entity_amount(g, compound),
        "disease": disease_label,
        "amount": g.value(subject=measurement, predicate=amount_pred),
        "unit": g.value(subject=measurement, predicate=unit_pred),
        "cohort": g.value(subject=measurement, predicate=cohort_pred),
        "sex": g.value(subject=measurement, predicate=sex_pred),
        "sample": deep_enrich_entity_amount(g, g.value(subject=measurement, predicate=sample_pred)),
        "pmid": g.value(subject=measurement, predicate=ref_pred),
    }

    compound_key = deep_enrich_entity_amount(g, compound)
    compound_measurements[compound_key][kind].append(data)

for compound, groups in compound_measurements.items():
    print(f"Compound: {compound}")
    print(f"  Reference: {len(groups['reference'])} measurements")
    print(f"  Disease: {len(groups['disease'])} measurements")


# Now we have amount, let's figure out how to connect the disease to the amount, 
# the compound and the disease to figure out what range is normal and what range is diseased

has_measurement = URIRef("http://MeNuGUIDE.local/hasMeasurement")

print("Testing: disease -- hasMeasurement --> measurement")
count = 0
for disease, _, measurement in g.triples((None, has_measurement, None)):
    print(f"{disease} -- hasMeasurement --> {measurement}")
    count += 1
    if count >= 10:
        break

if count == 0:
    print("No hasMeasurement triples found. Check URI.")

# is measurement conntected to compound?

is_measurement_of = URIRef("http://MeNuGUIDE.local/isMeasurementOf")

for disease, _, measurement in g.triples((None, has_measurement, None)):
    for _, _, compound in g.triples((measurement, is_measurement_of, None)):
        print(f"{measurement} -- isMeasurementOf --> {compound}")
        break


# Keep digging
is_compound_of = URIRef("http://MeNuGUIDE.local/HasBiomarker")

for compound in set(o for _, _, o in g.triples((None, is_measurement_of, None))):
    for _, _, content in g.triples((compound, is_compound_of, None)):
        print(f"{compound} -- isCompoundOf --> {content}")
        break


import re

# Utility: normalize strings for fuzzy matching
def normalize(text):
    return re.sub(r"[^a-z0-9]", "", text.lower()) if text else ""

# Utility: cleanly join sex + cohort if present
def clean_person_description(cohort, sex):
    parts = []
    if cohort: parts.append(str(cohort))
    if sex: parts.append(str(sex))
    return " ".join(parts)

# Final result list
final_sentences = []

for food, compound, disease, base_sentence in enriched_triples:
    c_key = normalize(compound)
    d_key = normalize(disease)

    if c_key in compound_measurements:
        found = False
        for d_entry in compound_measurements[c_key]["disease"]:
            if normalize(d_entry["disease"]) == d_key:
                for r_entry in compound_measurements[c_key]["reference"]:
                    # Only match units that are present and equal
                    if d_entry["unit"] and r_entry["unit"] and str(d_entry["unit"]).lower() == str(r_entry["unit"]).lower():
                        person_disease = clean_person_description(d_entry["cohort"], d_entry["sex"])
                        person_ref = clean_person_description(r_entry["cohort"], r_entry["sex"])
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
        final_sentences.append(base_sentence.strip())  # keep base if no comparison found


final_sentences = list(set(final_sentences))

final_sentences = [
    s for s in final_sentences
    if "which is a biomarker in normal" not in s.lower()
]

final_sentences

# save this and try the AI hub assistant first since I still don't have access to an API key
with open("final_sentences.txt", "w", encoding="utf-8") as f:
    for sentence in final_sentences:
        f.write(sentence.strip() + "\n")

  
