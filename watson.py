from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import CategoriesOptions, ConceptsOptions, Features

natural_language_understanding = NaturalLanguageUnderstandingV1(
        version='2017-02-27',
        username='96c83f46-c03c-4efb-bb6c-c9bbc5b5adc0',
        password='TLe8OPv0T7qK')
print('watson connection : done')


def get_categories(text):
    try:
	    response = natural_language_understanding.analyze(
		    text=text,
		    features=Features(
		            categories=CategoriesOptions()))

	    categories = []
	    for category in response["categories"]:
		categories.append(category["label"])

	    return categories
    except:
	return []


def get_concepts(text):
    try:
	    response2 = natural_language_understanding.analyze(
		    text=text,
		    features=Features(
		            concepts=ConceptsOptions(
		                    limit=3)))
	    concepts = []
	    for concept in response2["concepts"]:
		concepts.append(concept["text"])

	    return concepts
    except:
	return []
