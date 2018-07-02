from watson import get_categories, get_concepts

text2 = 'robots with artificial intelligence will be able to execute difficult tasks without help'

categories = get_categories(text2)

print(categories)

concepts = get_concepts(text2)

print(concepts)

def write_to_file(text,rating):
    conc = ""
    for x in get_concepts(text):
	conc += x+"_"
    cat = ""
    for x in get_categories(text):
	cat += x+"_"
    res = text + ';' + str(conc) + ';' + str(cat)+";"+str(rating)
    with open('somefile.csv', 'a') as the_file:
        the_file.write(res + "\n")
    print res

write_to_file(text2,3.7)

