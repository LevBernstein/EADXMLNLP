import os
import random
import re
from typing import List, Tuple

import nltk
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder, QuadgramCollocationFinder
from nltk.corpus import stopwords
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
from nltk.metrics import BigramAssocMeasures, TrigramAssocMeasures, QuadgramAssocMeasures
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import requests
from bs4 import BeautifulSoup


sources = "LCA", "AFC", "AD", "EUR", "GC", "GMD", "HISP", "MSS_A", "MI", "MUS", "PP", "RBC", "RS", "VHP"
xmlDirectoryStructure = "./xmlFiles/"
txtDirectoryStructure = "./txtFiles/"
wordDumpPath = "./words.txt"
elements = ["scopecontent", "processinfo", "arrangement"]
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def bulkDownloadXML(sourceList: Tuple[str]) -> None:
	# Scrape EAD XML files from the library of congress website.
	resultList = []
	print("Fetching links to documents...")
	for source in sourceList:
		url = "https://findingaids.loc.gov/source/" + source
		soup = BeautifulSoup(requests.get(url).content.decode("utf-8"), features="lxml")
		subList = [j.attrs["href"] for j in [i.find_all("a")[0] for i in soup.body.find_all("em")]]
		resultList += subList
		print("Source", source, "yielded", len(subList), "results")

	for index, url in enumerate(resultList):
		print("Downloading document #" + str(index) + "...")
		r = requests.get(url).content
		with open(f"{xmlDirectoryStructure}{index}.xml", "wb") as f:
			f.write(r)


def scrapeKeyElements(filename: str, elements: List[str]) -> None:
	# Scrape text from selected elements and output it to individual text files,
	# for further analysis. Also assemble a master list of all significant words
	# used through the selected elements.
	print("Scraping", filename + "...")
	with open(f"{xmlDirectoryStructure}{filename}", "r") as f:
		soup = BeautifulSoup(f.read(), features="lxml")
		words = " ".join(
			" ".join([
				re.sub(
					' +', ' ', re.sub('\n+|\t+', ' ', i.p.getText().strip())
				) for i in soup.find_all(element) if i.p is not None
			]) for element in elements
		).casefold()
	print("Dumping", filename + "...")
	with open(f"{txtDirectoryStructure}{filename}".replace("xml", "txt"), "w+") as f:
		f.write(words)
	print("Adding", filename, "to master word list...")
	tokenizer = RegexpTokenizer(r'\w+')
	words = [word for word in tokenizer.tokenize(words) if word not in stop_words]
	with open(f"{wordDumpPath}", "a") as f:
		f.write(" ".join(words))


def getCollocations() -> None:
	collocationTuples = (
		(2, BigramCollocationFinder, BigramAssocMeasures().likelihood_ratio),
		(3, TrigramCollocationFinder, TrigramAssocMeasures().likelihood_ratio),
		(4, QuadgramCollocationFinder, QuadgramAssocMeasures().likelihood_ratio)
	)
	corpus = PlaintextCorpusReader(txtDirectoryStructure, ".*")
	text = nltk.Text([lemmatizer.lemmatize(word) for word in corpus.words()])
	# lemmatizer converts word forms into their base; for instance,
	# running -> run; books -> book; etc. This can be disabled if desired,
	# though results will contain a higher incidence of similar phrases.


	def applyFilter(finder: nltk.collocations.AbstractCollocationFinder) -> None:
		finder.apply_freq_filter(2)
		finder.apply_word_filter(lambda w: len(w) < 3 or w in stop_words or "draw" in w or not w.isalpha())
		# Stop word lambda also removes prepositions and other words that don't
		# impart a ton of semantic meaning. This is also optional.


	for triplet in collocationTuples:
		print(f"Most common {triplet[0]}-word phrases:")
		finder = triplet[1].from_words(text, triplet[0])
		applyFilter(finder)
		print(finder.nbest(triplet[2], 40))


if __name__ == "__main__":
	for directory in (xmlDirectoryStructure, txtDirectoryStructure):
		if not os.path.isdir(directory):
			os.mkdir(directory)

	bulkDownloadXML(sources)

	# Clean master world list
	with open(f"{wordDumpPath}", "w+") as f:
		f.write("")

	for f in os.listdir(xmlDirectoryStructure):
		scrapeKeyElements(f, elements)

	getCollocations()