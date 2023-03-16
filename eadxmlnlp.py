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
from git import Remote, Repo


LOCSources = "LCA", "AFC", "AD", "EUR", "GC", "GMD", "HISP", "MSS_A", "MI", "MUS", "PP", "RBC", "RS", "VHP"
GitHubSources = (
	("https://github.com/NYULibraries/findingaids_eads.git", "./NYU"),
	("https://github.com/RockefellerArchiveCenter/data.git", "./Rockefeller"),
	("https://github.com/HeardLibrary/finding-aids.git", "./Heard")
)
LOCDirectory = "./LOC/"
txtDirectoryStructure = "./txtFiles/"
elements = ["scopecontent", "processinfo", "arrangement"]
ignoredWords = {"draw", "drawing"}
stopWords = set(stopwords.words("english")).union(ignoredWords)
lemmatizer = WordNetLemmatizer()
textFilePos = 0


def bulkDownloadXMLLOC(sourceList: Tuple[str]) -> None:
	# Scrape EAD XML files from the library of congress website.
	resultList = []
	print("Fetching links to documents...")
	for source in sourceList:
		url = "https://findingaids.loc.gov/source/" + source
		soup = BeautifulSoup(requests.get(url).content.decode("utf-8"), features="lxml")
		subList = [j.attrs["href"] for j in [i.find_all("a")[0] for i in soup.body.find_all("em")]]
		resultList += subList
		print("LOC Source", source, "yielded", len(subList), "results")

	for index, url in enumerate(resultList):
		print(f"Downloading document #{index}...")
		r = requests.get(url).content
		with open(f"{LOCDirectory}LOC{index}.xml", "wb") as f:
			f.write(r)


def scrapeKeyElements(filename: str, elements: List[str], textFilePos: int) -> int:
	# Scrape text from selected elements and output it to individual text files
	# for further analysis
	print("Scraping", filename + "...")
	try:
		with open(filename, "r") as f:
			content = f.read()
	except UnicodeDecodeError:
		print(filename, "is not EAD XML. Moving on.")
		return textFilePos
	else:
		if "<ead " not in content:
			print(filename, "is not EAD XML. Moving on.")
			return textFilePos
		soup = BeautifulSoup(content, features="lxml")
		words = " ".join(
			" ".join([
				re.sub(
					' +', ' ', re.sub('\n+|\t+', ' ', i.p.getText().strip())
				) for i in soup.find_all(element) if i.p is not None
			]) for element in elements
		).casefold()
	print("Dumping", filename + "...")
	with open(f"{txtDirectoryStructure}{textFilePos}.txt", "w+") as f:
		f.write(words)
	return textFilePos + 1


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
		condition = lambda w: len(w) < 3 or w in stopWords or not w.isalpha()
		finder.apply_word_filter(condition)
		# Stop word lambda also removes prepositions and other words that don't
		# impart a ton of semantic meaning. This is also optional.


	for triplet in collocationTuples:
		print(f"Most common {triplet[0]}-word phrases:")
		finder = triplet[1].from_words(text, triplet[0])
		applyFilter(finder)
		print(finder.nbest(triplet[2], 100))


if __name__ == "__main__":
	for directory in (LOCDirectory, txtDirectoryStructure):
		if not os.path.isdir(directory):
			os.mkdir(directory)

	bulkDownloadXMLLOC(LOCSources)

	for f in os.listdir(LOCDirectory):
		textFilePos = scrapeKeyElements(LOCDirectory + f, elements, textFilePos)

	for pair in GitHubSources:
		if not os.path.isdir(pair[1]):
			print("Cloning git archive", pair[1], "this may take a while...")
			Repo.clone_from(pair[0], pair[1])
		else:
			print("Pulling current version of", pair[1] + "...")
			commit = Remote(Repo(pair[1]), "origin").fetch()[0]
			print("Pulled commit", commit.commit.hexsha)

		for root, dirs, files in os.walk(pair[1]):
			for name in files:
				if name.endswith(".xml"):
					textFilePos = scrapeKeyElements(os.path.join(root, name), elements, textFilePos)

	getCollocations()