import os
import random
import re
from typing import Dict, List, Tuple

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
	("https://github.com/HeardLibrary/finding-aids.git", "./Heard"),
	("https://github.com/Ukrainian-History/finding-aids.git", "./UkrHEC")
)
repoDirectory = "./repos/"
LOCDirectory = repoDirectory + "LOC/"
txtDirectory = "./txtFiles/"
elements = ["scopecontent", "processinfo", "arrangement"]
ignoredWords = {"draw", "drawing", "york", "rockefeller", "president", "correspondent"}
stopWords = set(stopwords.words("english")).union(ignoredWords)
lemmatizer = WordNetLemmatizer()
textFilePos = 0
averageTagLength = {tag: [0, 0, 0] for tag in elements}

def bulkDownloadXMLLOC(sourceList: Tuple[str]) -> None:
	# Scrape EAD XML files from the library of congress website.
	resultList = []
	print("Fetching links to LOC documents...")
	for source in sourceList:
		url = "https://findingaids.loc.gov/source/" + source
		soup = BeautifulSoup(requests.get(url).content.decode("utf-8"), features="lxml")
		subList = [j.attrs["href"] for j in [i.find_all("a")[0] for i in soup.body.find_all("em")]]
		resultList += subList
		print("LOC Source", source, "yielded", len(subList), "results")

	for index, url in enumerate(resultList):
		print(f"Downloading LOC document #{index}...")
		r = requests.get(url).content
		with open(f"{LOCDirectory}LOC{index}.xml", "wb") as f:
			f.write(r)


def scrapeKeyElements(filename: str, elements: List[str], textFilePos: int) -> int:
	# Scrape text from selected elements and output it to individual text files
	# for further analysis
	print("Reading", filename + "...")
	try:
		with open(filename, "r") as f:
			content = f.read()
	except UnicodeDecodeError:
		print(filename, "uses utf-16. Moving on.")
		return textFilePos
	else:
		if not ("<ead " in content or "eadheader" in content):
			print(filename, "is not EAD XML. Moving on.")
			return textFilePos
		global averageTagLength
		soup = BeautifulSoup(content, features="lxml")
		words = ""
		for element in elements:
			elementList = soup.find_all(element)
			strippedWord = ""
			for i in elementList:
				if i.p is not None:
					text = i.p.getText()
					processedWord = re.sub(
						" +", " ", re.sub("\n+|\t+", " ", text.strip())
					).casefold()
					strippedWord += processedWord + " "
					averageTagLength[element][0] += 1
					averageTagLength[element][1] += len(processedWord)
					averageTagLength[element][2] += len(processedWord.split(" "))
			words += strippedWord[:-1]
	outputFilename = f"{textFilePos}.txt"
	print(f"Writing to {outputFilename}...")
	with open(f"{txtDirectory}{outputFilename}", "w+") as f:
		f.write(words)
	return textFilePos + 1


def getCollocations() -> None:
	collocationTuples = (
		(2, BigramCollocationFinder, BigramAssocMeasures().likelihood_ratio),
		(3, TrigramCollocationFinder, TrigramAssocMeasures().likelihood_ratio),
		(4, QuadgramCollocationFinder, QuadgramAssocMeasures().likelihood_ratio)
	)
	print("Building corpus...")
	corpus = PlaintextCorpusReader(txtDirectory, ".*")
	print("Lemmatizing...")
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


def processAverageTagLength(tags: Dict[str, List[int]]) -> None:
	x = lambda a, b: round(a/(max(b, 1)))
	for tag, group in tags.items():
		chars = x(group[1], group[0])
		words = x(group[2], group[0])
		print(f"Average length of <p> in {tag} tag: {chars} characters, {words} words")


if __name__ == "__main__":
	for directory in (repoDirectory, LOCDirectory, txtDirectory):
		if not os.path.isdir(directory):
			print("Creating", directory + "...")
			os.mkdir(directory)

	bulkDownloadXMLLOC(LOCSources)

	for f in os.listdir(LOCDirectory):
		textFilePos = scrapeKeyElements(LOCDirectory + f, elements, textFilePos)

	for pair in GitHubSources:
		path = repoDirectory + pair[1]
		if not os.path.isdir(path):
			print(f"Cloning git-hosted archive into {path}, this may take a while...")
			Repo.clone_from(pair[0], path)
		else:
			print("Pulling current version of", path + "...")
			commit = Remote(Repo(path), "origin").pull()[0]
			print("Pulled commit", commit.commit.hexsha)

		for root, dirs, files in os.walk(path):
			for name in files:
				if name.endswith(".xml"):
					textFilePos = scrapeKeyElements(os.path.join(root, name), elements, textFilePos)

	print(textFilePos, "valid EAD XML files. Excluded words:", ", ".join(ignoredWords))

	getCollocations()

	processAverageTagLength(averageTagLength)