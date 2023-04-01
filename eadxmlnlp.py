import logging
import os
import random
import re
from typing import Dict, List, Tuple
from sys import stdout

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
from dateutil import parser
from git import Remote, Repo


"""
Global variables:
	LOCSources: tuple of paths, each of which is appended to the Library of Congress URI.
	GitHubSources: GitHub repos and the directory into which they will be downloaded.
	repoDirectory: directory containing the above repos and the downloaded LOC files.
	txtDirectory: directory containing txt files scraped from xml.
	elements: tags containing <p> tags to scrape.
	ignoredWords: set of words to ignore when gathering collocations.
	stopWords: ignoredWords joined with a set of predefined stopWords that impart little meaning.
	lemmatizer: predefined nltk lemmatizer utility.
	textFilePos: tracking variable used for naming txt files stored in txtDirectory.
	averageTagLength: dict containing data for each tag in elements related to average length.
		Structued in the form tag: [# times this tag appears, total # characters stored
		in this tag, total # words stored in this tag].
	logLevel: level of logs to display in logging library. Anything higher than
		INFO (for instance, logging.ERROR), means info-lvel messages will not display.
	dateList: list of archival years, obtained through the first date tag in each file.
"""
LOCSources = "LCA", "AFC", "AD", "EUR", "GC", "GMD", "HISP", "MSS_A", "MI", "MUS", "PP", "RBC", "RS", "VHP"
GitHubSources = (
	("https://github.com/NYULibraries/findingaids_eads.git", "NYU"),
	("https://github.com/RockefellerArchiveCenter/data.git", "Rockefeller"),
	("https://github.com/HeardLibrary/finding-aids.git", "Heard"),
	("https://github.com/Ukrainian-History/finding-aids.git", "UkrHEC")
)
repoDirectory = "./repos/"
LOCDirectory = repoDirectory + "LOC/"
txtDirectory = "./txtFiles/"
elements = ["scopecontent", "processinfo", "arrangement"]
ignoredWords = {"draw", "drawing", "york", "rockefeller", "president", "correspondent", "united", "urban", "policy", "international"}
stopWords = set(stopwords.words("english")).union(ignoredWords)
lemmatizer = WordNetLemmatizer()
textFilePos = 0
averageTagLength = {tag: [0, 0, 0] for tag in elements}
logLevel = logging.INFO
dateList = []

def bulkDownloadXMLLOC(sourceList: Tuple[str]) -> None:
	# Scrape EAD XML files from the library of congress website.
	resultList = []
	logging.info("Fetching links to LOC documents...")
	for source in sourceList:
		url = "https://findingaids.loc.gov/source/" + source
		soup = BeautifulSoup(requests.get(url).content.decode("utf-8"), features="lxml")
		subList = [i.find("a").attrs["href"] for i in soup.body.find_all("em")]
		resultList += subList
		logging.info(f"LOC Source {source} yielded {len(subList)} results")

	for index, url in enumerate(resultList):
		logging.info(f"Downloading LOC document #{index}...")
		r = requests.get(url).content
		with open(f"{LOCDirectory}LOC{index}.xml", "wb") as f:
			f.write(r)


def scrapeKeyElements(filename: str, elements: List[str], textFilePos: int) -> int:
	# Scrape text from selected elements and output it to individual text files
	# for further analysis. Also collects year from first instance of date tag.
	logging.info(f"Reading {filename}...")
	try:
		with open(filename, "r") as f:
			content = f.read()
	except UnicodeDecodeError:
		logging.info(f"{filename} uses utf-16. Moving on.")
		return textFilePos
	else:
		if not ("<ead " in content or "eadheader" in content):
			logging.info(f"{filename} is not EAD XML. Moving on.")
			return textFilePos
		global averageTagLength
		soup = BeautifulSoup(content, features="lxml")
		try:
			d = soup.find("date")
			dateList.append(str(parser.parse(d.attrs["normal"] if "normal" in d.attrs else d.text).year))
		except:
			logging.info("No valid date in " + filename)
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
	outputFilename = f"{txtDirectory}{textFilePos}.txt"
	logging.info(f"Writing to {outputFilename}...")
	with open(outputFilename, "w+") as f:
		f.write(words)
	return textFilePos + 1


def getCollocations() -> None:
	# Get list of common phrases across all examined documents.
	collocationTuples = (
		(2, BigramCollocationFinder, BigramAssocMeasures().likelihood_ratio),
		(3, TrigramCollocationFinder, TrigramAssocMeasures().likelihood_ratio),
		(4, QuadgramCollocationFinder, QuadgramAssocMeasures().likelihood_ratio)
	)
	logging.info("Building corpus...")
	corpus = PlaintextCorpusReader(txtDirectory, ".*")
	logging.info("Lemmatizing...")
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
	# Calculate average length of each tag in terms of characters and words.
	x = lambda a, b: round(a/(max(b, 1)))
	for tag, group in tags.items():
		chars = x(group[1], group[0])
		words = x(group[2], group[0])
		print(f"Average length of <p> in {tag} tag: {chars} characters, {words} words")


if __name__ == "__main__":
	logging.basicConfig(
		format="%(asctime)s: %(message)s",
		datefmt="%m/%d %H:%M:%S",
		level=logLevel,
		stream=stdout,
		force=True
	)

	for directory in (repoDirectory, LOCDirectory, txtDirectory):
		if not os.path.isdir(directory):
			logging.info(f"Creating {directory}...")
			os.mkdir(directory)

	bulkDownloadXMLLOC(LOCSources)

	for f in os.listdir(LOCDirectory):
		textFilePos = scrapeKeyElements(LOCDirectory + f, elements, textFilePos)

	for pair in GitHubSources:
		path = repoDirectory + pair[1]
		if not os.path.isdir(path + "/.git"):
			logging.info(f"Cloning git-hosted archive into {path}, this may take a while...")
			Repo.clone_from(pair[0], path)
		else:
			logging.info(f"Pulling current version of {path}...")
			commit = Remote(Repo(path), "origin").pull()[0]
			logging.info(f"Pulled commit {commit.commit.hexsha}")

		for root, dirs, files in os.walk(path):
			for name in files:
				if name.endswith(".xml"):
					textFilePos = scrapeKeyElements(os.path.join(root, name), elements, textFilePos)

	print(textFilePos, "valid EAD XML files. Excluded words:", ", ".join(ignoredWords))
	getCollocations()
	processAverageTagLength(averageTagLength)

	with open("./dates.txt", "w") as f:
		f.write("\n".join(i for i in dateList))
