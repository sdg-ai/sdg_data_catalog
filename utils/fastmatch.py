###############################################################################
#
# FastMatch()
#
# Author: James Hodson (james@hodson.io), Cognism Ltd.
#
# A fast exact string matching algorithm for sets of candidate strings
# that may be contained in a larger text corpus.
#
# Matches are case-sensitive. If that's not what you want, then lowercase
# candidates and input text.
#
# Input:
#	__init__()
#		(candidates): A list of candidate strings to look for.
#		(valid start): 	A list of characters that mark a word boundary,
#						with beginning of string included by default.
#						Default: [BEGIN,'\n','\t','/',' ','-','.',',' ...]
#		(valid_end):	A list of characters that mark a word boundary,
#						with end of string included by default.
#						Default: [EOF,'\n','\t','/',' ','-','.',',' ...]
#
# Usage:
#	match()
#		(text): A string containing the text to search for candidate matches.
#
#	Returns a list of (match,start,end) tuples.
#
# NB:
#	* Matches cased as entered, no string modifications are performed.
#	* Computational Complexity is O(n), worst case complexity n*m, where
#	  n is length of document to match and m is number of candidate matches.
#	* valid_start and valid_end always implicitly consider the start and end
#	  of the document as valid boundaries.
#	* valid_start and valid_end are context-free tokenizers of the document.
#	  Candidates may contain end boundaries and start boundaries and will
#	  still match correctly.
#	* Substring matching is possible by expanding the valid_start and valid_end
#	  with every character that may be contained in a valid word. However, this
#	  is probably not the best use case for this module.
#
###############################################################################


class FastMatch(object):

	def __init__(self, candidates, valid_start=None, valid_end=None):

		self.candidates = candidates
		self.map = {}

		self.start = valid_start
		self.end = valid_end

		if not self.start:
			self.start = set(['\n','\t','/',' ','-','#','.','&','(',')','+','!',':',';','"',"'",'?','[',']'])

		if not self.end:
			self.end = set(['\n','\t','/',' ','-','#','.','&','(',')','+','!',':',';','"',"'",'?','[',']'])

		self.preprocess()

	# Build the hash-tree structure that is the
	# backbone of the matching procedure.
	def preprocess(self):

		for candidate in self.candidates:
			pointer = None
			for i, letter in enumerate(candidate,start=1):
				if i == 1:
					if letter not in self.map:
						self.map[letter] = {'__STOP__':None}
					pointer = self.map[letter]

				if i == len(candidate):
					if i == 1:
						pointer['__STOP__'] = letter
					else:
						if letter not in pointer:
							pointer[letter] = {}
						pointer[letter]['__STOP__'] = candidate

				if i > 1 and i < len(candidate):
					if letter not in pointer:
						pointer[letter] = {'__STOP__':None}
					pointer = pointer[letter]

	# Iterate over text and identify matches from candidate strings.
	def match(self,text):

		if not self.map:
			raise ValueError("No candidates in map. Did you preprocess()?")

		if not text:
			return None

		matches = []
		match_ptrs = []

		is_start = False

		for i, letter in enumerate(text,start=1):
			# Special case for beginning of string.
			if i == 1:
				is_start = True

			temp_ptrs = []
			for match in match_ptrs:
				if letter in match:
					temp_ptrs.append(match[letter])
			match_ptrs = temp_ptrs

			# If the previous character was a valid start
			# boundary, then add to potential match pointers.
			if is_start:
				if letter in self.map:
					match_ptrs.append(self.map[letter])
				is_start = False

			# If this is a valid end boundary, then check all match_ptrs for
			# valid matches and clear them as needed.
			if i == len(text) or text[i] in self.end:
				temp_ptrs = []
				for match in match_ptrs:
					if match['__STOP__']:
						matches.append((match['__STOP__'], i-len(match['__STOP__']), i))
					else:
						temp_ptrs.append(match)
				match_ptrs = temp_ptrs

			if letter in self.start:
				is_start = True

		return matches

