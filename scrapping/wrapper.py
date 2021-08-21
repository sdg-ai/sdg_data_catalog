#wrapper
import PyPDF2 as pdf
import gzip
import certifi
from pathlib import Path
import glob
import os
import re
from bs4 import BeautifulSoup
import ujson
import argparse
import time
from multiprocessing import Pool
from io import BytesIO
import pycurl
import urllib.parse as uparse

import warnings
warnings.filterwarnings("ignore")

'''
This script extract papers based on their title that we collected form the aminer website
It follow the corresponding methodology:
- make a google a search of the paper title and filtering for pdf results only
- trying to download the top 3 results from google
- save the first succesful download
scraperapi is used to enable the large scale scrapping 
'''

#key items for search
SEARCH_URL = "https://google.com/search"
search_filter = 'filetype:pdf'
scraper_api = 'http://api.scraperapi.com'
with open('api_key.json') as f:
	api_key = ujson.load(f)['api_key']

def extract_all_titles(doc_txt,begin=0,maxlines=0):
	'''
	Extract all the titles from the file and store them in a list
	'''
	title_list = []
	start = time.perf_counter()
	mf = open(doc_txt, 'rt', encoding='utf-8')
	i = 0
	for line in mf:
		i += 1
		js = ujson.loads(line)
		id = js.get('id',None)
		if not id or os.path.isfile('papers/'+id+'.pdf.gz'):
			continue
		if i <= begin:
			continue
		if maxlines and i - begin > maxlines:
			break
		nmes = []
		if 'authors' in js:
			for j in js.get('authors',[]):
				if len(j.get('name','').strip()) > 1 and len(nmes) < 5:
					nmes.append(max(j.get('name','').split(),key=len))
		title = (js.get('title','') + ' ' + ' '.join(nmes)).strip()
		if title and not os.path.isfile('papers/'+id+'.pdf.fail'):
			title_list.append((title,id))
		if i % 100000 == 0:
			print(i, ', time: ', time.perf_counter() - start,'; ',len(title_list),' added.')
	return title_list

def get_paper(title):
	'''
	Make requests and download papers from the web
	'''

	def curl_request(url, buf=False, timeout=20):
		# PyCurl needs somewhere to store headers.
		headers = {}
		def get_header(header_line):
			header_line = header_line.decode('iso-8859-1')
			if ':' not in header_line:
				return
			name, value = header_line.split(':', 1)
			name = name.strip().lower()
			value = value.strip()
			headers[name] = value

		buffer = BytesIO()
		c = pycurl.Curl()
		c.setopt(c.URL, url)
		c.setopt(c.WRITEFUNCTION, buffer.write)
		# Set our header function.
		c.setopt(c.HEADERFUNCTION, get_header)
		c.setopt(c.FOLLOWLOCATION, True)
		c.setopt(c.CAINFO, certifi.where())
		c.setopt(c.SSL_VERIFYPEER, 0)
		c.setopt(c.UNRESTRICTED_AUTH, True)
		c.setopt(c.TIMEOUT, timeout)
		try:
			c.perform()
		except pycurl.error as e:
			print(e)
			print(url)
		c.close()

		# Figure out what encoding was sent with the response, if any.
		# Check against lowercased header name.
		encoding = None
		content_type = None
		if 'content-type' in headers:
			content_type = headers['content-type'].lower()
			match = re.search('charset=(\S+)', content_type)
			if match:
				encoding = match.group(1)
		if encoding is None:
			# Default encoding for HTML is iso-8859-1.
			# Other content types may have different default encoding,
			# or in case of binary data, may have no encoding at all.
			encoding = 'iso-8859-1'

		if not buf:
			buffer = buffer.getvalue().decode(encoding)
		return content_type, buffer

	full_url = scraper_api+'?api_key='+api_key+'&url='+SEARCH_URL+'?q='+search_filter+uparse.quote(' '+title[0])

	#make the request and get content as html
	headers, body = curl_request(full_url,timeout=30)
	soup = BeautifulSoup(body, "html.parser")

	results = []
	for g in soup.find_all('div', class_='r')[:3]:
		anchors = g.find_all('a')
		if anchors:
			results.append(anchors[0]['href'])

	headers, body = None, None
	for res in results:
		headers, body = curl_request(res,buf=True,timeout=60)
		if headers and 'application/pdf' in headers:
			#we save the downloaded pdf
			with gzip.open('papers/{}.pdf.gz.tmp'.format(title[1]), 'wb') as f:
				f.write(body.getbuffer())
			try:
				pdf.PdfFileReader(gzip.open('papers/{}.pdf.gz.tmp'.format(title[1]), "rb"))
			except pdf.utils.PdfReadError:
				print("invalid PDF file")
				os.remove('papers/{}.pdf.gz.tmp'.format(title[1]))
				continue

			os.rename('papers/'+title[1]+'.pdf.gz.tmp','papers/'+title[1]+'.pdf.gz')
			return

	Path('papers/'+title[1]+'.pdf.fail').touch()
	return


if __name__ == '__main__':
	# Initialize the arguments
	prs = argparse.ArgumentParser()
	prs.add_argument('-t', '--num_threads', help='Number of Threads to use', type=int, default=8)
	prs.add_argument('-s', '--start_from', help='What record to start from?', type=int, default=0)
	prs.add_argument('-n', '--max_records', help='Number of records to try to retrieve', type=int, default=0)
	prs = prs.parse_args()

	titles_list = []
	for el in glob.glob('archive/*.txt'):
		for i in extract_all_titles(el,prs.start_from,prs.max_records):
			titles_list.append(i)
	print('titles extracted')

	# Make the Google search and download the corresponding papers
	p = Pool(prs.num_threads)
	start = time.perf_counter()
	p.map(get_paper, titles_list)
	end = time.perf_counter()
	print('Time to process the download part: ', end - start,' for ',len(titles_list),' documents.')
