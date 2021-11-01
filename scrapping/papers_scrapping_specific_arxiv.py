#wrapper
#import PyPDF2 as pdf
import gzip
import PyPDF2 as pdf
import certifi
from pathlib import Path
import glob
import os
import re
from bs4 import BeautifulSoup
import argparse
import time
from multiprocessing import Pool
from io import BytesIO
import pycurl
import urllib.parse as uparse
#from utils.keywords import *
import json

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
SEARCH_URL = "https://arxiv.org/search/?query="
search_filter = 'filetype:pdf'
scraper_api = 'http://api.scraperapi.com'
with open('../api_key.json') as f:
	api_key = json.load(f)['api_key']

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

def get_paper(title, sdg):
	'''
	Make requests and download papers from the web
	'''
	print(title)
	title = '+'.join(title.split(' '))

	full_url = scraper_api+'?api_key='+api_key+'&url='+SEARCH_URL+title+ '&searchtype=all&source=header'+'&size=50'
	print('\n '+full_url+'\n ')

	#make the request and get content as html
	headers, body = curl_request(full_url)
	soup = BeautifulSoup(body, "html.parser")

	results = []
	#for g in soup.find_all('div', class_='r')[:3]:
	anchors = soup.find_all('a')
	if anchors:
		for el in anchors:
			#print(el)
			if 'pdf' in el:
				try:
				    results.append(el['href'])
				except:
					pass
	else:
		pass
		print('passing')
	print(len(results))

	headers, body = None, None
	for i, res in enumerate(results):
		print(res)
		headers, body = curl_request(res,buf=True,timeout=60)
        
		if not os.path.exists('papers/{}'.format(sdg)):
			os.makedirs('papers/{}'.format(sdg))
		#we save the downloaded pdf
		with gzip.open('papers/{}/{}_{}.pdf.gz'.format(sdg, title, i), 'wb') as f:
			f.write(body.getbuffer())
			print('saved')
			#try:
			#	pdf.PdfFileReader(gzip.open('papers/{}_{}.pdf.gz.tmp'.format(title, i), "rb"))
			#except pdf.utils.PdfReadError:
			#	print("invalid PDF file")
			#	os.remove('papers/{}_{}.pdf.gz.tmp'.format(title, i))
			#	print('removed')
			#	continue

			#os.rename('papers/'+title+'_'+i+'.pdf.gz.tmp','papers/'+title+'_'+i+'.pdf.gz')

	#Path('papers/'+title+'_'+i+'.pdf.fail').touch()
	print('')
	return


if __name__ == '__main__':
	# Initialize the arguments
	prs = argparse.ArgumentParser()
	prs.add_argument('-t', '--num_threads', help='Number of Threads to use', type=int, default=8)
	prs.add_argument('-s', '--start_from', help='What record to start from?', type=int, default=0)
	prs.add_argument('-n', '--max_records', help='Number of records to try to retrieve', type=int, default=0)
	prs.add_argument('-k', '--key_word', help='What key word to use for the search', type=str, default='gender')
	prs.add_argument('-sd', '--sdg', help='What is the sdg name (to name to folder accordingly)', type=str, default='gender')
	prs = prs.parse_args()

	#titles_list = []
	#for el in glob.glob('archive/*.txt'):
	#	for i in extract_all_titles(el,prs.start_from,prs.max_records):
#			titles_list.append(i)
#	print('titles extracted')
	titles_list = ['No Poverty', 'Zero Hunger', 'Good Health and Well-being'] 

	# Make the Google search and download the corresponding papers
	#p = Pool(prs.num_threads)
	start = time.perf_counter()
	#p.map(get_paper, prs.key_word)
	get_paper(prs.key_word, prs.sdg)
	end = time.perf_counter()
	print('Time to process the download part: ', end - start)
