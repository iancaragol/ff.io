import sys
import os
import re
import io

from requests import get
from requests.exceptions import RequestException
from contextlib import closing
from bs4 import BeautifulSoup

"""
This webscraper works for fancfiction.net. It reads urls without chapters from a text file
then saves all stories into a single text file.
"""

def get_links(filepath):
    """
    Gets URLs from the url text file, one per line
    """
    urls = []
    with open(filepath, 'r') as f:
        for line in f:
            urls.append(line.strip())
    return urls

def main():
    urls_filepath = sys.path[0]+"\\_data\\ff_urls.txt" # File to read url's from
    urls = get_links(urls_filepath)
    scrap = Scraper()
    
    print("|------ Scraping fanfiction.net ------|")

    with io.open(urls_filepath, 'w', encoding="utf-8") as text_file:
        for u in urls:
            story = scrap.to_text(u)
            text_file.write(story)
            text_file.write('\n')

    print()


class Scraper:
    def to_text(self, url):
        """
        Gets chapter count from url and saves entire story to .txt file.
        """
        chapter_id = 1

        # Url format is url/chapter_id
        url = url + '/' + str(chapter_id)
        soup = BeautifulSoup(self.simple_get(url), 'html.parser')
        soup = soup.find("div", {"id":"profile_top"})
        soup_text = soup.text

        # Get the number of chapters from the stories profile
        r = re.search('Chapters: (\d+)', soup_text, re.IGNORECASE)
        chapter_count = 1
        if r is not None:
            chapter_count = int(re.search('Chapters: (\d+)', soup_text, re.IGNORECASE).group(1))
        
        text = ''
       
        # While there are still chapters left, read them and add to text
        while(chapter_id <= chapter_count):
            print(f"Story: {url}, Chapter: #{chapter_id}/{chapter_count}         ", end='\r', flush=True)
            text += self.chapter_to_text(url)
            url = url[:-(len(str(chapter_id))+1)]
            chapter_id += 1
            url = url + '/' + str(chapter_id)

        return text

    def chapter_to_text(self, url):
        """
        Parses out chapters text.
        """
        soup = BeautifulSoup(self.simple_get(url), 'html.parser')
        soup = soup.find('div', id='storytext').findAll('p')

        text = ''

        for s in soup:
            text += '\n' + ''.join(s.findAll(text = True))
        return text

    def simple_get(self, url):
        """
        Gets page content.
        """
        try:
            with closing(get(url, stream=True)) as resp:
                if self.is_good_response(resp):
                    return resp.content
                else:
                    return None
        except RequestException as e:
            self.log_error('Error during requests to {0} : {1}'.format(url, str(e)))
            raise Exception('Chapter limit reached.')

    def is_good_response(self, resp):
        content_type = resp.headers['Content-Type'].lower()
        return (resp.status_code == 200 and content_type is not None and content_type.find('html') > -1)

    def log_error(self, e):
        """
        Catches and prints errors.
        """
        print(e)



if __name__ == "__main__":
    main()