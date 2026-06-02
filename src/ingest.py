import xml.etree.ElementTree as ET
import urllib.request
import urllib.parse  # ◀── Added this to fix the URL characters!
import os

def download_and_parse_arxiv_corpus(max_results=500):
    """
    Downloads live AI agent papers from the arXiv API, extracts their text,
    and cuts them into manageable chunks for our retriever database.
    """
    # Define our search keywords
    raw_query = 'cat:cs.CL+OR+cat:cs.AI+OR+cat:cs.LG'
    
    # ◀── This safely converts spaces to %20 so Python doesn't crash!
    encoded_query = urllib.parse.quote(raw_query, safe='+:')
    
    # Combine it into the final clean web address
    url = f'http://export.arxiv.org/api/query?search_query={encoded_query}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending'
    
    print(f"Connecting to arXiv API via url: {url}")
    response = urllib.request.urlopen(url)
    xml_data = response.read()
    
    root = ET.fromstring(xml_data)
    corpus_chunks = []
    
    for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
        arxiv_id_url = entry.find('{http://www.w3.org/2005/Atom}id').text
        arxiv_id = arxiv_id_url.split('/abs/')[-1].split('v')[0]
        title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip().replace('\n', ' ')
        summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip().replace('\n', ' ')
        
        print(f"-> Processing paper match [{arxiv_id}]: {title[:50]}...")
        
        full_paper_text = f"Title: {title}. Abstract: {summary}"
        
        chunk_size = 400
        overlap = 100
        
        start = 0
        while start < len(full_paper_text):
            end = start + chunk_size
            text_segment = full_paper_text[start:end]
            
            corpus_chunks.append({
                "arxiv_id": arxiv_id,
                "text": text_segment
            })
            start += (chunk_size - overlap)
            
    print(f"Successfully processed and generated {len(corpus_chunks)} text database chunks.")
    return corpus_chunks