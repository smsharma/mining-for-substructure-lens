import re
import bibtexparser
from bibtexparser.bparser import BibTexParser

# Open contents of bib file
with open('lensing-lfi.bib') as bibtex_file:
    bib_database = bibtexparser.load(bibtex_file)
    
# Edit contents
bib_entries = []

for bib_entry in bib_database.entries:
    bib_entry_0 = bib_entry
    if 'journal' in bib_entry_0:
        if bool(re.match('mon', bib_entry_0['journal'], re.I)) or bool(re.match('MNRAS', bib_entry_0['journal'], re.I)) :
            bib_entry_0['journal'] = '\\mnras'
        if bool(re.match('Journal of Cosmology', bib_entry_0['journal'], re.I)) or bool(re.match('JCAP', bib_entry_0['journal'], re.I)):
            bib_entry_0['journal'] = '\\jcap'
        if bool(re.match('Physical Review D', bib_entry_0['journal'], re.I)):
            bib_entry_0['journal'] = '\\prd'
        if bool(re.match('Journal of High Energy Physics', bib_entry_0['journal'], re.I)):
            bib_entry_0['journal'] = 'JHEP'
        if bool(re.match('Astronomy and Astrophysics', bib_entry_0['journal'], re.I)):
            bib_entry_0['journal'] = '\\aap'
        if bool(re.match('Physical Review Letters', bib_entry_0['journal'], re.I)):
            bib_entry_0['journal'] = '\\prl'
        if bool(re.search('Astrophysical', bib_entry_0['journal'])):
            bib_entry_0['journal'] = '\\apj'
        if bool(re.match('arxiv', bib_entry_0['journal'], re.I)):
            bib_entry_0['journal'] = ''
            bib_entry_0['pages'] = ''
    bib_entries.append(bib_entry_0)
    
# Save edited contents to bib file

bib_database.entries = bib_entries

with open('lensing-lfi.bib', 'w') as bibtex_file:
    bibtexparser.dump(bib_database, bibtex_file)