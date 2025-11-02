#!/bin/bash
set -e

python -m pip install --upgrade pip
pip install -r requirements.txt

# Run All Languages Sequentially
echo Running preprocessing for Korean
python preprocessing.py -lang ko -n 10000

echo Running preprocessing for French 
python preprocessing.py -lang fr -n 10000

echo Running preprocessing for Portuguese
python preprocessing.py -lang pt -n 10000

echo Running preprocessing for Spanish
python preprocessing.py -lang es -n 10000

echo Running preprocessing for English
python preprocessing.py -lang en -n 10000

echo Running preprocessing for Tamil
python preprocessing.py -lang ta -n 10000

echo Running preprocessing for Russian
python preprocessing.py -lang ru -n 10000

echo Running preprocessing for Belarussian
python preprocessing.py -lang be -n 10000

echo Running preprocessing for German
python preprocessing.py -lang ge -n 10000

echo Running preprocessing for Italian
python preprocessing.py -lang it -n 10000

echo all done
