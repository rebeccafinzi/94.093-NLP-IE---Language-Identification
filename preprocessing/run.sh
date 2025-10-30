#!/bin/bash
set -e
# python -m pip install -U pip
# pip install -r requirements.txt

echo Running preprocessing for Korean
python preprocessing.py -lang ko -n 1000  

echo Running preprocessing for French 
python preprocessing.py -lang fr -n 1000  

echo Running preprocessing for Portuguese
python preprocessing.py -lang pt -n 1000  

echo Running preprocessing for Russian 
python preprocessing.py -lang ru -n 1000  

echo Running preprocessing for Belarusian
python preprocessing.py -lang be -n 1000 

echo all done
