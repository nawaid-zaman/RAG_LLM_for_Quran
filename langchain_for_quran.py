#%%
import PyPDF2
import nltk
import re
import fitz  # PyMuPDF
from spellchecker import SpellChecker
import pandas as pd
from word_list_dictionary import replacement_dict, additional_english_words, names, islamic_words, locations

#%%
pdf_path = ("C:\\Users\\nawai\\Downloads\\The-Quran-Saheeh-International.pdf")
skip_pages = [2, 8, 9, 10, 712]


# Download nltk words corpus
nltk.download('words')
from nltk.corpus import words



#%%
# Function to extract text from a PDF
# def extract_text_from_pdf(pdf_path):
#     with open(pdf_path, "rb") as file:
#         reader = PyPDF2.PdfReader(file)
#         text = ""
#         for page_num in range(len(reader.pages)):
#             text += reader.pages[page_num].extract_text()
#     return text




# def extract_text_from_pdf(pdf_path):
#     full_text = ""

#     with fitz.open(pdf_path) as pdf:
#         # Iterate over all the pages
#         for page in pdf:
#             text = page.get_text()
#             full_text += text + "\n"  # Append the text of each page
#     return full_text


import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path, skip_pages=None):
    """
    Extract text from a PDF while skipping specific pages.

    Parameters:
        pdf_path (str): The path to the PDF file.
        skip_pages (list): List of page numbers to skip. Defaults to None.

    Returns:
        str: Extracted text from the PDF.
    """
    full_text = ""
    skip_pages = skip_pages or []  # Initialize skip_pages to an empty list if not provided
    skip_pages = [x - 1 for x in skip_pages]

    with fitz.open(pdf_path) as pdf:
        # Iterate over all the pages by page number
        for page_num in range(len(pdf)):
            if page_num in skip_pages:
                continue  # Skip the page if it's in the skip_pages list

            # Get text from the page
            page = pdf.load_page(page_num)
            text = page.get_text()

            full_text += text + "\n"  # Append the text of each page

    return full_text



# Function to check if words are English
def is_english_word(word):
    return word.lower() in english_words

# Initialize the spell checker for English (default is 'en')
spell = SpellChecker()

# Function to check if a word is in the dictionary
def is_english_word(word):
    return word in spell


def remove_english_words(non_english_words):
    return [word for word in non_english_words if word not in spell]


# Function to split numbers and words
def split_numbers_and_words(mixed_list):
    split_list = []
    for item in mixed_list:
        # Use regex to split numbers and words
        split_parts = re.findall(r'\d+|\D+', item)  # \d+ matches numbers, \D+ matches non-numbers
        split_list.extend(split_parts)  # Extend the list with the separated parts
    return split_list

# Define a function to check for Roman numerals
def is_roman_numeral(word):
    # Roman numeral pattern: I, V, X, L, C, D, M (upper or lower case)
    roman_pattern = r'^(M{0,4})(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$'
    return re.match(roman_pattern, word, re.IGNORECASE) is not None

def replace_words(text, replacement_dict):
    # Iterate through the dictionary and replace each key with its corresponding value
    for word, replacement in replacement_dict.items():
        text = text.replace(word.lower(), replacement.lower())
    return text


#%%
# Set of valid English words
english_words = set(words.words())
type(english_words)

# Convert all words to lowercase
english_words = set(word.lower() for word in english_words)

#%%
# Extract words and check if they are English
text = extract_text_from_pdf(pdf_path,skip_pages).lower()

#%%
# Replace words with correct format words
text = replace_words(text, replacement_dict)
# print(text)
#%%
# Extract words using regex \b[\w´õœ-]+\b
# words_in_pdf = re.findall(r'\b\w+\b', text)  

# re_expression = r'\b[\w´õœŒ]+\b'
re_expression = r'\b[\w´õœŒ¥§‹œ]+\b'

words_in_pdf = re.findall(re_expression, text)  
words_in_pdf = set(words_in_pdf)
 

#%%
# remove english words from text 
non_english_words = [word for word in words_in_pdf if not is_english_word(word)]

# Call the function split_numbers_and_words
non_english_words = split_numbers_and_words(non_english_words)

# Update the list by removing English words through SpellChecker
non_english_words = remove_english_words(non_english_words)

# Remove custom islamic words
# non_english_words = ([word for word in non_english_words if word not in islamic_words])
non_english_words = [word for word in non_english_words if word.lower() not in (x.lower() for x in islamic_words)]


# Remove or replace custom names irrespective of case
non_english_words = [word for word in non_english_words if word.lower() not in (x.lower() for x in names)]

# Remove custom additional english words
non_english_words = [word for word in non_english_words if word.lower() not in (x.lower() for x in additional_english_words)]

# Remove custom locations
# non_english_words = [word for word in non_english_words if word not in locations]
non_english_words = [word for word in non_english_words if word.lower() not in (x.lower() for x in locations)]

# Removing numbers from the list
non_english_words = [word for word in non_english_words if not word.isdigit()]

# Remove Roman numerals from the list
non_english_words = [word for word in non_english_words if not is_roman_numeral(word)]
len(non_english_words)

#%%
non_english_words = list(set(non_english_words))
non_english_words

#%%

# replacement_values = list(replacement_dict.values())
# non_english_words = [word for word in non_english_words if word.lower() not in (x.lower() for x in replacement_values)]

#%%
len(non_english_words)


#%%
def count_occurrences(text, non_english_words):

    words_in_pdf = re.findall(re_expression, text) 

    word_dict = {}
    for word in non_english_words:
        word_dict[word] = words_in_pdf.count(word)

    word_counts_df = pd.DataFrame(word_dict.items(), columns=['word', 'count']).sort_values('count', ascending=False)
    
    return word_counts_df

result = count_occurrences(text, non_english_words)
result
#%%
result['word'].tolist()
# %%
len(non_english_words)
# %%
print(text)
# %%
with open('quran_draft.txt', 'w', encoding='utf-8') as file:
        file.write(text)




#%%
