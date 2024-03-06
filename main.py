# import fitz  # PyMuPDF
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with fitz.open(pdf_path) as pdf_file:
#         for page_num in range(len(pdf_file)):
#             page = pdf_file.load_page(page_num)
#             text += page.get_text()
#     return text
#
# if __name__ == "__main__":
#     # Replace 'your_pdf_file.pdf' with the path to your PDF file
#     pdf_path = 'handbook.pdf'
#     extracted_text = extract_text_from_pdf(pdf_path)
#     print(extracted_text)


# import fitz
# from langdetect import detect
# import pytesseract
# from PIL import Image
# from summa.summarizer import summarize
# from summa import keywords
#
# # Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
#
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with fitz.open(pdf_path) as pdf_file:
#         for page_num in range(len(pdf_file)):
#             page = pdf_file.load_page(page_num)
#             pix = page.get_pixmap()
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#             detected_lang = detect(pytesseract.image_to_string(img, lang='mar'))
#             if detected_lang == 'mr':
#                 text += pytesseract.image_to_string(img, lang='mar') + "\n"
#     return text
#
#
# def summarize_text(text, ratio=0.2):
#     summary = summarize(text, ratio=ratio)
#     return summary
#
# # if __name__ == "__main__":
# #     # Assuming you have the extracted Marathi text stored in a variable 'extracted_text'
# import nltk
# from gensim.summarization import summarize
# from indicnlp.tokenize import indic_tokenize
#
# # Text preprocessing
# def preprocess_text(text):
#     tokens = indic_tokenize.trivial_tokenize(text)
#     words = [word for word in tokens if word.isalpha()]  # Remove non-alphabetic tokens
#     return ' '.join(words)
#
# # Keyword extraction
# def extract_keywords(text):
#     keywords = set()
#     words = nltk.word_tokenize(text)
#     tagged_words = nltk.pos_tag(words, lang='marathi')
#     for word, pos in tagged_words:
#         if pos.startswith('NN'):  # Extract only nouns
#             keywords.add(word)
#     return keywords
#
# # Summarization
# def generate_summary(text, ratio=0.2):
#     summary = summarize(text, ratio=ratio)
#     return summary
#
# # Highlighting keywords
# def highlight_keywords(summary, keywords):
#     for keyword in keywords:
#         summary = summary.replace(keyword, f"**{keyword}**")
#     return summary
#
#
# if __name__ == "__main__":
#     pdf_path = 'diwali.pdf'
#     extracted_text = extract_text_from_pdf(pdf_path)
#     print(extracted_text)
#     # extracted_text = "..."  # Your extracted Marathi text here
#
#     print("this is the distinction")
#     summary = summarize_text(extracted_text)
#     print(summary)
#     print("this is the distinction")
#     print(summarize(extracted_text, language='marathi'))
#     print("distinction")
#     print(keywords.keywords(extracted_text))
#     # Preprocess text
#     processed_text = preprocess_text(extracted_text)
#
#     # Extract keywords
#     keywords = extract_keywords(processed_text)
#
#     # Generate summary
#     summary = generate_summary(processed_text)
#
#     # Highlight keywords in summary
#     highlighted_summary = highlight_keywords(summary, keywords)
#
#     print("Original Text:")
#     print(extracted_text)
#
#     print("\nSummarized Text:")
#     print(highlighted_summary)



# import fitz
# from langdetect import detect
# import pytesseract
# from PIL import Image
# from summa.summarizer import summarize
# from summa import keywords
# import nltk
# from gensim.summarization import summarize as gensim_summarize
# from indicnlp.tokenize import indic_tokenize
#
# # Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
#
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with fitz.open(pdf_path) as pdf_file:
#         for page_num in range(len(pdf_file)):
#             page = pdf_file.load_page(page_num)
#             pix = page.get_pixmap()
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#             detected_lang = detect(pytesseract.image_to_string(img, lang='mar'))
#             if detected_lang == 'mr':
#                 text += pytesseract.image_to_string(img, lang='mar') + "\n"
#     return text
#
# def preprocess_text(text):
#     tokens = indic_tokenize.trivial_tokenize(text)
#     words = [word for word in tokens if word.isalpha()]  # Remove non-alphabetic tokens
#     return ' '.join(words)
#
# def extract_keywords(text):
#     keywords_set = set()
#     words = nltk.word_tokenize(text)
#     tagged_words = nltk.pos_tag(words, lang='marathi')
#     for word, pos in tagged_words:
#         if pos.startswith('NN'):  # Extract only nouns
#             keywords_set.add(word)
#     return keywords_set
#
# def generate_summary(text, ratio=0.2):
#     summary = gensim_summarize(text, ratio=ratio)
#     return summary
#
# def highlight_keywords(summary, keywords_set):
#     for keyword in keywords_set:
#         summary = summary.replace(keyword, f"**{keyword}**")
#     return summary
#
# if __name__ == "__main__":
#     pdf_path = 'diwali.pdf'
#     extracted_text = extract_text_from_pdf(pdf_path)
#     print(extracted_text)
#
#     summary = summarize(extracted_text)
#     print("Summarized Text using Summa:")
#     print(summary)
#
#     gensim_summary = generate_summary(extracted_text)
#     print("Summarized Text using Gensim:")
#     print(gensim_summary)
#
#     extracted_keywords = keywords.keywords(extracted_text)
#     print("Extracted Keywords:")
#     print(extracted_keywords)
#
#     processed_text = preprocess_text(extracted_text)
#     extracted_keywords = extract_keywords(processed_text)
#     highlighted_summary = highlight_keywords(gensim_summary, extracted_keywords)
#
#     print("\nSummarized Text with Highlighted Keywords:")
#     print(highlighted_summary)

# import fitz
# from langdetect import detect
# import pytesseract
# from PIL import Image
# from summa.summarizer import summarize
# from summa import keywords
# from indicnlp.tokenize import indic_tokenize
# from polyglot.text import Text
#
# # Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
#
# # Download polyglot resources for Marathi language
# from polyglot.downloader import downloader
# downloader.download("embeddings2.marathi")
# downloader.download("morph2.marathi")
#
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with fitz.open(pdf_path) as pdf_file:
#         for page_num in range(len(pdf_file)):
#             page = pdf_file.load_page(page_num)
#             pix = page.get_pixmap()
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#             detected_lang = detect(pytesseract.image_to_string(img, lang='mar'))
#             if detected_lang == 'mr':
#                 text += pytesseract.image_to_string(img, lang='mar') + "\n"
#     return text
#
# def preprocess_text(text):
#     tokens = indic_tokenize.trivial_tokenize(text)
#     words = [word for word in tokens if word.isalpha()]  # Remove non-alphabetic tokens
#     return ' '.join(words)
#
# def extract_keywords(text):
#     keywords_set = set()
#     words = Text(text, hint_language_code='mr').words
#     tagged_words = Text(" ".join(words)).pos_tags
#     for word, pos in tagged_words:
#         if pos.startswith('N'):  # Extract only nouns
#             keywords_set.add(word)
#     return keywords_set
#
# def polyglot_marathi_text_summarizer(text, num_sentences=3):
#     text_obj = Text(text, hint_language_code='mr')
#     summary_sentences = text_obj.summary(num_sentences=num_sentences)
#     summary = ' '.join(summary_sentences)
#     return summary
#
# def highlight_keywords(summary, keywords_set):
#     for keyword in keywords_set:
#         summary = summary.replace(keyword, f"**{keyword}**")
#     return summary
#
# if __name__ == "__main__":
#     pdf_path = 'diwali.pdf'
#     extracted_text = extract_text_from_pdf(pdf_path)
#     print(extracted_text)
#
#     summary = summarize(extracted_text)
#     print("Summarized Text using Summa:")
#     print(summary)
#
#     polyglot_summary = polyglot_marathi_text_summarizer(extracted_text)
#     print("Summarized Text using Polyglot:")
#     print(polyglot_summary)
#
#     extracted_keywords = keywords.keywords(extracted_text)
#     print("Extracted Keywords:")
#     print(extracted_keywords)
#
#     processed_text = preprocess_text(extracted_text)
#     extracted_keywords = extract_keywords(processed_text)
#     highlighted_summary = highlight_keywords(polyglot_summary, extracted_keywords)
#
#     print("\nSummarized Text with Highlighted Keywords:")
#     print(highlighted_summary)


# import fitz
# from langdetect import detect
# import pytesseract
# from PIL import Image
# from summa.summarizer import summarize
# from summa import keywords
# from indicnlp.tokenize import indic_tokenize
# import spacy
#
# # Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
#
# # Load the spaCy Marathi model
# nlp = spacy.load('xx_ent_wiki_sm')  # xx_ent_wiki_sm is the language model for multiple languages including Marathi
#
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with fitz.open(pdf_path) as pdf_file:
#         for page_num in range(len(pdf_file)):
#             page = pdf_file.load_page(page_num)
#             pix = page.get_pixmap()
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#             detected_lang = detect(pytesseract.image_to_string(img, lang='mar'))
#             if detected_lang == 'mr':
#                 text += pytesseract.image_to_string(img, lang='mar') + "\n"
#     return text
#
# def preprocess_text(text):
#     tokens = indic_tokenize.trivial_tokenize(text)
#     words = [word for word in tokens if word.isalpha()]  # Remove non-alphabetic tokens
#     return ' '.join(words)
#
# def extract_keywords(text):
#     doc = nlp(text)
#     keywords_set = set()
#     for token in doc:
#         if token.pos_ in ['NOUN', 'PROPN']:  # Extract only nouns and proper nouns
#             keywords_set.add(token.text)
#     return keywords_set
#
# def summarize_text(text, ratio=0.2):
#     summary = summarize(text, ratio=ratio)
#     return summary
#
# def highlight_keywords(summary, keywords_set):
#     for keyword in keywords_set:
#         summary = summary.replace(keyword, f"**{keyword}**")
#     return summary
#
# if __name__ == "__main__":
#     pdf_path = 'diwali.pdf'
#     extracted_text = extract_text_from_pdf(pdf_path)
#     print(extracted_text)
#
#     summary = summarize_text(extracted_text)
#     print("Summarized Text using Summa:")
#     print(summary)
#
#     extracted_keywords = extract_keywords(extracted_text)
#     print("Extracted Keywords using spaCy:")
#     print(extracted_keywords)
#
#     processed_text = preprocess_text(extracted_text)
#     highlighted_summary = highlight_keywords(summary, extracted_keywords)
#
#     print("\nSummarized Text with Highlighted Keywords:")
#     print(highlighted_summary)


#
# import fitz
# from langdetect import detect
# import pytesseract
# from PIL import Image
# from summa.summarizer import summarize
# from summa import keywords
# from indicnlp.tokenize import indic_tokenize
# import spacy
#
# # Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
#
# # Load the spaCy model for Marathi
# nlp = spacy.load('xx_ent_wiki_sm')
#
# # Add the sentencizer component to the pipeline
# if 'sentencizer' not in nlp.pipe_names:
#     nlp.add_pipe('sentencizer')
#
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with fitz.open(pdf_path) as pdf_file:
#         for page_num in range(len(pdf_file)):
#             page = pdf_file.load_page(page_num)
#             pix = page.get_pixmap()
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#             detected_lang = detect(pytesseract.image_to_string(img, lang='mar'))
#             if detected_lang == 'mr':
#                 text += pytesseract.image_to_string(img, lang='mar') + "\n"
#     return text
#
# def preprocess_text(text):
#     tokens = indic_tokenize.trivial_tokenize(text)
#     words = [word for word in tokens if word.isalpha()]  # Remove non-alphabetic tokens
#     return ' '.join(words)
#
# def extract_keywords(text):
#     doc = nlp(text)
#     # Extracting only nouns and proper nouns as keywords
#     keywords_set = {token.text for token in doc if token.pos_ in ('NOUN', 'PROPN')}
#     return keywords_set
#
# def text_summarizer(text, num_sentences=3):
#     doc = nlp(text)
#     # Summarize the document
#     summary_sentences = [sent.text for sent in doc.sents][:num_sentences]
#     summary = ' '.join(summary_sentences)
#     return summary
#
# def highlight_keywords(summary, keywords_set):
#     for keyword in keywords_set:
#         summary = summary.replace(keyword, f"**{keyword}**")
#     return summary
#
# if __name__ == "__main__":
#     pdf_path = 'diwali.pdf'
#     extracted_text = extract_text_from_pdf(pdf_path)
#     print(extracted_text)
#
#     summary = summarize(extracted_text)
#     print("Summarized Text using Summa:")
#     print(summary)
#
#     summarized_text = text_summarizer(extracted_text)
#     print("Summarized Text using spaCy:")
#     print(summarized_text)
#
#     extracted_keywords = keywords.keywords(extracted_text)
#     print("Extracted Keywords using Summa:")
#     print(extracted_keywords)
#
#     processed_text = preprocess_text(extracted_text)
#     extracted_keywords = extract_keywords(processed_text)
#     print("Extracted Keywords using spaCy:")
#     print(extracted_keywords)
#
#     highlighted_summary = highlight_keywords(summarized_text, extracted_keywords)
#
#     print("\nSummarized Text with Highlighted Keywords:")
#     print(highlighted_summary)









#
#
# import sys
# import fitz
# from langdetect import detect
# import pytesseract
# from PIL import Image
# from summa.summarizer import summarize
# from summa import keywords
# from indicnlp.tokenize import indic_tokenize
# import spacy
#
# # Set the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'
#
# # Load the spaCy model for Marathi
# nlp = spacy.load('xx_ent_wiki_sm')
#
# # Add the sentencizer component to the pipeline
# if 'sentencizer' not in nlp.pipe_names:
#     nlp.add_pipe('sentencizer')
#
# def extract_text_from_pdf(pdf_path):
#     text = ""
#     with fitz.open(pdf_path) as pdf_file:
#         for page_num in range(len(pdf_file)):
#             page = pdf_file.load_page(page_num)
#             pix = page.get_pixmap()
#             img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
#             detected_lang = detect(pytesseract.image_to_string(img, lang='mar'))
#             if detected_lang == 'mr':
#                 text += pytesseract.image_to_string(img, lang='mar') + "\n"
#     return text
#
# def preprocess_text(text):
#     tokens = indic_tokenize.trivial_tokenize(text)
#     words = [word for word in tokens if word.isalpha()]  # Remove non-alphabetic tokens
#     return ' '.join(words)
#
# def extract_keywords(text):
#     doc = nlp(text)
#     # Extracting only nouns and proper nouns as keywords
#     keywords_set = {token.text for token in doc if token.pos_ in ('NOUN', 'PROPN')}
#     return keywords_set
#
# def text_summarizer(text, num_sentences=3):
#     doc = nlp(text)
#     # Summarize the document
#     summary_sentences = [sent.text for sent in doc.sents][:num_sentences]
#     summary = ' '.join(summary_sentences)
#     return summary
#
# def highlight_keywords(summary, keywords_set):
#     for keyword in keywords_set:
#         summary = summary.replace(keyword, f"**{keyword}**")
#     return summary
#
# if __name__ == "__main__":
#     if len(sys.argv) != 2:
#         print("Usage: python main.py <pdf_file_path>")
#         sys.exit(1)
#
#     pdf_path = sys.argv[1]
#     extracted_text = extract_text_from_pdf(pdf_path)
#     print(extracted_text)
#
#     summary = summarize(extracted_text)
#     print("Summarized Text using Summa:")
#     print(summary)
#
#     summarized_text = text_summarizer(extracted_text)
#     print("Summarized Text using spaCy:")
#     print(summarized_text)
#
#     extracted_keywords = keywords.keywords(extracted_text)
#     print("Extracted Keywords using Summa:")
#     print(extracted_keywords)
#
#     processed_text = preprocess_text(extracted_text)
#     extracted_keywords = extract_keywords(processed_text)
#     print("Extracted Keywords using spaCy:")
#     print(extracted_keywords)
#
#     highlighted_summary = highlight_keywords(summarized_text, extracted_keywords)
#
#     print("\nSummarized Text with Highlighted Keywords:")
#     print(highlighted_summary)




import sys
import fitz
from langdetect import detect
from PIL import Image
from summa.summarizer import summarize
from summa import keywords
from indicnlp.tokenize import indic_tokenize
import spacy
from inltk.inltk import setup
from inltk.inltk import tokenize
from inltk.inltk import get_similar_sentences

# Setup iNLTK for Marathi
setup('mr')

# Load the spaCy model for Marathi
nlp = spacy.load('xx_ent_wiki_sm')

# Add the sentencizer component to the pipeline
if 'sentencizer' not in nlp.pipe_names:
    nlp.add_pipe('sentencizer')

def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as pdf_file:
        for page_num in range(len(pdf_file)):
            page = pdf_file.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # Extract text using iNLTK OCR
            extracted_text = ' '.join(tokenize(img, "mr"))
            text += extracted_text + "\n"
    return text

def preprocess_text(text):
    tokens = indic_tokenize.trivial_tokenize(text)
    words = [word for word in tokens if word.isalpha()]  # Remove non-alphabetic tokens
    return ' '.join(words)

def extract_keywords(text):
    doc = nlp(text)
    # Extracting only nouns and proper nouns as keywords
    keywords_set = {token.text for token in doc if token.pos_ in ('NOUN', 'PROPN')}
    return keywords_set

def text_summarizer(text, num_sentences=3):
    doc = nlp(text)
    # Summarize the document
    summary_sentences = [sent.text for sent in doc.sents][:num_sentences]
    summary = ' '.join(summary_sentences)
    return summary

def highlight_keywords(summary, keywords_set):
    for keyword in keywords_set:
        summary = summary.replace(keyword, f"**{keyword}**")
    return summary

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python main.py <pdf_file_path>")
        sys.exit(1)

    pdf_path = sys.argv[1]
    extracted_text = extract_text_from_pdf(pdf_path)
    print(extracted_text)

    summary = summarize(extracted_text)
    print("Summarized Text using Summa:")
    print(summary)

    summarized_text = text_summarizer(extracted_text)
    print("Summarized Text using spaCy:")
    print(summarized_text)

    processed_text = preprocess_text(extracted_text)
    extracted_keywords = extract_keywords(processed_text)
    print("Extracted Keywords using spaCy:")
    print(extracted_keywords)

    highlighted_summary = highlight_keywords(summarized_text, extracted_keywords)

    print("\nSummarized Text with Highlighted Keywords:")
    print(highlighted_summary)
