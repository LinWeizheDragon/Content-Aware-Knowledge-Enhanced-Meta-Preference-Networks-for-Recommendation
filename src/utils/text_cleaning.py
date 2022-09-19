"""
text_clearning.py:  
    function for cleaning texts for CBF models
"""

def clean_text(text):
    # text = self.book_dict[self.id2asin[i]]['description']
    # print(text)
    text = text.lower()
    text = text.replace('-', ' ')
    text = text.replace('--', ' ')
    text = text.replace('―', ' ')
    # text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", text)
    return text