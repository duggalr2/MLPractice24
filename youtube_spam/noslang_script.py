import string
from robobrowser.browser import RoboBrowser


alphabet = list(string.ascii_lowercase)
browser = RoboBrowser()
for letter in alphabet:
    browser.open('https://www.noslang.com/dictionary/' + letter)
    x = browser.find_all('div', class_='dictionary-word')
    for div in x:
        a = div.find('abbr')
        english_word = a['title']
        b = a.find('span')
        spam_word = b.text
        spam_word = spam_word.replace(':', '')
        with open('word', 'a') as f:
            f.write(spam_word + '=>' + english_word + '\n')
