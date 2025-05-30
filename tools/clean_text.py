import argparse
import unicodedata
import re
import sys


french_chars = re.compile(r"[^ 'a-zàâæçèéêëîïôùûüÿœ\-]")
spanish_chars = re.compile(r"[^ a-záéíñóúüi\-]")
charsets = {
    'es': spanish_chars,
    'fr': french_chars
}


def _init_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('in_text', help='Path to input text file <utt> <text>')
    parser.add_argument('out_text', help='Path to write text file')
    parser.add_argument('--lang', '-l', choices=['es', 'fr'], required=True, help='Language to prepare')
    parser.add_argument('--lm', action='store_true', help='Don\'t do uttid')

    return parser.parse_args()

def web_domains(text, lang='es'):
    if lang == 'es':
        dot = 'punto'
    elif lang == 'fr':
        dot = 'point'
    pattern = [r'\.(com|net|org)', dot + r' \1']
    return re.sub(pattern[0], pattern[1], text)


def clean_text(text, lang='es'):
    sub_list = [
        ['\^', ''], # Remove ^ character
        ['- ', ''], # Remove - character
        ['\[.*?\]', ''], # Remove things in between [] brackets
        ['\(.*?\)', ''], # Remove things in between () parentheses
        ['\<.*?\>', ''], # Remove things in between <> brackets
        [' -- ', ' '], # Remove double hyphen dashes
        [' - ', ' '], # Remove single hyphen dashes
        ['([0-9]),([0-9])', '$1.$2'],
        ['\.', ' '], # Replace period with one space
        ['[,:;¿?"¡!()\[\]]', ''], # Remove left-over punctuation
        ['^ ', ''],
        ['’', "'"],
    ]

    text = web_domains(text, lang)

    for sub in sub_list:
        text = re.sub(sub[0], sub[1], text)

    text = charsets[lang].sub('', text)

    # Removing multiple spaces -- goes last to ensure any other replacement doesn't cause them
    text = re.sub(' +', ' ', text)
    return text


if __name__ == "__main__":
    args = _init_args()

    outfile = open(args.out_text, 'w', encoding='utf-8')
    with open(args.in_text, 'r', encoding='utf-8') as f:
        buffer_size = 10000
        outlines = []
        for line in f:
            if not args.lm:
                uttid, text = line.split(maxsplit=1)
            else:
                text = line
            text = unicodedata.normalize('NFKC', text)
            text = text.lower()
            text = clean_text(text, lang=args.lang)
            # we want to remove these lines since theres no information we can use
            if len(text.strip()) == 0:
                continue
            if not args.lm:
                outlines.append(f"{uttid} {text}\n")
            else:
                outlines.append(f"{text}\n")
            if len(outlines) > buffer_size:
                outfile.writelines(outlines)
                outlines.clear()
        if len(outlines) > 0:
            outfile.writelines(outlines)

    outfile.close()
