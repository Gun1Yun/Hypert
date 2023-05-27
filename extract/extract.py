import os
import re
import argparse
import pickle
from tqdm import tqdm


import nltk
from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import stopwords
from multiprocessing import Pool

chunk_patterns = r"""NP:{<DT>?<JJ.*>*<NN.*>+}
                    {<NN.*>+}
                """

nounphrase_chunker = nltk.RegexpParser(chunk_patterns)
pos_tagger = PerceptronTagger()
hearst_patterns = []

# Todo : change to args
pattern_path = "./hearst_patterns.txt"
with open(pattern_path, "r") as fr:
    for line in fr:
        reg, hyp = line.strip().split("\t")
        hearst_patterns.append((reg, hyp))

stop_words = set(stopwords.words("english"))


def prepare(raw_text):
    sentences = nltk.sent_tokenize(raw_text.strip())
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    sentences = [pos_tagger.tag(sent) for sent in sentences]
    return sentences


def prepare_chunks(chunks):
    terms = []
    for chunk in chunks:
        label = None
        try:
            label = chunk.label()
        except:
            pass
        if label is None:
            token = chunk[0]
            terms.append(token)
        else:
            np = "NP_" + "_".join([a[0] for a in chunk])
            terms.append(np)
    return " ".join(terms)


def chunk(raw_text):
    sentences = prepare(raw_text.strip())
    all_chunks = []

    for sent in sentences:
        chunks = prepare_chunks(nounphrase_chunker.parse(sent))
        all_chunks.append(chunks)
    all_sentences = []

    for raw in all_chunks:
        sentence = re.sub(r"(NP_\w+ NP_\w+)+", lambda m: m.expand(r"\1").replace(" NP_", "_"), raw)
        all_sentences.append(sentence)
    return all_sentences


def remove_np_term(term):
    return term.replace("NP_", "").replace("_", " ")


def clean_text(text):
    text = text.lower()
    table = text.maketrans({"…": "", "=": "", "%": "", "]": "", "[": ""})
    splits = text.split()
    new_text = [w for w in splits if not w in stop_words]
    text = " ".join(new_text)

    return text.translate(table).strip()


def process(line):
    sentences = line.strip()
    np_tagged_sent = chunk(sentences)
    ext_sentences = []
    for np_sent in np_tagged_sent:
        for hearst_pattern, parser in hearst_patterns:
            matches = re.search(hearst_pattern, np_sent)
            if matches:
                match_str = matches.group(0)
                ext_sentence = remove_np_term(match_str)
                # because extacted file has single line
                ext_sentences.append(ext_sentence)
                break
                # return ext_sentences

    return ext_sentences


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="input file path .txt format")
    parser.add_argument("--output", "-o", default="output.txt", help="output file path")
    parser.add_argument("--core", "-c", default=24, help="number of cores")

    args = parser.parse_args()
    input_path = args.input
    output_path = args.output
    cores = args.core

    msg = f"Check file path\ninput:{input_path}\noutput:{output_path}"
    assert input_path.endswith(".txt"), msg

    with open(input_path, "r") as fr:
        corpus = fr.readlines()

    print(f"Read complete >> {input_path}")

    extracted_sentences = []
    pool = Pool(processes=cores)

    for ext_sent in tqdm(pool.imap(process, corpus), total=len(corpus)):
        extracted_sentences += ext_sent

    pool.close()
    pool.join()

    with open(output_path, "w") as fw:
        fw.write("\n".join(extracted_sentences))

    print(f"Extracted <{len(extracted_sentences)}> sentences!")
    print(f"Saved in >> {output_path}")
