import numpy as np
import json
from collections import Counter

from pycocoevalcap.bleu.bleu import Bleu

def count_novel(path, all_train_captions):
    captions = json.load(open(path, 'r'))
    all_captions = []
    for item in captions:
        all_captions.append(item['caption'])
    count_captions = dict(Counter(all_captions))
    count_captions = {k: v for k, v in sorted(count_captions.items(), key=lambda item: item[1], reverse=True)}
    print(len(count_captions))
    novel_caption_cnt = len(all_captions)
    for cap in all_captions:
        if cap in all_train_captions:
            novel_caption_cnt -= 1
    print('novel', novel_caption_cnt, len(all_captions), novel_caption_cnt / len(all_captions))


def count_unique(path):
    with open(path, 'r') as fr:
        res = json.load(fr)
    all_captions = []
    captions = {}
    unique_captions = {}
    unique_length_level_num = 0
    for i in range(1, 5):
        captions[str(i)] = []
        for k, v in  res[str(i)].items():
            captions[str(i)].append(v[0]['caption'])
        unique_captions[str(i)] = list(set(captions[str(i)]))
        unique_length_level_num += len(unique_captions[str(i)])
        print(len(unique_captions[str(i)]))
        all_captions += captions[str(i)]
    print('unique', len(all_captions), len(set(all_captions)), unique_length_level_num)


def compute_div_n(caps,n=1, return_all=False):
    '''
    Div-n is the ratio of distinct n-grams per caption
    to the total number of words generated per set of diverse captions.
    definition refer to https://arxiv.org/pdf/2011.00966.pdf
    which should be originated from https://arxiv.org/pdf/1510.03055.pdf
    calculating the number of distinct unigrams and bigrams in generated responses.
    The value is scaled by total number of generated tokens to avoid favoring long sentences
    :param caps:
             {'391895': ['a man that is sitting on a motorcycle.',
                        'a man riding a motorcycle down a dirt road.',
                        'a man is riding a motorcycle on a dirt road.',
                        'a man in a red shirt is riding a motorcycle on a dirt road.',
                        'a man on a motorcycle with a bag on his shoulder on a road with mountains in the background.'],
             '60623': ['a woman eating a dessert with a spoon.',
                       'a woman sitting at a table with a spoon.',
                       'a woman holding a spoon over a cup of coffee.',
                       'a woman sitting at a table eating something from a cup with a spoon.',
                       'a woman sitting at a table with a cup of coffee and a spoon in front of her face.'],
             ...}
    :param n:
    :return:
    '''

    def find_ngrams(input_list, n):
        return zip(*[input_list[i:] for i in range(n)])

    def find_ngrams_len(input_list, n):
        return len([input_list[i:] for i in range(n)][-1])

    aggr_div = []
    aggr_div_ng = []
    for k in caps:
        all_ngrams = set()
        lenT = 0.  # total number of words generated per set of diverse captions
        len_ng = 0.  # the total number of n-grams in a set of generated sentences.
        for c in caps[k]:
            # tkns = c.split()
            tkns = [e for e in c.replace('.','').replace('\n','').split(' ') if e != '']
            lenT += len(tkns)
            ng = find_ngrams(tkns, n)
            all_ngrams.update(ng)
            len_ng += find_ngrams_len(tkns, n)
        aggr_div.append(float(len(all_ngrams)) / (1e-6 + float(lenT)))
        aggr_div_ng.append(float(len(all_ngrams)) / (1e-6 + float(len_ng)))
    if return_all:
        return np.array(aggr_div), np.array(aggr_div_ng)
    return np.array(aggr_div).mean(), np.array(aggr_div_ng).mean()


def lines_to_ngrams(lines, n=3):
    ngrams = []
    all_words = []
    for s in lines:
        words = [e for e in s.replace('.','').replace('\n','').split(' ') if e != '']
        ngrams.append([tuple(words[i:i + n]) for i in range(len(words) - n + 1)])
        all_words.append(words)
    return [ngrams, all_words]

def normalized_unique_ngrams(input):
    """
    Calc the portion of unique n-grams out of all n-grams.
    :param ngram_lists: list of lists of ngrams
    :return: value in (0,1]
    """
    ngram_lists, word_lists = input
    ngrams = [item for sublist in ngram_lists for item in sublist]  # flatten
    words = [item for sublist in word_lists for item in sublist]  # flatten
    div_ngram = len(set(ngrams)) / len(ngrams) if len(ngrams) > 0 else 0.
    div_word = len(set(ngrams)) / len(words) if len(words) > 0 else 0.
    return [div_ngram, div_word]


def compute_div_n2(result, n, return_all=False):
    '''
    Equals to compute_div_n(), two implementations
    '''
    aggr_div = []
    for k, v in result.items():
        score = normalized_unique_ngrams(lines_to_ngrams(v, n=n))
        aggr_div.append(score)
    if return_all:
        div_ng = [item[0] for item in aggr_div]
        div = [item[1] for item in aggr_div]
    else:
        div_ng, div = np.array(aggr_div).mean(axis=0)
    return div, div_ng


def get_div_n(path):
    result = json.load(open(path, 'r'))
    if len(result) == 4:  # 4 levels format
        res = {}
        for k, v in result.items():
            for imgid, cap in v.items():
                if imgid in res:
                    res[imgid].append(cap[0]['caption'])
                else:
                    res[imgid] = [cap[0]['caption']]
        result = res
    div_1, div_ng_1 = compute_div_n2(result, 1)
    div_2, div_ng_2 = compute_div_n2(result, 2)
    return div_1, div_2


def get_mbleu(capsById, n_caps_perimg=5, return_all=False):
    scorer = Bleu(4)
    all_scrs = []
    scrperimg = np.zeros((n_caps_perimg, len(capsById)))

    for i in range(n_caps_perimg):
        tempRefsById = {}
        candsById = {}
        for k in capsById:
            tempRefsById[k] = capsById[k][:i] + capsById[k][i + 1:]
            candsById[k] = [capsById[k][i]]

        score, scores = scorer.compute_score(tempRefsById, candsById)
        all_scrs.append(score)
        scrperimg[i, :] = scores[1]

    all_scrs = np.array(all_scrs)
    if return_all:
        mbleu = all_scrs
    else:
        mbleu = all_scrs.mean(axis=0)
    return mbleu
