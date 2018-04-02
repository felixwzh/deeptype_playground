from operator import itemgetter
import numpy as np
import re
import MeCab

def mention_index(ts, ss, sep=' '):
    con = [[] for t in ts]
    fin = [False for t in ts]
    for i, s in enumerate(ss):
        for j, t in enumerate(ts):
            if fin[j] is True:
                continue            
            if con[j]:
                tmp = itemgetter(*con[j])(ss)
                if isinstance(tmp, tuple):
                    tmp = list(tmp) + [ss[i]]
                elif isinstance(tmp, str):
                    arr = []
                    arr.append(tmp)
                    tmp = arr + [ss[i]]
                ndl = sep.join(tmp)
            else:
                ndl = ss[i]
            if ndl == t:
                con[j].append(i)
                fin[j] = True
            elif t.startswith(ss[i]) and not t.startswith(ndl):
                con[j] = [i]
                if ss[i] == t:
                    fin[j] = True
            elif t.startswith(ndl):
                con[j].append(i)
            else:
                con[j] = []
    return con

def is_all_none(arr):
    for a in arr:
        if a is not None:
            return False
    return True

def index_in_sentence(con):
    con_s = con[:]
    stack_list = []
    while(not is_all_none(con_s)):
        appeared_value = []
        appeared_index = []
        stack = []
        for i, c in enumerate(con_s):
            if c is None:
                continue
            if np.sum(np.in1d(c, appeared_value)) > 0:
                continue
            else:
                stack.append(c)
                appeared_index.append(i)
                for v in c:
                    appeared_value.append(v)
        for index in appeared_index:
            con_s[index] = None
        stack_list.append(stack)
    return stack_list

def build_sentence(ss, sl):
    stack = sl[:]
    stack.sort()
    stack = [x for x in stack if x]
    sentence = []
    c = stack.pop(0)
    tmp = []
    for i, s in enumerate(ss):
        if i in c:
            tmp.append(i)
            if c[-1] == i:
                sentence.append(tmp)
                tmp = []
                if stack:
                    c = stack.pop(0)
                    
        else:
            sentence.append([i])
    return sentence

def build_sentences(ss, slist, sep=' '):
    sentences = []
    for sl in slist:
        line = []
        sentence = build_sentence(ss, sl)
        for word in sentence:
            w = itemgetter(*word)(ss)
            if isinstance(w, tuple):
                line.append(sep.join(list(w)))
            elif isinstance(w, str):
                line.append(w)
        sentences.append(line)
    return sentences

def mecab_tokenize(sentence):
    tagger = MeCab.Tagger("-Owakati")
    return tagger.parse(sentence).split()

def run(ts, ss, sep=' '):
    return build_sentences(ss, index_in_sentence(mention_index(ts, ss, sep=sep)), sep=sep)

def replace_mulalpha(sentence):
    mul_alpha = list('ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ')
    sin_alpha = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz')
    characters = list(sentence)
    stack = []
    for c in characters:
        if c in mul_alpha:
            stack.append(sin_alpha[mul_alpha.index(c)])
        else:
            stack.append(c)
    return ''.join(stack)           

def ja_tokenize(sentence, ts, tokenize=mecab_tokenize):
    sentence = replace_mulalpha(sentence).lower()
    ss = tokenize(sentence)
    result = []
    for d in run(ts, ss, sep=''):
        result += d
    return result

def en_tokenize(sentence, ts):
    ss = sentence.lower().split()
    result = []
    for d in run(ts, ss):
        result += d
    return result

if __name__ == '__main__':
    #ts = ['street', 'street musician', 'musician at japan', 'japan']
    #ss = 'he is a street musician at japan'.split()
    ts = ['ボディー', 'ボディービルダー', '日本','日本人', '大学', '大学生', '日本人大学生']
    ss = '彼は ボディー ビルダー を やって いる 一人 の 日本 人 大学 生 です'.split()
    sentence = '彼はボディービルダーをやっている一人の日本人大学生です'

    con = mention_index(ts, ss, '')
    slist = index_in_sentence(con)

    print(con)
    print(slist)
    print(build_sentence(ss, slist[0]))
    print(build_sentences(ss, slist, ''))
    print(ja_tokenize(sentence, ts))
