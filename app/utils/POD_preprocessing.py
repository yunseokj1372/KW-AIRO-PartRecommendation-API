import nltk
import pandas as pd
from nltk.corpus import words
from Levenshtein import distance
import re
from nltk.tokenize import MWETokenizer

def splitter(txt):
    # TODO: check temp comment
    """
    Splits string accodring to multiple delimeters: ['/', ';', ',', ' ']
    
    Keyword arguments:
    txt -- string input text
    """

    # function : takes in string text and split accordingly with the delimiter
    return re.split('/|;|,| ',txt)

def mwe_splitter(txt, mwe):
    # TODO: check temp comment
    """
    Splits tokens based on multiple delimeters: ['/', ';', ',', ' '], merges specified tuples in mwe

    Keyword arguments:
    txt -- string input text
    mwe -- sequence of multi-word expressions to be merged, where each mwe is a sequence of strings
    """

    # function : multi word tokenization splitter
    # mwe : list of tuples for multi word
    tk = MWETokenizer(mwes = mwe, separator = ' ')
    return tk.tokenize(re.split('/|;|,| ',txt))

def get_new_list_value(row,new_list_dict,column):
    # TODO: check temp comment
    """
    Retrieves wid value for row in dataframe and returns the corresponding value in input dictionary.
    
    Keyword arguments:
    row -- row of dataframe
    new_list_dict -- dictionary with values corresponding to row wids
    column -- column of dataframe
    """
    index = row['wid']
    return new_list_dict.get(index, row[column])


def sentence_detector(df):
    # TODO: check temp comment
    """
    Returns dictionary of part keywords containing formatted sentences from the 'NOTE' column if they contain at lease one of these words: ['order', 'need', 'request']

    Keyword arguments:
    df -- dataframe
    """

    # function : creates a separate column at a dataframe that separates the sentences and extracts only the sentence with the word 'order', 'need', 'request'
    df['NOTE'] = df['NOTE'].str.lower()
    df['sent_divider'] = df.apply(lambda row: nltk.sent_tokenize(str(row['NOTE'])), axis=1)
    tkts = list(set(df['ID']))
    final_dict = {}
    for tkt in tkts:
        temp_df = df[df['ID'] == tkt]
        rank = list(set(temp_df['WID_ORDER']))
        for i in rank[:-1]:
            wid = temp_df['WID'][temp_df['WID_ORDER'] == i].iloc[0]
            key = str(tkt)+'_'+str(i)
            note = temp_df['sent_divider'][temp_df['WID_ORDER'] == i].iloc[0]
            final_sent = ''
            for sent in note:
                if 'order' in sent:
                    final_sent = sent
                    final_sent =final_sent.split("order",1)[1]
                    break
                elif 'need' in sent:
                    final_sent = sent
                    final_sent =final_sent.split("need",1)[1]
                    break
                elif 'request' in sent:
                    final_sent = sent
                    final_sent =final_sent.split("request",1)[1]
                    break

            if key not in final_dict.keys() and final_sent != '':
                add = f'''{final_sent}'''
                add = add.replace('\n', ' ')
                final_dict[key] = {}
                final_dict[key]['target_note'] = add.strip('.?,!')
                final_dict[key]['wid'] = wid
    return final_dict


def sentence_pred_detector(df):
    # TODO: check temp comment
    """
    Returns dictionary of part keywords containing formatted sentences fron the 'NOTE' column, as well as indicating whether the note has any of the following special instructions: ['retain', 'keep', 'resend', 'hold part', 'bring back'] if the note contains at least one of these words: ['order', 'need', 'request']

    Keyword arguments:
    df -- dataframe
    """
    # function : creates a separate column at a dataframe that separates the sentences and extracts only the sentence with the word 'order', 'need', 'request'
    df['NOTE'] = df['NOTE'].str.lower()
    df['sent_divider'] = df.apply(lambda row: nltk.sent_tokenize(str(row['NOTE'])), axis=1)

    highest_ranks_df = df.sort_values(by='WID_ORDER', ascending=False).groupby('ID').head(1)
    wids = list(set(highest_ranks_df['WID']))
    final_dict = {}
    for wid in wids:
        prev = 'NO'
        temp_df = df[df['WID'] == wid]
        key = str(wid)
        note = temp_df['sent_divider'].iloc[0]
        tkt = temp_df['ID'].iloc[0]
        final_sent = ''
        for sent in note:
            if 'retain' in sent:
                prev = 'YES'
            elif 'keep' in sent:
                prev = 'YES'
            elif 'resend' in sent:
                prev = 'YES'
            elif 'hold part' in sent:
                prev = 'YES'
            elif 'bring back' in sent:
                prev = 'YES'

            if 'order' in sent:
                final_sent = sent
                final_sent =final_sent.split("order",1)[1]
                break
            elif 'need' in sent:
                final_sent = sent
                final_sent =final_sent.split("need",1)[1]
                break
            elif 'request' in sent:
                final_sent = sent
                final_sent =final_sent.split("request",1)[1]
                break
            

        if key not in final_dict.keys() and final_sent != '':
            add = f'''{final_sent}'''
            add = add.replace('\n', ' ')
            final_dict[key] = {}
            final_dict[key]['target_note'] = add.strip('.?,!')
            final_dict[key]['tktno'] = tkt
            final_dict[key]['wid'] = wid
            final_dict[key]['prev'] = prev

    return final_dict


def part_filter(df,final_dict):
    # TODO: check temp comment
    """
    Filters out non-parts from input dictionary, returns dictionary that aligns the next workorder part number with the workorder

    Keyword arguments:
    df -- dataframe
    final_dict -- dictionary containing keywords extracted from notes
    """
    # function : filters out non parts, then it creates a dictionary that aligns the next workorder part number with the workorder
    remove_parts = []
    all_parts = list(set(df['PARTNO']))
    for i in all_parts:
        i = str(i)
        if 'SQ' in i:
            remove_parts.append(i)
        elif 'JOB' in i:
            remove_parts.append(i)
        elif 'CSP' in i:
            remove_parts.append(i)


    for id in final_dict.keys():
        tkt, wid_order = id.split('_')
        # tkt = int(tkt)
        wid_order = int(wid_order)+1
        final_parts = list(set(list(df['PARTNO'][(df['ID'] == tkt)& (df['WID_ORDER'] == wid_order)])) - set(remove_parts) )
        final_dict[id]['parts'] = final_parts


    return final_dict


def dataframe_converter(final_dict):
    # TODO: check temp comment
    """
    Converts input dictionary to data frame and applies splitter function to notes column

    Keyword arguments:
    final_dict -- dictionary containing keywords extracted from notes
    """
    # function : splits the sentence with the splitter function (in the future we may use multi word tokenization function)
    final_df = pd.DataFrame(final_dict).transpose()
    final_df['word_divider'] = final_df.apply(lambda row: splitter(row['target_note'])[1:], axis=1)
    return final_df



def mwe_dataframe_converter(final_dict,mwe):
    # TODO: check temp comment
    """
    Converts input dictionary to data frame and applies mwe splitter function to notes column

    Keyword arguments:
    final_dict -- dictionary containing keywords extracted from notes
    mwe -- sequence of multi-word expressions to be merged, where each mwe is a sequence of strings
    """
    # function : splits the sentence with the mwe_splitter function 
    final_df = pd.DataFrame(final_dict).transpose()
    final_df['word_divider'] = final_df.apply(lambda row: mwe_splitter(row['target_note'],mwe)[1:], axis=1)
    return final_df

def dictionary_filter(tokens):
    # TODO: check temp comment
    """
    Sets token to '0000000000000' if it is not recognized as an English word

    Keyword arguments:
    tokens -- list of tokens
    """
    english_words = set(words.words())
    return {token: '0000000000000' for token in tokens if token not in english_words}

def dictionary_adder(df):
    # TODO: check temp comment
    """
    Creates new column 'wrong' and applies English word filter

    Keyword arguments:
    df - dataframe
    """
    df['wrong'] = df['word_divider'].apply(dictionary_filter)
    return df

def levenshtein_distance(df, part_corpus):
    # TODO: check temp comment
    """
    Calculates Levenshtein distance between keywords and parts

    Keyword arguments:
    df -- dataframe
    part_corpus -- list of parts
    """
    for i,row in enumerate(df['wrong']):
        if row:
            for key in row.keys():
                for part in part_corpus:
                    lev = distance(key,part)
                    if lev <= 1:
                        df['wrong'].iloc[i][key] = part
    return df

def mwe_levenshtein_distance(df,mwe,part_corpus):
    # TODO: check temp comment
    """
    Calculates Levenshtein distance between mwe keywords and parts
    
    Keyword arguments:
    df -- dataframe
    mwe -- sequence of multi-word expressions to be merged, where each mwe is a sequence of strings
    part_corpus -- list of parts
    """
    for mw in mwe:
        mword = ' '.join(mw)
        part_corpus.append(mword)
    for i,row in enumerate(df['wrong']):
        if row:
            for key in row.keys():
                for part in part_corpus:
                    lev = distance(key,part)
                    if lev <= 1:
                        df['wrong'].iloc[i][key] = part
    return df


def levenshtein_json(df, part_json):
    # TODO: check temp comment
    """
    Calculates Levenshtein distance between keywords and parts from JSON file
    
    Keyword arguments:
    df -- dataframe
    part_json -- JSON file of parts
    """
    for i,row in enumerate(df['wrong']):
        if row:
            for key in row.keys():
                for part in part_json.keys():
                    lev = distance(key,part)
                    if lev <= 1:
                        df['wrong'].iloc[i][key] = part
    return df


def reverse_mapping(df):
    # TODO: check temp comment
    """
    Extracts list of values from dictionaries in 'word_divider' column of input dataframe

    Keyword arguments:
    df -- dataframe
    """
    for i, rev in enumerate(df['wrong']):
        df['word_divider'].iloc[i] = list(map( lambda x: rev.get(x,x), df['word_divider'].iloc[i]))
    return df


def corpus_check(df, part_corpus):
    # TODO: check temp comment
    """
    Adds keywords from 'word_divider' column to a list in 'note_parts' column if the keyword is in the part corpus
    
    Keyword arguments:
    df -- dataframe
    part_corpus -- list of parts
    """
    # corpus: list of words for technician parts recommendation (ex. tray)
    df['note_parts'] = [[] for _ in range(len(df))]
    for i,row in enumerate(df['word_divider']):
        for word in row:
            if word in part_corpus:
                df['note_parts'].iloc[i].append(word)

    return df

def mwe_corpus_check(df,mwe,part_corpus):
    # TODO: check temp comment
    """
    Adds keywords from 'word_divider" column to a list in 'note_parts' column if the keyword is in the part corpus, which includes mwe tokens
    
    Keyword arguments:
    df -- dataframe
    mwe -- sequence of multi-word expressions to be merged, where each mwe is a sequence of strings
    part_corpus -- list of parts
    """
    df['note_parts'] = [[] for _ in range(len(df))]

    for mw in mwe:
        mword = ' '.join(mw)
        part_corpus.append(mword)

    for i,row in enumerate(df['word_divider']):
        for word in row:
            if word in part_corpus:
                df['note_parts'].iloc[i].append(word)

    return df

def json_check(df, part_json, accurate,original_df):
    # TODO: check temp comment
    """
    
    
    Keyword arguments:
    df -- dataframe
    part_json -- JSON file of parts
    accurate -- boolean indicating accuracy of keyword extraction in 'word_divider'
    original_df -- unedited input dataframe
    """
    df['corresponding_parts'] = [[] for _ in range(len(df))]
    df['note_parts'] = [[] for _ in range(len(df))]
    for i, row in enumerate(df['word_divider']):
        row = set(row)
        for word in row:
            if word in part_json.keys():
                df['note_parts'].iloc[i].append(word)
                df['corresponding_parts'].iloc[i].append(part_json[word][0])
    if accurate:
        df = df.drop(columns = ['word_divider', 'wrong'])
    else:
        df = df.drop(columns = ['word_divider'])
    

    # adding previous parts to each df
    wids = list(set(df['wid']))

    convert_cpart = {}
    convert_npart = {}

    for w in wids:
        w_df = df[df['wid'] ==w]
        prev_det = w_df['prev'].iloc[0]
        if prev_det == 'YES':
            prev_parts = list(original_df['PARTNO'][original_df['WID']==w])
            note_det  = ['previous']*int(len(prev_parts))

            c_part = df['corresponding_parts'][df['wid']==w].iloc[0]
            n_part =  df['note_parts'][df['wid']==w].iloc[0]
            c_part.extend(prev_parts)
            n_part.extend(note_det)

            convert_cpart[w] = c_part
            convert_npart[w] = n_part
    # Conditionally update the 'List_Column' based on the 'Decision_Column'
    df['corresponding_parts'] = df.apply(get_new_list_value, args = (convert_cpart,'corresponding_parts',), axis=1)
    df['note_parts'] = df.apply(get_new_list_value, args = (convert_npart,'note_parts',), axis=1)
    return df


def partdesc_match(df, key,counter):
    # TODO: check temp comment
    """
    Returns part number and part description if the keyword is in the part description in the input dataframe
    
    Keyword arguments:
    df -- dataframe
    key -- keyword
    counter -- collection of dictionaries containing part numbers
    """
    upper_p = key.upper()
    proper_partno = 'unclear'
    for p_count in counter[key].keys():
        partdesc = df['PARTDESC'][df['PARTNO'] == p_count].iloc[0]
        if upper_p in partdesc:
            proper_partno = p_count
            break
        else:
            partdesc = None
    return proper_partno, partdesc

### Part Converter

def part_converter(instruction, POD_json):
    # TODO: check temp comment
    """
    Creates dictionary of keys from intrtuction keys dictionary and values from keywords from POD_json file
    
    Keyword arguments:
    instruction -- 
    POD_json -- 
    """
    # returns : dictionary that has the converted version of instruction json file
    
    instruct_keys = instruction.keys()
    outcome = {}
    
    for key in instruct_keys:
        instruct_parts = instruction[key]
        parts = {}
        for keyword in instruct_parts:
            try:
                parts[keyword] = POD_json[keyword][0]
                outcome[key] = parts
            
            except:
                continue
        
    
    return outcome


###

def model_construction(df, part_corpus):
    print('Model Construction')
    part_corpus = list(part_corpus['part'])
    print('---Sentence Detection Pipeline---')
    final_dict = sentence_detector(df)
    print('---Part Filter Pipeline---')
    final_dict = part_filter(df,final_dict)
    print('---Dataframe Converter Pipeline---')
    final_df = dataframe_converter(final_dict)
    print('---Dictionary Filter Pipeline---')
    final_df = dictionary_adder(final_df)
    print('---Levenshtein Distance Pipeline---')
    final_df = levenshtein_distance(final_df, part_corpus)
    print('---Reverse Mapping Pipeline---')
    final_df = reverse_mapping(final_df)
    print('---Corpus Check Pipeline---')
    final_df = corpus_check(final_df,part_corpus)
    return final_df

def model_prediction(df,part_json, accurate = True):
    print('Model Prediction')
    print('---Sentence Detection Pipeline---')
    final_dict = sentence_pred_detector(df)
    print('---Dataframe Converter Pipeline---')
    final_df = dataframe_converter(final_dict)
    if accurate:
        print('---Dictionary Filter Pipeline---')
        final_df = dictionary_adder(final_df)
        print('---Levenshtein Distance Pipeline---')
        final_df = levenshtein_json(final_df, part_json)
        print('---Reverse Mapping Pipeline---')
        final_df = reverse_mapping(final_df)
    print('---JSON Check Pipeline---')
    final_df = json_check(final_df,part_json, accurate, original_df = df)
    return final_df


def mwe_model_construction(df, part_corpus):
    print('Model Construction')
    part_corpus['tuple'] = part_corpus.apply(lambda row: tuple(splitter(row.iloc[0])), axis=1)
    part_corpus['length'] = part_corpus.apply(lambda row: len(row['tuple']), axis = 1)
    one_word = list(part_corpus['part'][part_corpus['length'] ==1])
    mwe = list(part_corpus['tuple'][part_corpus['length'] >1])
    print('---Sentence Detection Pipeline---')
    final_dict = sentence_detector(df)
    print('---Part Filter Pipeline---')
    final_dict = part_filter(df,final_dict)
    print('---Dataframe Converter Pipeline---')
    final_df = mwe_dataframe_converter(final_dict,mwe)
    print('---Dictionary Filter Pipeline---')
    final_df = dictionary_adder(final_df)
    print('---Levenshtein Distance Pipeline---')
    final_df = mwe_levenshtein_distance(final_df,mwe, one_word)
    print('---Reverse Mapping Pipeline---')
    final_df = reverse_mapping(final_df)
    print('---Corpus Check Pipeline---')
    final_df = mwe_corpus_check(final_df,mwe, one_word)
    return final_df


def mwe_model_prediction(df,part_json, accurate = True):
    print('Model Prediction')
    part_corpus['tuple'] = part_corpus.apply(lambda row: tuple(splitter(row.iloc[0])), axis=1)
    part_corpus['length'] = part_corpus.apply(lambda row: len(row['tuple']), axis = 1)
    one_word = list(part_corpus['part'][part_corpus['length'] ==1])
    mwe = list(part_corpus['tuple'][part_corpus['length'] >1])
    print('---Sentence Detection Pipeline---')
    final_dict = sentence_pred_detector(df)
    print('---Dataframe Converter Pipeline---')
    final_df = mwe_dataframe_converter(final_dict,mwe)
    if accurate:
        print('---Dictionary Filter Pipeline---')
        final_df = dictionary_adder(final_df)
        print('---Levenshtein Distance Pipeline---')
        final_df = mwe_levenshtein_json(final_df, part_json)
        print('---Reverse Mapping Pipeline---')
        final_df = reverse_mapping(final_df)
    print('---JSON Check Pipeline---')
    final_df = json_check(final_df,part_json, accurate)
    return final_df
    