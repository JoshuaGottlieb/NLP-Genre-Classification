import pandas as pd
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import spacy

def extract_lyrics(file):
    '''
    Extracts the lyrics from specified file, returns a Pandas dataframe containing lyric and artist information.
    '''
    
    # Read file
    with open(file, 'r') as f:
        lyrics_raw = f.readlines()
    
    # Extract lyrics that do not contain phrase indicating non-lyrical data
    matched_lyrics = [x for x in lyrics_raw if re.search('We do not have the lyrics', x) is None]
    # Split the lyrics on predetermined pipe delimiter
    matched_lyrics_split = [x.split('|') for x in matched_lyrics]
    # Extract lyrics with have the correct length (4: lyrics, title, artist, clean_title)
    correct_length = [x for x in matched_lyrics_split if len(x) == 4]
    # Write to dataframe and clean up newline characters
    df = pd.DataFrame(correct_length, columns = ['lyrics', 'title_name', 'artist_name', 'clean_title'])
    df.clean_title = df.clean_title.str.replace(r'\n', r'', regex = True)
    
    return df

def spacy_lemmatizer(text, nlp):
    '''
    Helper function to lemmatize tokens based on spaCy lemmatization.
    text: A list containing the tokens to lemmatize.
    nlp: A spaCy nlp object.
    '''
    
    # Rejoin tokens as a string, apply the nlp object
    doc = nlp(' '.join(text))
    
    # Return lemmatized tokens
    return [token.lemma_ for token in doc]

def process_lyrics(df, min_valid_tokens = 15):
    '''
    Processes the lyrics, removing scraping artifacts, non-alphabetic characters, and stopwords.
    Lemmatizes and tokenizes each document, returning the processed lyrics dataframe.
    
    df: A pandas dataframe where each record consists of a single song. Must have column labeled lyrics.
    min_valid_tokens: An optional integer to specify the minimum number of tokens needed to be considered a valid document.
                      Default 15.
    '''
    
    processed_df = df.copy()
    # Remove scraping artifacts
    processed_df.lyrics = processed_df.lyrics.str.replace('^(.+\s{4,})', r' ', regex = True, flags = re.IGNORECASE)
    processed_df.lyrics = processed_df.lyrics.str.replace(r'\w+:', r' ', regex = True)

    # Remove non-alphabetic characters and remove words with length <= 2
    processed_df.lyrics = processed_df.lyrics.str.replace(r'[\W\d]',
                                                          r' ', regex = True)
    processed_df.lyrics = processed_df.lyrics.str.replace(r'\b\w{0,2}\b',
                                                          r' ', regex = True)
    # Convert all lengths of whitespace to single whitespaces, strip outer whitespaces
    processed_df.lyrics = processed_df.lyrics.str.replace(r'\s+',
                                                          r' ', regex = True).str.strip()
    # Lowercase all words
    processed_df.lyrics = processed_df.lyrics.str.lower()
    # Reconvert to list
    processed_df.lyrics = processed_df.lyrics.apply(lambda x: word_tokenize(x))
    # Remove stopwords
    stop_words = stopwords.words('english')
    processed_df.lyrics = processed_df.lyrics.apply(lambda x:\
                                            [word for word in x if word not in stop_words])
    # Keep records with a minimum number of tokens
    processed_df = processed_df.loc[processed_df.lyrics.apply(lambda x: len(x)) >= min_valid_tokens]
    # Lemmatization
    nlp = spacy.load('en_core_web_sm')
    processed_df.lyrics = processed_df.lyrics.apply(lambda x: spacy_lemmatizer(x, nlp))

    return processed_df

def extract_and_process_lyrics(files, min_valid_tokens = 15):
    '''
    Helper function to extract lyrics and process lyrics from a given set of files. Returns the processed lyrics dataframe.
    files: A string representing a file or a list of strings representing a set of files to extract lyrics from.
    min_valid_tokens: An optional integer to specify the minimum number of tokens needed to be considered a valid document.
                      Passed on to the process_lyrics function. Default 15.
    '''
    
    # Extract and process lyrics from file or files. If files is not a valid type, print error message.
    if type(files) == str:
        df = extract_lyrics(files)
        processed_df = process_lyrics(df, min_valid_tokens)
        
        return processed_df
    elif type(files) == list:
        dfs = [process_lyrics(extract_lyrics(file), min_valid_tokens) for file in files]
        processed_df = pd.concat(dfs, axis = 0)
        
        return processed_df
    else:
        print('Must input a single file string or a list of file strings.')
    return
    
def stitch_lyrics_and_metadata_frames(lyrics_df, metadata_df):
    '''
    Helper function to stitch a lyrics dataframe and a metadata dataframe together.
    lyrics_df: A Pandas dataframe consisting of records containing lyrics, title_name, and artist_name columns
    metadata_df: A Pandas dataframe consisting of records containing title_name and artist_name columns, at minimum
    '''
    
    # Merge frames on mutual keys, keeping only matches. Drop unnecessary columns.
    merged_df = metadata_df.merge(lyrics_df, how = 'inner', on = ['title_name', 'artist_name'])
    merged_df = merged_df.drop(['title_id', 'genre_id', 'album_id', 'album_name', 'artist_id'], axis = 1)
    return merged_df

def process_data(lyric_files, metadata_file, dest_file, min_valid_tokens = 15):
    '''
    A function which performs the entire preprocessing steps for the NLP task.
    Writes a parquet file containing the merged dataframe and returns the merged dataframe.
    lyric_files: A string or list of strings denoting the file(s) containing lyric information.
    metadata_file: A string denoting the file containing metadata information.
    dest_file: A string denoting the destination to write the resultant parquet file to.
    min_valid_tokens: An optional integer to specify the minimum number of tokens needed to be considered a valid document.
                      Passed on to the extract_and_process_lyrics function. Default 15.
    '''
    
    # Extract and process lyrics files
    lyrics_df = extract_and_process_lyrics(lyric_files, min_valid_tokens)
    
    # Extract metadata from file
    columns = ['title_id', 'title_name', 'genre_id', 'genre_name', 'album_id', 'album_name', 'artist_id', 'artist_name']
    metadata_df = pd.read_csv(metadata_file, sep = '\t', header = None)
    metadata_df.columns = columns
    
    # Merge lyrics and metadata frames
    merged_df = stitch_lyrics_and_metadata_frames(lyrics_df, metadata_df)
    merged_df = merged_df.drop_duplicates(subset = ['title_name', 'artist_name'])
    
    # Write to file
    merged_df.to_parquet(dest_file)
    
    return merged_df