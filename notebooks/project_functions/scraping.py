import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
import time
import re
import pickle

def get_musix_genres(key):
    '''
    Accesses the musiXmatch API using the given API key to obtain a list of all genres
    present in musiXmatch's song database.
    
    key: String, representing the API key to use.
    '''
    
    # Set up url and html parameters
    url = 'https://api.musixmatch.com/ws/1.1/'
    sub_url = 'music.genres.get'
    params = {'apikey': key}
    
    # Make the request
    musix_genres_json = requests.get(url + sub_url, params = params).json()['message']['body']['music_genre_list']
    
    # Parse the json container
    musix_genres = []
    
    for dictionary in musix_genres_json:
        genre = dictionary['music_genre']
        musix_genres.append({'genre_id': genre['music_genre_id'],
                             'genre_name': genre['music_genre_vanity']})
    
    return musix_genres

def get_musix_track_info_by_genre(genres, key, file, id_limit = 5000):
    '''
    Retrieves song metadata from the musiXmatch API based off of genre.
    
    genres: List of strings, representing the genres to retreive from the musiXmatch API.
    key: String, representing the musiXmatch API key to use.
    file: String, representing the file destination where the retrieved data will be written to.
    id_limit: Integer, representing the maximum number of records to pull for each genre from the API.
              Will default to the lower of the maximum number of attainable records for the genre or the id_limit.
              Default: 5000
    '''
    
    # Set up url and html parameters
    url = 'https://api.musixmatch.com/ws/1.1/'
    sub_url = 'track.search'
    
    
    for genre in genres:
        # Set up additional parameters based on genre
        # Retrieve only songs marked as having lyrics in English
        params = {
            'apikey': key,
            'q_track': '*',
            'f_music_genre_id': genre['genre_id'],
            'f_has_lyrics': 1,
            'f_lyrics_language': 'en',
            'page_size': 100,
            'page': 1
        }
        
        # Obtain the maximum number of tracks per genre, then determine page limit for further requests
        num_tracks = requests.get(url + sub_url, params = params).json()['message']['header']['available']
        page_limit = min(num_tracks, id_limit)
        page_max = (page_limit // 100) + 1
        pages = range(1, page_max)
        
        # Status statement
        print('{}: Retrieving {} ids in {} pages'.format(genre['genre_name'], page_limit, page_max - 1))
        
        # Retrieves the song information for each page, appends to file
        for page in pages:
            time.sleep(1.0)
            params['page'] = page
            print('Retrieving page {} of {}'.format(page, page_max - 1))
            tracks = requests.get(url + sub_url, params = params).json()['message']['body']['track_list']
            
            with open(file, 'a') as f:
                track_info = ["{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(track['track']['track_id'],
                                                                        track['track']['track_name'],
                                                                        genre['genre_id'],
                                                                        genre['genre_name'],
                                                                        track['track']['album_id'],
                                                                        track['track']['album_name'],
                                                                        track['track']['artist_id'],
                                                                        track['track']['artist_name'])
                              for track in tracks]

                f.writelines(track_info)
    
        print('Retrieved {} ids for genre {}'.format(page_limit, genre['genre_name']))
    
    return

def clean_titles(original_df):
    '''
    Takes in a Pandas dataframe with a column named title_name, then performs cleaning on the title.
    Returns a new Pandas dataframe with an additional column named clean_title.
    
    original_df: DataFrame, containing column title_name consisting of strings.
    '''
    
    
    df = original_df.copy()
    df['clean_title'] = df.title_name
    # Replace any expressions in parentheses or brackets with spaces
    df.clean_title = df.clean_title.str.replace(r'\(.+\)', r' ', regex = True)
    df.clean_title = df.clean_title.str.replace(r'\[.+\]', r' ', regex = True)
    # Replace any instances of Remix, Remaster, Live, Radio Edit, Extended, and Bonus with spaces
    df.clean_title = df.clean_title.str.replace(r'[Rr]emix(ed)?', r' ', regex = True)
    df.clean_title = df.clean_title.str.replace(r'[Rr]emaster(ed)?', r' ', regex = True)
    df.clean_title = df.clean_title.str.replace(r'[Ll]ive', r' ', regex = True)
    df.clean_title = df.clean_title.str.replace(r'[Rr]adio [Ee]dit', r' ', regex = True)
    df.clean_title = df.clean_title.str.replace(r'[Ee]xtended', r' ', regex = True)
    df.clean_title = df.clean_title.str.replace(r'[Bb]onus( [Tt]rack)?', r' ', regex = True)
    # Replace all Japanese/Chinese characters with spaces
    df.clean_title = df.clean_title.str.replace(r'[一-龠ぁ-んァ-ヴｦ-ﾟ]', r' ', regex = True)
    # Replace any non-alphanumeric, non-Latin, non-apostrophe character with spaces
    df.clean_title = df.clean_title.str.replace(r'/[^A-Za-zFâäèéêëîïôœùûüÿçÀÂÄÈÉÊËÎÏÔŒÙÛÜŸÇÖöǾǿØ̈ø̈\d\']+/',
                                              r' ', regex = True)
    # Convert any non-hyphen dashes and following characters into spaces
    df.clean_title = df.clean_title.str.replace(r'- [\w\W]+', r' ', regex = True)
    # Convert all series of spaces to single space
    df.clean_title = df.clean_title.str.replace(r'\s+', r' ', regex = True)
    # Strip leading and trailing dashes and spaces
    df.clean_title = df.clean_title.str.strip('- ')
    # Drop rows whose titles are now empty
    df = df.loc[df.clean_title != '']
    
    return df.reset_index().drop('index', axis = 1)

def scrape_songlyrics(df, file, title_key = 'title_name'):
    '''
    Scrapes songlyrics.com for song lyrics.
    
    df: DataFrame, containing song metadata, including at minimum columns title_name and artist_name.
    file: String, representing the file location to write scraped data to.
    title_key: String, to represent the column to be used for scraping data. Defaults to title_name.
    '''
    
    # Define base url and url ending for each webscraping attempt
    base_url = 'https://www.songlyrics.com/'
    url_tail = '-lyrics'
    
    # For each song in df, scrape from songlyrics.com
    for row in df.index:
        time.sleep(0.5)
        # Set the title, raw_title (identical to title unless title_key specified), and artist
        title = df.iloc[row][title_key]
        title_raw = df.iloc[row].title_name
        artist = df.iloc[row].artist_name
        
        # Construct the url-fragments for the title and artist, join to base_url and url_tail
        title_string = '-'.join(title.split(' '))
        artist_string = '-'.join(artist.split(' '))
        url = base_url + title_string + '/' + artist_string + url_tail
        
        # Use requests and BeautifulSoup to scape lyrics
        print('Scraping for {} by {}'.format(title, artist))
        r = requests.get(url = url)
        soup = BeautifulSoup(r.text)
        # Lyrics live in <p> object with id songLyricsDiv
        # If not found, print error statement and continue
        try:
            lyrics = soup.find_all(id = "songLyricsDiv")[0].text
        except Exception:
            print('Error retrieving lyrics for {} by {}'.format(title, artist))
            continue
        # If the lyrics begin with 'Sorry, we have no', there are no lyrics for the song
        # Print statement and continue
        if ' '.join(lyrics.split()[0:4]) == 'Sorry, we have no':
            print('No lyrics found for {} by {}'.format(title, artist))
            continue
        # Else, write the lyrics to file
        else:
            lyrics = re.sub('\n', ' ', lyrics)     
            print('Writing lyrics for {} by {} to {}'.format(title, artist, file))
            with open(file, 'a') as f:
                f.write('{}|{}|{}|{}\n'.format(lyrics, title_raw, artist, title))
        
    return