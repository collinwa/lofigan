import sys
import pydub
import pytube
import re
import os 
import requests

def url_list(link):
    r = requests.get(link)
    all_urls = re.findall('watch\?v=[a-zA-Z0-9_-]*', r.text)
    return all_urls

def retrieve_videos(all_urls, out_dir):
    youtube_string = 'www.youtube.com/%s'
    for i, suffix in enumerate(all_urls):
        this_song = youtube_string % suffix
        track_name = 'track_%i' % i 
        if not os.path.exists(out_dir): os.makedirs(out_dir)
        try:
            yt = pytube.YouTube(this_song)
            yt.streams.filter(only_audio=True).first().download(out_dir, filename=track_name)
            print('Downloaded Track %i' % i, end='\r', flush=True) 
        except:
            print('Skipping %i' % i)
            continue
 

if __name__ == '__main__':
    playlist_link = sys.argv[1]
    out_dir = sys.argv[2]
    all_urls = url_list(playlist_link)
    retrieve_videos(all_urls, out_dir) 

