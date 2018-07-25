
# Using Reddit's API for Predicting Comments

## Executive Summary:

Reddit is an online content platform which recently surpassed facebook as the third most trafficed site in the US. Because Reddit has the structure of a community-driven platform, and its users are more interested in user-generated content than in paid advertizements, companies have strong incentives to create content that appears user-generated. 

The site consists of posts on which users can vote or comment. Each post gets a score based on the number of "upvotes" and "downvotes" it receives, and each post pertains to a particular "subreddit", a page organized around a particular topic. Posts from a variety of subreddits are aggregated to the front page (reddit.com), and can be sorted by 'new', 'hot', 'trending', and 'top'.

This presents an important question for advertizers and their opponents: can reddit be gamed? In general, what influences the popularity of a post?

In this analysis I attempt to predict whether a reddit post gets an above- or below-average amount of interaction (as measured by number of comments), using natural language processing and classification models.

I find that the most important predictors of a post's success are its score and the length of time since it was posted. The contributions of individual words added very little explanatory power compared to a model which predicted comment levels based on a post's subreddit, length of time online, and score. This strongly suggests that it is difficult to game reddit based solely on picking good titles. Most of a post's success appears to depend on its quality as judged by reddit users (and bots). 

### OUTLINE:

In this project I attempt to figure out what information can help predict the number of comments a reddit post receives. The two major steps are:
1. Scraping data from reddit into a usable format.
2. Building a model using that data to predict a post's number of comments, and interpreting that model. 

My problem statement is: _What characteristics of a post on Reddit contribute most to the number of comments?_

The source of data is the 'hot' tab of reddit's homepage (https://www.reddit.com/hot). I'll acquire 5 pieces of information about each thread:
1. The title of the thread
2. The subreddit that the thread corresponds to
3. The length of time it has been up on Reddit
4. The post's score (a function of upvotes and downvotes)
5. The number of comments on the thread

Then, I build a classification model that uses Natural Language Processing and predicts whether or not a post will have more or fewer than the _median_ number of comments for all the posts I scraped.

# Scraping Post Info from Reddit.com:


```python
import requests
import json
import time
import pandas as pd
```


```python
# the URL from which I'm going to scrape posts
URL = "http://www.reddit.com/hot.json"

# In order to scrape reddit I need to give a custom user agent (kind of like a user name), otherwise
# Python will use the default user name, and reddit will block it since there are so many people
# using the same user name, so here I define my new reddit username.
headers = {'User-agent': 'Ben_Ironside'}

# make a request object to get and store the data from the above URL
res = requests.get(URL, headers = headers)

# check the status of the connection with the URL. Since I'm using a custom
# user agent, it should be fine (status = 200)
res.status_code
```




    200




```python
# download the webpage's data in JSON format using the res (requests) object
json_data = res.json()

# check how the data is organized in the JSON
json_data.keys()
```




    dict_keys(['kind', 'data'])




```python
# check out what's in each of those keys
json_data['kind']
```




    'Listing'




```python
json_data['data']
```




    {'after': 't3_8o8nsm',
     'before': None,
     'children': [{'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'maxwellhill',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528066997.0,
        'created_utc': 1528038197.0,
        'distinguished': None,
        'domain': 'thehill.com',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o917w',
        'is_crosspostable': False,
        'is_reddit_media_domain': False,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': None,
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o917w',
        'no_follow': False,
        'num_comments': 3225,
        'num_crossposts': 0,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/worldnews/comments/8o917w/trudeau_its_insulting_that_the_us_considers/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'link',
        'preview': {'enabled': False,
         'images': [{'id': 'UPq31VlrvZ6XMJCJ-nAqotpxibElKMS51EDx0KFl9EQ',
           'resolutions': [{'height': 60,
             'url': 'https://i.redditmedia.com/xYpMhO3a8uw877Ii_1VC88lZXW1KV3N8ZbWbWKMv4oo.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=165d6563c8bbfdfacfb4274c636e9a04',
             'width': 108},
            {'height': 121,
             'url': 'https://i.redditmedia.com/xYpMhO3a8uw877Ii_1VC88lZXW1KV3N8ZbWbWKMv4oo.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=ae414e52af67f67898b26f71a7edec98',
             'width': 216},
            {'height': 179,
             'url': 'https://i.redditmedia.com/xYpMhO3a8uw877Ii_1VC88lZXW1KV3N8ZbWbWKMv4oo.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=d99a21eee8ba40780f8c28d9387357c5',
             'width': 320},
            {'height': 359,
             'url': 'https://i.redditmedia.com/xYpMhO3a8uw877Ii_1VC88lZXW1KV3N8ZbWbWKMv4oo.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;s=f9fd84e4168d21238cced85c16fef051',
             'width': 640},
            {'height': 539,
             'url': 'https://i.redditmedia.com/xYpMhO3a8uw877Ii_1VC88lZXW1KV3N8ZbWbWKMv4oo.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=960&amp;s=63113b7973cac3e0191410a0be1e5101',
             'width': 960}],
           'source': {'height': 551,
            'url': 'https://i.redditmedia.com/xYpMhO3a8uw877Ii_1VC88lZXW1KV3N8ZbWbWKMv4oo.jpg?s=5d44f51987887d365a94875c51537d81',
            'width': 980},
           'variants': {}}]},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 26783,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': False,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'worldnews',
        'subreddit_id': 't5_2qh13',
        'subreddit_name_prefixed': 'r/worldnews',
        'subreddit_subscribers': 18800039,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'default',
        'thumbnail_height': 78,
        'thumbnail_width': 140,
        'title': "Trudeau: It's 'insulting' that the US considers Canada a national security threat",
        'ups': 26783,
        'url': 'http://thehill.com/policy/international/390425-trudeau-its-insulting-that-the-us-considers-canada-a-national-security',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'DanDelta100',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528061969.0,
        'created_utc': 1528033169.0,
        'distinguished': None,
        'domain': 'i.imgur.com',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o8l4h',
        'is_crosspostable': False,
        'is_reddit_media_domain': False,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': None,
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o8l4h',
        'no_follow': False,
        'num_comments': 300,
        'num_crossposts': 6,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/WhitePeopleTwitter/comments/8o8l4h/this_modelling_job_was_a_mistake/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'image',
        'preview': {'enabled': True,
         'images': [{'id': '-fuAdxoX20QM4qfKpnAFYgj5Fubho3EucANUO1in3Ss',
           'resolutions': [{'height': 148,
             'url': 'https://i.redditmedia.com/EtUvnVGMej7malWhBBousuUs67GFdjoSH2RDNmw3WVM.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=c37a419c8c682d3a3295d48345f1fe71',
             'width': 108},
            {'height': 297,
             'url': 'https://i.redditmedia.com/EtUvnVGMej7malWhBBousuUs67GFdjoSH2RDNmw3WVM.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=ecc27ca387ba94c79228cba6e48a01a3',
             'width': 216},
            {'height': 440,
             'url': 'https://i.redditmedia.com/EtUvnVGMej7malWhBBousuUs67GFdjoSH2RDNmw3WVM.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=831ac811249b8896f4d72d5bc9f3058c',
             'width': 320},
            {'height': 880,
             'url': 'https://i.redditmedia.com/EtUvnVGMej7malWhBBousuUs67GFdjoSH2RDNmw3WVM.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;s=a98e92750bc90ccc7011668273fb14f7',
             'width': 640},
            {'height': 1320,
             'url': 'https://i.redditmedia.com/EtUvnVGMej7malWhBBousuUs67GFdjoSH2RDNmw3WVM.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=960&amp;s=61cf5b1f3eac0257903c4f8ed8a59a1c',
             'width': 960}],
           'source': {'height': 1459,
            'url': 'https://i.redditmedia.com/EtUvnVGMej7malWhBBousuUs67GFdjoSH2RDNmw3WVM.jpg?s=0a8a33a7094867646630bd84a8aa6b45',
            'width': 1061},
           'variants': {}}]},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 22464,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': True,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'WhitePeopleTwitter',
        'subreddit_id': 't5_35n7t',
        'subreddit_name_prefixed': 'r/WhitePeopleTwitter',
        'subreddit_subscribers': 524358,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://b.thumbs.redditmedia.com/Fc1oYgekK2xuf0QWz6VRZMo6w9ClCEipfDqndiWBCSY.jpg',
        'thumbnail_height': 140,
        'thumbnail_width': 140,
        'title': 'This Modelling job was a Mistake 🍌😔',
        'ups': 22464,
        'url': 'https://i.imgur.com/8ss2pJ6.jpg',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'joha7609',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528067437.0,
        'created_utc': 1528038637.0,
        'distinguished': None,
        'domain': 'self.AskReddit',
        'downs': 0,
        'edited': False,
        'gilded': 1,
        'hidden': False,
        'hide_score': False,
        'id': '8o937s',
        'is_crosspostable': False,
        'is_reddit_media_domain': False,
        'is_self': True,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': None,
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o937s',
        'no_follow': False,
        'num_comments': 4082,
        'num_crossposts': 0,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/AskReddit/comments/8o937s/what_are_some_of_the_best_choices_youve_made_in/',
        'pinned': False,
        'post_categories': None,
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 9819,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': True,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'AskReddit',
        'subreddit_id': 't5_2qh1i',
        'subreddit_name_prefixed': 'r/AskReddit',
        'subreddit_subscribers': 19317713,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'self',
        'thumbnail_height': None,
        'thumbnail_width': None,
        'title': "What are some of the BEST choices, you've made in your life?",
        'ups': 9819,
        'url': 'https://www.reddit.com/r/AskReddit/comments/8o937s/what_are_some_of_the_best_choices_youve_made_in/',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'HelpMeImPhat',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528064216.0,
        'created_utc': 1528035416.0,
        'distinguished': None,
        'domain': 'i.redd.it',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o8sss',
        'is_crosspostable': False,
        'is_reddit_media_domain': True,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': '/r/all',
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o8sss',
        'no_follow': False,
        'num_comments': 475,
        'num_crossposts': 1,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/iamverysmart/comments/8o8sss/he_was_ten_and_realized_this_how_could_you_not/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'image',
        'preview': {'enabled': True,
         'images': [{'id': 'MSeaXNWglASF0C-KSADCSQPmWJuDDCDwkvZdaxcE_gE',
           'resolutions': [{'height': 108,
             'url': 'https://i.redditmedia.com/sgmN0FDsl3IJVAyOnpH7Xh2N7ynuWYjfu3ZY_0GRSSc.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=fab98c9d871deb4ed4d16f0c347e1735',
             'width': 108},
            {'height': 216,
             'url': 'https://i.redditmedia.com/sgmN0FDsl3IJVAyOnpH7Xh2N7ynuWYjfu3ZY_0GRSSc.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=653d68e8568292f96d9fd2c774956118',
             'width': 216},
            {'height': 320,
             'url': 'https://i.redditmedia.com/sgmN0FDsl3IJVAyOnpH7Xh2N7ynuWYjfu3ZY_0GRSSc.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=d8d27f313c1852299bd97f7a4b11e96c',
             'width': 320},
            {'height': 640,
             'url': 'https://i.redditmedia.com/sgmN0FDsl3IJVAyOnpH7Xh2N7ynuWYjfu3ZY_0GRSSc.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;s=1f353a266237138f75ab21f01a498692',
             'width': 640},
            {'height': 960,
             'url': 'https://i.redditmedia.com/sgmN0FDsl3IJVAyOnpH7Xh2N7ynuWYjfu3ZY_0GRSSc.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=960&amp;s=e4cbe50fe71f3a8f609a831058db1838',
             'width': 960},
            {'height': 1080,
             'url': 'https://i.redditmedia.com/sgmN0FDsl3IJVAyOnpH7Xh2N7ynuWYjfu3ZY_0GRSSc.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=1080&amp;s=769d90b20400cbdde16a217ea06312e2',
             'width': 1080}],
           'source': {'height': 2048,
            'url': 'https://i.redditmedia.com/sgmN0FDsl3IJVAyOnpH7Xh2N7ynuWYjfu3ZY_0GRSSc.jpg?s=7f4bf3736e8840da0c72021c6ac3e65f',
            'width': 2048},
           'variants': {}}]},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 15823,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': True,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'iamverysmart',
        'subreddit_id': 't5_2yuej',
        'subreddit_name_prefixed': 'r/iamverysmart',
        'subreddit_subscribers': 692219,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://b.thumbs.redditmedia.com/nI-OgqFPWbb2T1QiAnp8T-JrvQjKDzyFGusfyUCF0XM.jpg',
        'thumbnail_height': 140,
        'thumbnail_width': 140,
        'title': 'He was TEN and realized this, how could you not?!',
        'ups': 15823,
        'url': 'https://i.redd.it/7b95fykaos111.jpg',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'Salvadore1',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528064151.0,
        'created_utc': 1528035351.0,
        'distinguished': None,
        'domain': 'i.redd.it',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o8skp',
        'is_crosspostable': False,
        'is_reddit_media_domain': True,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': None,
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o8skp',
        'no_follow': False,
        'num_comments': 484,
        'num_crossposts': 1,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/quityourbullshit/comments/8o8skp/redditor_on_rgifs_calls_out_op_who_is_claiming_to/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'image',
        'preview': {'enabled': True,
         'images': [{'id': 'c0zVf-JrFoQtbqU7HoXlXY27GCFCa4IPp6jb3KCF620',
           'resolutions': [{'height': 192,
             'url': 'https://i.redditmedia.com/H3pEOQRTdFC62QKfhB85iEIBHo1T4A3Yofot5CKTK9A.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=7359e82fdeef77d1b42d0fde9b17a4a3',
             'width': 108},
            {'height': 384,
             'url': 'https://i.redditmedia.com/H3pEOQRTdFC62QKfhB85iEIBHo1T4A3Yofot5CKTK9A.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=0ce8ff590b3dfef92e212cc6ff9183b6',
             'width': 216},
            {'height': 568,
             'url': 'https://i.redditmedia.com/H3pEOQRTdFC62QKfhB85iEIBHo1T4A3Yofot5CKTK9A.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=78dd33969700835402ca47d13383bcac',
             'width': 320},
            {'height': 1137,
             'url': 'https://i.redditmedia.com/H3pEOQRTdFC62QKfhB85iEIBHo1T4A3Yofot5CKTK9A.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;s=e753a43d90c659f3dd9f2e2c1bd78d66',
             'width': 640}],
           'source': {'height': 1280,
            'url': 'https://i.redditmedia.com/H3pEOQRTdFC62QKfhB85iEIBHo1T4A3Yofot5CKTK9A.jpg?s=5eeb84b25b865ab42db1632e9780509d',
            'width': 720},
           'variants': {}}]},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 11795,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': False,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'quityourbullshit',
        'subreddit_id': 't5_2y8xf',
        'subreddit_name_prefixed': 'r/quityourbullshit',
        'subreddit_subscribers': 703324,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://b.thumbs.redditmedia.com/2O3OhQiwet4Gvo8t-ykqtf5LRJ92yuQvGcLnmfTFiwA.jpg',
        'thumbnail_height': 140,
        'thumbnail_width': 140,
        'title': "Redditor on r/gifs calls out OP, who is claiming to have the world's deadliest acid in a glass.",
        'ups': 11795,
        'url': 'https://i.redd.it/5k2s4fa3os111.jpg',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'minxwell',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528061324.0,
        'created_utc': 1528032524.0,
        'distinguished': None,
        'domain': 'i.imgur.com',
        'downs': 0,
        'edited': False,
        'gilded': 1,
        'hidden': False,
        'hide_score': False,
        'id': '8o8ixe',
        'is_crosspostable': False,
        'is_reddit_media_domain': False,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': None,
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o8ixe',
        'no_follow': False,
        'num_comments': 3900,
        'num_crossposts': 4,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/pics/comments/8o8ixe/unfriendly_reminder_that_uc_davis_paid_100k_to/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'image',
        'preview': {'enabled': True,
         'images': [{'id': 'StszwK7S7H1rnarRwc8hRqlpxJeKvREaBYozNWHK70Q',
           'resolutions': [{'height': 72,
             'url': 'https://i.redditmedia.com/0Z43_FZ_opfxlEzMwyDm5AU-sL7pfWwyTnRmroo_mxY.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=4ec623a5a49e34a0d4e0fd4b1e18f767',
             'width': 108},
            {'height': 144,
             'url': 'https://i.redditmedia.com/0Z43_FZ_opfxlEzMwyDm5AU-sL7pfWwyTnRmroo_mxY.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=015b65fea4022162819ccf2838658290',
             'width': 216},
            {'height': 213,
             'url': 'https://i.redditmedia.com/0Z43_FZ_opfxlEzMwyDm5AU-sL7pfWwyTnRmroo_mxY.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=d3870f8b16ca69766ec77061c55632b3',
             'width': 320},
            {'height': 427,
             'url': 'https://i.redditmedia.com/0Z43_FZ_opfxlEzMwyDm5AU-sL7pfWwyTnRmroo_mxY.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;s=1b79dd0ce843589f11fac051e3f49dd8',
             'width': 640}],
           'source': {'height': 614,
            'url': 'https://i.redditmedia.com/0Z43_FZ_opfxlEzMwyDm5AU-sL7pfWwyTnRmroo_mxY.jpg?s=b0b82aaf4eb585aa698754cd27d8fa0d',
            'width': 920},
           'variants': {}}]},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 103591,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': True,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'pics',
        'subreddit_id': 't5_2qh0u',
        'subreddit_name_prefixed': 'r/pics',
        'subreddit_subscribers': 18711293,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://b.thumbs.redditmedia.com/4DWRG9Ve_UoLzAOt7tD-bcVrwK0iQOGo5Xn2f6pkLpE.jpg',
        'thumbnail_height': 93,
        'thumbnail_width': 140,
        'title': 'Unfriendly reminder that UC Davis paid &gt;$100k to remove this photo from the internet. Let’s not forget the pepper spray incident.',
        'ups': 103591,
        'url': 'https://i.imgur.com/TV93pH9.jpg',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'germshots',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528063137.0,
        'created_utc': 1528034337.0,
        'distinguished': None,
        'domain': 'v.redd.it',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o8ox6',
        'is_crosspostable': False,
        'is_reddit_media_domain': True,
        'is_self': False,
        'is_video': True,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': None,
        'locked': False,
        'media': {'reddit_video': {'dash_url': 'https://v.redd.it/j7rtc461ls111/DASHPlaylist.mpd',
          'duration': 16,
          'fallback_url': 'https://v.redd.it/j7rtc461ls111/DASH_2_4_M',
          'height': 480,
          'hls_url': 'https://v.redd.it/j7rtc461ls111/HLSPlaylist.m3u8',
          'is_gif': True,
          'scrubber_media_url': 'https://v.redd.it/j7rtc461ls111/DASH_600_K',
          'transcoding_status': 'completed',
          'width': 480}},
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o8ox6',
        'no_follow': False,
        'num_comments': 156,
        'num_crossposts': 1,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/gifs/comments/8o8ox6/mr_peabody/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'hosted:video',
        'preview': {'enabled': False,
         'images': [{'id': 'HPjRKL2Y8wPQagpq-OcZN7UMLA4NoUjgsl8ve6irYRE',
           'resolutions': [{'height': 108,
             'url': 'https://i.redditmedia.com/0ogznvN_fj7oz9wD4oYPjCPWJ4lY2aHbYve7wPka22E.png?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=2c22e55d31877f6b3bea6a1a6088c162',
             'width': 108},
            {'height': 216,
             'url': 'https://i.redditmedia.com/0ogznvN_fj7oz9wD4oYPjCPWJ4lY2aHbYve7wPka22E.png?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=3348ca398a42f8ac9273a7013e43da9c',
             'width': 216},
            {'height': 320,
             'url': 'https://i.redditmedia.com/0ogznvN_fj7oz9wD4oYPjCPWJ4lY2aHbYve7wPka22E.png?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=e504113546ac18fe0acb616dda963e24',
             'width': 320},
            {'height': 640,
             'url': 'https://i.redditmedia.com/0ogznvN_fj7oz9wD4oYPjCPWJ4lY2aHbYve7wPka22E.png?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;s=d279b68f1cc984b6babe68230985d3b2',
             'width': 640}],
           'source': {'height': 640,
            'url': 'https://i.redditmedia.com/0ogznvN_fj7oz9wD4oYPjCPWJ4lY2aHbYve7wPka22E.png?s=16a7bb8592b45aa6e9c135499a995b0a',
            'width': 640},
           'variants': {}}]},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 12293,
        'secure_media': {'reddit_video': {'dash_url': 'https://v.redd.it/j7rtc461ls111/DASHPlaylist.mpd',
          'duration': 16,
          'fallback_url': 'https://v.redd.it/j7rtc461ls111/DASH_2_4_M',
          'height': 480,
          'hls_url': 'https://v.redd.it/j7rtc461ls111/HLSPlaylist.m3u8',
          'is_gif': True,
          'scrubber_media_url': 'https://v.redd.it/j7rtc461ls111/DASH_600_K',
          'transcoding_status': 'completed',
          'width': 480}},
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': False,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'gifs',
        'subreddit_id': 't5_2qt55',
        'subreddit_name_prefixed': 'r/gifs',
        'subreddit_subscribers': 16185387,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://b.thumbs.redditmedia.com/Oof0ebKWhuy--jlpFlT64O6GXO8RFz3FWnEjDJEnDoU.jpg',
        'thumbnail_height': 140,
        'thumbnail_width': 140,
        'title': 'Mr Peabody??',
        'ups': 12293,
        'url': 'https://v.redd.it/j7rtc461ls111',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'comanderz',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528065712.0,
        'created_utc': 1528036912.0,
        'distinguished': None,
        'domain': 'i.imgur.com',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o8x9u',
        'is_crosspostable': False,
        'is_reddit_media_domain': False,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': None,
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o8x9u',
        'no_follow': False,
        'num_comments': 71,
        'num_crossposts': 2,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/funny/comments/8o8x9u/wait_let_me_help_you/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'link',
        'preview': {'enabled': True,
         'images': [{'id': 'jjb9a_GO6I6S3zj3ukB2gM9bFDCGtk1Y1gD4HT1dYHM',
           'resolutions': [{'height': 105,
             'url': 'https://i.redditmedia.com/jjb9a_GO6I6S3zj3ukB2gM9bFDCGtk1Y1gD4HT1dYHM.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;fm=jpg&amp;s=327f5b45ebc7cc54bf9018d68e267f87',
             'width': 108},
            {'height': 211,
             'url': 'https://i.redditmedia.com/jjb9a_GO6I6S3zj3ukB2gM9bFDCGtk1Y1gD4HT1dYHM.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;fm=jpg&amp;s=fc4a08c7fbef9414a1dbf93aebc57852',
             'width': 216}],
           'source': {'height': 304,
            'url': 'https://i.redditmedia.com/jjb9a_GO6I6S3zj3ukB2gM9bFDCGtk1Y1gD4HT1dYHM.gif?fm=jpg&amp;s=de09600d7aaef5c8070f99b008fa41ae',
            'width': 311},
           'variants': {'gif': {'resolutions': [{'height': 105,
               'url': 'https://g.redditmedia.com/jjb9a_GO6I6S3zj3ukB2gM9bFDCGtk1Y1gD4HT1dYHM.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=c37af5130d394cfab6742a8983420f58',
               'width': 108},
              {'height': 211,
               'url': 'https://g.redditmedia.com/jjb9a_GO6I6S3zj3ukB2gM9bFDCGtk1Y1gD4HT1dYHM.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=1b706bf40b763180402fe4e2bf168845',
               'width': 216}],
             'source': {'height': 304,
              'url': 'https://g.redditmedia.com/jjb9a_GO6I6S3zj3ukB2gM9bFDCGtk1Y1gD4HT1dYHM.gif?s=31b85d328082cfac10b5dc7d4cc1df8e',
              'width': 311}},
            'mp4': {'resolutions': [{'height': 105,
               'url': 'https://g.redditmedia.com/jjb9a_GO6I6S3zj3ukB2gM9bFDCGtk1Y1gD4HT1dYHM.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=957995ee6821c3aeefc5d11a4cc1030b',
               'width': 108},
              {'height': 211,
               'url': 'https://g.redditmedia.com/jjb9a_GO6I6S3zj3ukB2gM9bFDCGtk1Y1gD4HT1dYHM.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=b4526fda06bb77b0fa20c07274e0ab55',
               'width': 216}],
             'source': {'height': 304,
              'url': 'https://g.redditmedia.com/jjb9a_GO6I6S3zj3ukB2gM9bFDCGtk1Y1gD4HT1dYHM.gif?fm=mp4&amp;mp4-fragmented=false&amp;s=08e00aff08f131ce38ff623834328fe6',
              'width': 311}}}}],
         'reddit_video_preview': {'dash_url': 'https://v.redd.it/mk25bhiqls111/DASHPlaylist.mpd',
          'duration': 10,
          'fallback_url': 'https://v.redd.it/mk25bhiqls111/DASH_600_K',
          'height': 240,
          'hls_url': 'https://v.redd.it/mk25bhiqls111/HLSPlaylist.m3u8',
          'is_gif': True,
          'scrubber_media_url': 'https://v.redd.it/mk25bhiqls111/DASH_600_K',
          'transcoding_status': 'completed',
          'width': 244}},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 6513,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': True,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'funny',
        'subreddit_id': 't5_2qh33',
        'subreddit_name_prefixed': 'r/funny',
        'subreddit_subscribers': 19643480,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://b.thumbs.redditmedia.com/MsRBUNFQ-qudlqLyBRQcMYHYN26B80COYXGhQRFhK-o.jpg',
        'thumbnail_height': 136,
        'thumbnail_width': 140,
        'title': 'Wait! Let me help you!',
        'ups': 6513,
        'url': 'https://i.imgur.com/J3g0KCk.gifv',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'BlockedEyes',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528059038.0,
        'created_utc': 1528030238.0,
        'distinguished': None,
        'domain': 'i.imgur.com',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o8bhl',
        'is_crosspostable': False,
        'is_reddit_media_domain': False,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': None,
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o8bhl',
        'no_follow': False,
        'num_comments': 502,
        'num_crossposts': 3,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/bigboye/comments/8o8bhl/sweet_african_danger_dog/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'link',
        'preview': {'enabled': True,
         'images': [{'id': 'YWkEfdaK3zGF4dIrHINDGziGicaoMHb4CSrwos8LV14',
           'resolutions': [{'height': 135,
             'url': 'https://i.redditmedia.com/z1_Kc8-Wlfr9mYWTYHboM47-YfXU4_2cxcSQLP0KDnY.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;fm=jpg&amp;s=28ce075617a57ea98be856476656d02c',
             'width': 108},
            {'height': 270,
             'url': 'https://i.redditmedia.com/z1_Kc8-Wlfr9mYWTYHboM47-YfXU4_2cxcSQLP0KDnY.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;fm=jpg&amp;s=f23dcd46606e020b59853d507439e2d0',
             'width': 216},
            {'height': 400,
             'url': 'https://i.redditmedia.com/z1_Kc8-Wlfr9mYWTYHboM47-YfXU4_2cxcSQLP0KDnY.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;fm=jpg&amp;s=a1176e9a6d5c3f57faba969d03ded246',
             'width': 320},
            {'height': 800,
             'url': 'https://i.redditmedia.com/z1_Kc8-Wlfr9mYWTYHboM47-YfXU4_2cxcSQLP0KDnY.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;fm=jpg&amp;s=fe346923c9af3d8bdaad5eeb7e910560',
             'width': 640}],
           'source': {'height': 910,
            'url': 'https://i.redditmedia.com/z1_Kc8-Wlfr9mYWTYHboM47-YfXU4_2cxcSQLP0KDnY.gif?fm=jpg&amp;s=252332b26f0bfe3dbda44a56c6a3e6d5',
            'width': 728},
           'variants': {'gif': {'resolutions': [{'height': 135,
               'url': 'https://g.redditmedia.com/z1_Kc8-Wlfr9mYWTYHboM47-YfXU4_2cxcSQLP0KDnY.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=be1383573b739860fe1017ac933ae149',
               'width': 108},
              {'height': 270,
               'url': 'https://g.redditmedia.com/z1_Kc8-Wlfr9mYWTYHboM47-YfXU4_2cxcSQLP0KDnY.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=54f4ca7bc91e63470d0ab57c8acfe9c1',
               'width': 216},
              {'height': 400,
               'url': 'https://g.redditmedia.com/z1_Kc8-Wlfr9mYWTYHboM47-YfXU4_2cxcSQLP0KDnY.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=31b642132418e98eb3e3382e71127afa',
               'width': 320},
              {'height': 800,
               'url': 'https://g.redditmedia.com/z1_Kc8-Wlfr9mYWTYHboM47-YfXU4_2cxcSQLP0KDnY.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;s=708eb5be42a86f17aa38b894d0ceebb1',
               'width': 640}],
             'source': {'height': 910,
              'url': 'https://g.redditmedia.com/z1_Kc8-Wlfr9mYWTYHboM47-YfXU4_2cxcSQLP0KDnY.gif?s=467d5b9764df6a086ce8e09616b12c75',
              'width': 728}},
            'mp4': {'resolutions': [{'height': 135,
               'url': 'https://g.redditmedia.com/z1_Kc8-Wlfr9mYWTYHboM47-YfXU4_2cxcSQLP0KDnY.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=701a34209e7bb6979fef200a80766bf9',
               'width': 108},
              {'height': 270,
               'url': 'https://g.redditmedia.com/z1_Kc8-Wlfr9mYWTYHboM47-YfXU4_2cxcSQLP0KDnY.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=9810bfbf65ef975b66cc14810b27952e',
               'width': 216},
              {'height': 400,
               'url': 'https://g.redditmedia.com/z1_Kc8-Wlfr9mYWTYHboM47-YfXU4_2cxcSQLP0KDnY.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=99a4e3aedfeffa4fd8a657a3742e39ce',
               'width': 320},
              {'height': 800,
               'url': 'https://g.redditmedia.com/z1_Kc8-Wlfr9mYWTYHboM47-YfXU4_2cxcSQLP0KDnY.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=053ed4743fd473eb706bb2e37c7e1bf7',
               'width': 640}],
             'source': {'height': 910,
              'url': 'https://g.redditmedia.com/z1_Kc8-Wlfr9mYWTYHboM47-YfXU4_2cxcSQLP0KDnY.gif?fm=mp4&amp;mp4-fragmented=false&amp;s=ccdf2fc992ffc7f321988763f2eed1d2',
              'width': 728}}}}],
         'reddit_video_preview': {'dash_url': 'https://v.redd.it/rthbv0vy8s111/DASHPlaylist.mpd',
          'duration': 12,
          'fallback_url': 'https://v.redd.it/rthbv0vy8s111/DASH_4_8_M',
          'height': 720,
          'hls_url': 'https://v.redd.it/rthbv0vy8s111/HLSPlaylist.m3u8',
          'is_gif': True,
          'scrubber_media_url': 'https://v.redd.it/rthbv0vy8s111/DASH_600_K',
          'transcoding_status': 'completed',
          'width': 576}},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 13562,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': True,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'bigboye',
        'subreddit_id': 't5_3k32c',
        'subreddit_name_prefixed': 'r/bigboye',
        'subreddit_subscribers': 95022,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://a.thumbs.redditmedia.com/dfuBSRed5KnsInI4URR9DE8XIgW4mph3dUucWiULwd4.jpg',
        'thumbnail_height': 140,
        'thumbnail_width': 140,
        'title': 'Sweet African Danger Dog',
        'ups': 13562,
        'url': 'https://i.imgur.com/qah0K39.gifv',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'rosesarewet',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528066493.0,
        'created_utc': 1528037693.0,
        'distinguished': None,
        'domain': 'i.imgur.com',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o8zmm',
        'is_crosspostable': False,
        'is_reddit_media_domain': False,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': 'l-i',
        'link_flair_text': 'Image',
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o8zmm',
        'no_follow': False,
        'num_comments': 64,
        'num_crossposts': 1,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/Damnthatsinteresting/comments/8o8zmm/bioluminescent_photoplankton_at_night/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'image',
        'preview': {'enabled': True,
         'images': [{'id': '-YOdeE16wFns5QmWI8GfbUZaMtsmT4RV6ANDlmfApn0',
           'resolutions': [{'height': 134,
             'url': 'https://i.redditmedia.com/Eq_bAO6zXFauon5Uu2zEJF9OAz4zlqF8gVj6Orb4dHw.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=af9a10348c64dda3a4d0cbe43bed5a75',
             'width': 108},
            {'height': 268,
             'url': 'https://i.redditmedia.com/Eq_bAO6zXFauon5Uu2zEJF9OAz4zlqF8gVj6Orb4dHw.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=af727f620ec9505d511786df5ab80410',
             'width': 216},
            {'height': 397,
             'url': 'https://i.redditmedia.com/Eq_bAO6zXFauon5Uu2zEJF9OAz4zlqF8gVj6Orb4dHw.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=cad9d81c0a04b5aebbeca325da85f44d',
             'width': 320},
            {'height': 794,
             'url': 'https://i.redditmedia.com/Eq_bAO6zXFauon5Uu2zEJF9OAz4zlqF8gVj6Orb4dHw.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;s=fb2f9cf8600dc99321725ef86a06f11d',
             'width': 640},
            {'height': 1191,
             'url': 'https://i.redditmedia.com/Eq_bAO6zXFauon5Uu2zEJF9OAz4zlqF8gVj6Orb4dHw.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=960&amp;s=d1703f19f81d3d5483b0056e17b2e920',
             'width': 960},
            {'height': 1340,
             'url': 'https://i.redditmedia.com/Eq_bAO6zXFauon5Uu2zEJF9OAz4zlqF8gVj6Orb4dHw.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=1080&amp;s=a749d63bde614f2f7160b081d631697d',
             'width': 1080}],
           'source': {'height': 1542,
            'url': 'https://i.redditmedia.com/Eq_bAO6zXFauon5Uu2zEJF9OAz4zlqF8gVj6Orb4dHw.jpg?s=604438a73cc7853c19d3ebc68703a61f',
            'width': 1242},
           'variants': {}}]},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 6608,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': True,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'Damnthatsinteresting',
        'subreddit_id': 't5_2xxyj',
        'subreddit_name_prefixed': 'r/Damnthatsinteresting',
        'subreddit_subscribers': 733642,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://a.thumbs.redditmedia.com/Z0xNv2etImoCox890yrElMGzc5euYt0lDNGLedqs6z0.jpg',
        'thumbnail_height': 140,
        'thumbnail_width': 140,
        'title': 'Bioluminescent photoplankton at night',
        'ups': 6608,
        'url': 'https://i.imgur.com/DvqZBpS.jpg',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'diddylongbreadz_rm',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528060438.0,
        'created_utc': 1528031638.0,
        'distinguished': None,
        'domain': 'i.imgur.com',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o8g0c',
        'is_crosspostable': False,
        'is_reddit_media_domain': False,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': None,
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o8g0c',
        'no_follow': False,
        'num_comments': 227,
        'num_crossposts': 0,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/OldSchoolCool/comments/8o8g0c/lynda_carter_joking_with_some_extras_on_the_set/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'image',
        'preview': {'enabled': True,
         'images': [{'id': 'Skc-YzEQRJ5EJ8Xm3pn5QbzcMSk82bDC4jCal8QWWqk',
           'resolutions': [{'height': 155,
             'url': 'https://i.redditmedia.com/cRaJ65YYyMDK3sHW4skmfihZS6OOMtd6WHy1GVbpfmc.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=05c962316df1d8cb06776a47aff0674b',
             'width': 108},
            {'height': 310,
             'url': 'https://i.redditmedia.com/cRaJ65YYyMDK3sHW4skmfihZS6OOMtd6WHy1GVbpfmc.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=de36eeb5b4c5b287849ab701d7f67486',
             'width': 216},
            {'height': 460,
             'url': 'https://i.redditmedia.com/cRaJ65YYyMDK3sHW4skmfihZS6OOMtd6WHy1GVbpfmc.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=04c50554bff95ed0ebbb5bc2bacb4119',
             'width': 320}],
           'source': {'height': 906,
            'url': 'https://i.redditmedia.com/cRaJ65YYyMDK3sHW4skmfihZS6OOMtd6WHy1GVbpfmc.jpg?s=ef5f5335e3a71cdba68193ba835a2338',
            'width': 630},
           'variants': {}}]},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 11301,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': True,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'OldSchoolCool',
        'subreddit_id': 't5_2tycb',
        'subreddit_name_prefixed': 'r/OldSchoolCool',
        'subreddit_subscribers': 12863928,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://b.thumbs.redditmedia.com/0FlCF7kPLHrGe2vfjBcJf5hIBqx_PeDwtEcZPvgalnQ.jpg',
        'thumbnail_height': 140,
        'thumbnail_width': 140,
        'title': "Lynda Carter joking with some extras on the set of 'Wonder Woman'. (1976)",
        'ups': 11301,
        'url': 'https://i.imgur.com/R1cIsXY.jpg',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'stop_hating1',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528066579.0,
        'created_utc': 1528037779.0,
        'distinguished': None,
        'domain': 'i.imgur.com',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o8zvw',
        'is_crosspostable': False,
        'is_reddit_media_domain': False,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': None,
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o8zvw',
        'no_follow': False,
        'num_comments': 60,
        'num_crossposts': 2,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/BetterEveryLoop/comments/8o8zvw/collecting_your_luggage/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'link',
        'preview': {'enabled': True,
         'images': [{'id': 'vSvXRglXfw3bYBRDn6-ucU_bqeAwbagdpBOaJZuAlbE',
           'resolutions': [{'height': 108,
             'url': 'https://i.redditmedia.com/mcdUmk3ubbEv-SGikDdPdknQMbZCpNcQKucWR6X90ss.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;fm=jpg&amp;s=637933d579aac732f160b656cb371f12',
             'width': 108},
            {'height': 216,
             'url': 'https://i.redditmedia.com/mcdUmk3ubbEv-SGikDdPdknQMbZCpNcQKucWR6X90ss.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;fm=jpg&amp;s=c1aaf89aec84ddb9a93e8f4019b85828',
             'width': 216},
            {'height': 320,
             'url': 'https://i.redditmedia.com/mcdUmk3ubbEv-SGikDdPdknQMbZCpNcQKucWR6X90ss.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;fm=jpg&amp;s=8c204efb7193a413e7f3b43b543986ef',
             'width': 320}],
           'source': {'height': 480,
            'url': 'https://i.redditmedia.com/mcdUmk3ubbEv-SGikDdPdknQMbZCpNcQKucWR6X90ss.gif?fm=jpg&amp;s=3bef62092cbec33413a567cbd19ce931',
            'width': 480},
           'variants': {'gif': {'resolutions': [{'height': 108,
               'url': 'https://g.redditmedia.com/mcdUmk3ubbEv-SGikDdPdknQMbZCpNcQKucWR6X90ss.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=a9e75b7ebf27ca2b70b3f437b1251827',
               'width': 108},
              {'height': 216,
               'url': 'https://g.redditmedia.com/mcdUmk3ubbEv-SGikDdPdknQMbZCpNcQKucWR6X90ss.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=14c2d43f80a2e74dee87b5261074e340',
               'width': 216},
              {'height': 320,
               'url': 'https://g.redditmedia.com/mcdUmk3ubbEv-SGikDdPdknQMbZCpNcQKucWR6X90ss.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=4d54f07620b7add5255d630d5e56fbb5',
               'width': 320}],
             'source': {'height': 480,
              'url': 'https://g.redditmedia.com/mcdUmk3ubbEv-SGikDdPdknQMbZCpNcQKucWR6X90ss.gif?s=501eb643201243fee929994eb14c3af3',
              'width': 480}},
            'mp4': {'resolutions': [{'height': 108,
               'url': 'https://g.redditmedia.com/mcdUmk3ubbEv-SGikDdPdknQMbZCpNcQKucWR6X90ss.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=f53b97787286418bb7f4af6bf9ac2a1d',
               'width': 108},
              {'height': 216,
               'url': 'https://g.redditmedia.com/mcdUmk3ubbEv-SGikDdPdknQMbZCpNcQKucWR6X90ss.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=b6f70e20953d205942f106fa629db7b0',
               'width': 216},
              {'height': 320,
               'url': 'https://g.redditmedia.com/mcdUmk3ubbEv-SGikDdPdknQMbZCpNcQKucWR6X90ss.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=1465ecd7eabbb91f87062bcbcc4c5882',
               'width': 320}],
             'source': {'height': 480,
              'url': 'https://g.redditmedia.com/mcdUmk3ubbEv-SGikDdPdknQMbZCpNcQKucWR6X90ss.gif?fm=mp4&amp;mp4-fragmented=false&amp;s=b67bd46fc5d25d4c3849a2213c0c6abb',
              'width': 480}}}}],
         'reddit_video_preview': {'dash_url': 'https://v.redd.it/j0rs3i5cvs111/DASHPlaylist.mpd',
          'duration': 22,
          'fallback_url': 'https://v.redd.it/j0rs3i5cvs111/DASH_2_4_M',
          'height': 480,
          'hls_url': 'https://v.redd.it/j0rs3i5cvs111/HLSPlaylist.m3u8',
          'is_gif': True,
          'scrubber_media_url': 'https://v.redd.it/j0rs3i5cvs111/DASH_600_K',
          'transcoding_status': 'completed',
          'width': 480}},
        'previous_visits': [],
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 6502,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': False,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'BetterEveryLoop',
        'subreddit_id': 't5_3abwq',
        'subreddit_name_prefixed': 'r/BetterEveryLoop',
        'subreddit_subscribers': 699670,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://b.thumbs.redditmedia.com/naZyP6AOCEYLSFI4FREJ-kBC-zlg8uvSz0wZKCKEl7M.jpg',
        'thumbnail_height': 140,
        'thumbnail_width': 140,
        'title': 'Collecting your luggage.',
        'ups': 6502,
        'url': 'https://i.imgur.com/qBDeAC4.gifv',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'Hdalby33',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528064997.0,
        'created_utc': 1528036197.0,
        'distinguished': None,
        'domain': 'i.imgur.com',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o8v5f',
        'is_crosspostable': False,
        'is_reddit_media_domain': False,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': '/r/all',
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o8v5f',
        'no_follow': False,
        'num_comments': 87,
        'num_crossposts': 1,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/Eyebleach/comments/8o8v5f/formerly_paralyzed_patient_surprises_the_nurse/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'link',
        'preview': {'enabled': False,
         'images': [{'id': 'xuh3vJS01LMYALTnQQT12hsituPydPOI1Sq8uRN-VEw',
           'resolutions': [{'height': 191,
             'url': 'https://i.redditmedia.com/_kh5_5FLvOz_dR97jtKh5pl-wEotq1qeP6oF8mBOvLY.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=a9cd5a33fa7cc25bb763b90d00e0f46f',
             'width': 108},
            {'height': 382,
             'url': 'https://i.redditmedia.com/_kh5_5FLvOz_dR97jtKh5pl-wEotq1qeP6oF8mBOvLY.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=d4e016bc67d80a41d5d59aa84cf75eff',
             'width': 216},
            {'height': 567,
             'url': 'https://i.redditmedia.com/_kh5_5FLvOz_dR97jtKh5pl-wEotq1qeP6oF8mBOvLY.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=3383d9a3e2cc868aab85619b3dccad57',
             'width': 320},
            {'height': 1134,
             'url': 'https://i.redditmedia.com/_kh5_5FLvOz_dR97jtKh5pl-wEotq1qeP6oF8mBOvLY.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;s=c354ab41dfd2d43113432d59f16a3311',
             'width': 640}],
           'source': {'height': 1276,
            'url': 'https://i.redditmedia.com/_kh5_5FLvOz_dR97jtKh5pl-wEotq1qeP6oF8mBOvLY.jpg?s=4eb3be4c437bebe7da2d8a50e2308b50',
            'width': 720},
           'variants': {}}],
         'reddit_video_preview': {'dash_url': 'https://v.redd.it/1y5wrl5fqs111/DASHPlaylist.mpd',
          'duration': 16,
          'fallback_url': 'https://v.redd.it/1y5wrl5fqs111/DASH_9_6_M',
          'height': 1080,
          'hls_url': 'https://v.redd.it/1y5wrl5fqs111/HLSPlaylist.m3u8',
          'is_gif': True,
          'scrubber_media_url': 'https://v.redd.it/1y5wrl5fqs111/DASH_600_K',
          'transcoding_status': 'completed',
          'width': 610}},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 7863,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': True,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'Eyebleach',
        'subreddit_id': 't5_2s427',
        'subreddit_name_prefixed': 'r/Eyebleach',
        'subreddit_subscribers': 860864,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://b.thumbs.redditmedia.com/h74pK9eqXsc9_6Cqc6TzqBPExeYNMT7C5mvtJ-dpq2M.jpg',
        'thumbnail_height': 140,
        'thumbnail_width': 140,
        'title': 'Formerly paralyzed patient surprises the nurse that took care of her by standing up.',
        'ups': 7863,
        'url': 'https://i.imgur.com/2TR6W9P.gifv',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'onthewall2983',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528070871.0,
        'created_utc': 1528042071.0,
        'distinguished': None,
        'domain': 'self.movies',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o9ikh',
        'is_crosspostable': False,
        'is_reddit_media_domain': False,
        'is_self': True,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': None,
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o9ikh',
        'no_follow': False,
        'num_comments': 339,
        'num_crossposts': 1,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/movies/comments/8o9ikh/blade_runner_2049_premiered_on_hbo_last_night/',
        'pinned': False,
        'post_categories': None,
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 3935,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': 'HBO is infamous for showing widescreen movies in the pan &amp; scan format in the old days, and more recently scanning them to fit modern TVs. But lately for the last few years they have shown several films (off the top of my head, Gone Girl, The Martian, The Revenant and Logan, mostly Fox films) in their original aspect ratios. \n\nIt was a real treat to revisit this movie this way almost a year after seeing it on the big screen.',
        'selftext_html': '&lt;!-- SC_OFF --&gt;&lt;div class="md"&gt;&lt;p&gt;HBO is infamous for showing widescreen movies in the pan &amp;amp; scan format in the old days, and more recently scanning them to fit modern TVs. But lately for the last few years they have shown several films (off the top of my head, Gone Girl, The Martian, The Revenant and Logan, mostly Fox films) in their original aspect ratios. &lt;/p&gt;\n\n&lt;p&gt;It was a real treat to revisit this movie this way almost a year after seeing it on the big screen.&lt;/p&gt;\n&lt;/div&gt;&lt;!-- SC_ON --&gt;',
        'send_replies': False,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'movies',
        'subreddit_id': 't5_2qh3s',
        'subreddit_name_prefixed': 'r/movies',
        'subreddit_subscribers': 17646541,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'self',
        'thumbnail_height': None,
        'thumbnail_width': None,
        'title': "Blade Runner 2049 premiered on HBO last night, shown fully in it's widescreen format",
        'ups': 3935,
        'url': 'https://www.reddit.com/r/movies/comments/8o9ikh/blade_runner_2049_premiered_on_hbo_last_night/',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'berrysardar',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528055336.0,
        'created_utc': 1528026536.0,
        'distinguished': None,
        'domain': 'i.redd.it',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o817m',
        'is_crosspostable': False,
        'is_reddit_media_domain': True,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': None,
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o817m',
        'no_follow': False,
        'num_comments': 1201,
        'num_crossposts': 10,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/space/comments/8o817m/temperature_of_the_universe_from_absolute_cold_to/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'image',
        'preview': {'enabled': True,
         'images': [{'id': 'exKRPBLwnrQ5Kc-cMYfRoXdC8mWCDK2JuKhOcVYvdHk',
           'resolutions': [{'height': 216,
             'url': 'https://i.redditmedia.com/qOOZ6sy-qDfV2MYGWxuPrDAT_P7qYzhQh7tAKS4Yx2w.png?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=b206a4225d501338642fc7024e224118',
             'width': 108},
            {'height': 432,
             'url': 'https://i.redditmedia.com/qOOZ6sy-qDfV2MYGWxuPrDAT_P7qYzhQh7tAKS4Yx2w.png?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=dca2d89b768537068a3e1a3f3fc9f1df',
             'width': 216},
            {'height': 640,
             'url': 'https://i.redditmedia.com/qOOZ6sy-qDfV2MYGWxuPrDAT_P7qYzhQh7tAKS4Yx2w.png?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=1e12dc03295a10cb413780575bcd7b8d',
             'width': 320},
            {'height': 1280,
             'url': 'https://i.redditmedia.com/qOOZ6sy-qDfV2MYGWxuPrDAT_P7qYzhQh7tAKS4Yx2w.png?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;s=5c9ef75d214d3444e522e6544db8c48c',
             'width': 640}],
           'source': {'height': 4000,
            'url': 'https://i.redditmedia.com/qOOZ6sy-qDfV2MYGWxuPrDAT_P7qYzhQh7tAKS4Yx2w.png?s=ce694e67ae98d7428d744755ff57ef5e',
            'width': 883},
           'variants': {}}]},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 29911,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': False,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'space',
        'subreddit_id': 't5_2qh87',
        'subreddit_name_prefixed': 'r/space',
        'subreddit_subscribers': 13882075,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://b.thumbs.redditmedia.com/HabSQRAQ7Dp5eL4_1nqIMs1C6HRd99lR9DKm62LGtcQ.jpg',
        'thumbnail_height': 140,
        'thumbnail_width': 140,
        'title': 'Temperature of the Universe from Absolute Cold to Absolute Hot',
        'ups': 29911,
        'url': 'https://i.redd.it/cc55dg4wxr111.png',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'booboo1998',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528059822.0,
        'created_utc': 1528031022.0,
        'distinguished': None,
        'domain': 'i.imgur.com',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o8dxg',
        'is_crosspostable': False,
        'is_reddit_media_domain': False,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': None,
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o8dxg',
        'no_follow': False,
        'num_comments': 356,
        'num_crossposts': 2,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/gaming/comments/8o8dxg/there_are_no_winners_in_war_only_losers/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'image',
        'preview': {'enabled': True,
         'images': [{'id': 'yr5AKSzzhUCjufWJ4OIOvfuIUjZDif7P7_rg56_7zKk',
           'resolutions': [{'height': 60,
             'url': 'https://i.redditmedia.com/vDnu7R3ZqVZdyy3s5Hv1sus2p1xX6GlZ73mgKgSn_Do.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;fm=jpg&amp;s=1524620c35100fb59db0b7f5e38a6d68',
             'width': 108},
            {'height': 121,
             'url': 'https://i.redditmedia.com/vDnu7R3ZqVZdyy3s5Hv1sus2p1xX6GlZ73mgKgSn_Do.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;fm=jpg&amp;s=560620025678c8a413cba33959d0b78b',
             'width': 216},
            {'height': 179,
             'url': 'https://i.redditmedia.com/vDnu7R3ZqVZdyy3s5Hv1sus2p1xX6GlZ73mgKgSn_Do.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;fm=jpg&amp;s=b48f73ebd0f6e449b65a3546845f9ff1',
             'width': 320},
            {'height': 358,
             'url': 'https://i.redditmedia.com/vDnu7R3ZqVZdyy3s5Hv1sus2p1xX6GlZ73mgKgSn_Do.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;fm=jpg&amp;s=4b7ff348d2f08919a7f8aba228a22b98',
             'width': 640}],
           'source': {'height': 408,
            'url': 'https://i.redditmedia.com/vDnu7R3ZqVZdyy3s5Hv1sus2p1xX6GlZ73mgKgSn_Do.gif?fm=jpg&amp;s=e4cde2be91954766a5c83a1358595703',
            'width': 728},
           'variants': {'gif': {'resolutions': [{'height': 60,
               'url': 'https://g.redditmedia.com/vDnu7R3ZqVZdyy3s5Hv1sus2p1xX6GlZ73mgKgSn_Do.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=9d9fe32d7ea59d3d5bad7051f2155cd4',
               'width': 108},
              {'height': 121,
               'url': 'https://g.redditmedia.com/vDnu7R3ZqVZdyy3s5Hv1sus2p1xX6GlZ73mgKgSn_Do.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=76d64a1b6fbe1a207969252c2ce99f8f',
               'width': 216},
              {'height': 179,
               'url': 'https://g.redditmedia.com/vDnu7R3ZqVZdyy3s5Hv1sus2p1xX6GlZ73mgKgSn_Do.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=b2d29167dc5aabceef89350b788265a1',
               'width': 320},
              {'height': 358,
               'url': 'https://g.redditmedia.com/vDnu7R3ZqVZdyy3s5Hv1sus2p1xX6GlZ73mgKgSn_Do.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;s=a41130dbf49d2cf9434d9c09290c5452',
               'width': 640}],
             'source': {'height': 408,
              'url': 'https://g.redditmedia.com/vDnu7R3ZqVZdyy3s5Hv1sus2p1xX6GlZ73mgKgSn_Do.gif?s=b74280dfac951138a31740305a5f5e8b',
              'width': 728}},
            'mp4': {'resolutions': [{'height': 60,
               'url': 'https://g.redditmedia.com/vDnu7R3ZqVZdyy3s5Hv1sus2p1xX6GlZ73mgKgSn_Do.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=45052290f5838e0a3c9da6e61c11215b',
               'width': 108},
              {'height': 121,
               'url': 'https://g.redditmedia.com/vDnu7R3ZqVZdyy3s5Hv1sus2p1xX6GlZ73mgKgSn_Do.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=348a1ce0fe827f261a325fa7fb5f53b4',
               'width': 216},
              {'height': 179,
               'url': 'https://g.redditmedia.com/vDnu7R3ZqVZdyy3s5Hv1sus2p1xX6GlZ73mgKgSn_Do.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=fd837e47f4f4b21a4354e2cecda0e175',
               'width': 320},
              {'height': 358,
               'url': 'https://g.redditmedia.com/vDnu7R3ZqVZdyy3s5Hv1sus2p1xX6GlZ73mgKgSn_Do.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=96b813fd70a76e020ee77567c87262d7',
               'width': 640}],
             'source': {'height': 408,
              'url': 'https://g.redditmedia.com/vDnu7R3ZqVZdyy3s5Hv1sus2p1xX6GlZ73mgKgSn_Do.gif?fm=mp4&amp;mp4-fragmented=false&amp;s=4bfe01fbb7a42ed8aba55dd0f6323729',
              'width': 728}}}}],
         'reddit_video_preview': {'dash_url': 'https://v.redd.it/yfvf3q4has111/DASHPlaylist.mpd',
          'duration': 11,
          'fallback_url': 'https://v.redd.it/yfvf3q4has111/DASH_1_2_M',
          'height': 360,
          'hls_url': 'https://v.redd.it/yfvf3q4has111/HLSPlaylist.m3u8',
          'is_gif': True,
          'scrubber_media_url': 'https://v.redd.it/yfvf3q4has111/DASH_600_K',
          'transcoding_status': 'completed',
          'width': 640}},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 28140,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': True,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'gaming',
        'subreddit_id': 't5_2qh03',
        'subreddit_name_prefixed': 'r/gaming',
        'subreddit_subscribers': 18208527,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://b.thumbs.redditmedia.com/iJZF3bi5JUgpYbbdGRCyt4J3WBJUySFhnVu9UfjJ1YE.jpg',
        'thumbnail_height': 78,
        'thumbnail_width': 140,
        'title': "'There are no winners in war, only losers'",
        'ups': 28140,
        'url': 'https://i.imgur.com/WnuTjH7.gif',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'cartoonartist',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528056645.0,
        'created_utc': 1528027845.0,
        'distinguished': None,
        'domain': 'i.imgur.com',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o84py',
        'is_crosspostable': False,
        'is_reddit_media_domain': False,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': None,
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o84py',
        'no_follow': False,
        'num_comments': 344,
        'num_crossposts': 5,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/comics/comments/8o84py/natural_selection/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'image',
        'preview': {'enabled': True,
         'images': [{'id': 'gRMVFnyJ6Cu0cJJc2hgHkawv5mPmLtyTCapcwm7yAzk',
           'resolutions': [{'height': 180,
             'url': 'https://i.redditmedia.com/THJAgRHIOQ7bo58lcGf39dPfnndoQBiIXbvZ1pfP8gg.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=27c68665cfb39a4925815ca451cd160c',
             'width': 108},
            {'height': 360,
             'url': 'https://i.redditmedia.com/THJAgRHIOQ7bo58lcGf39dPfnndoQBiIXbvZ1pfP8gg.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=61bfdc604d39b4267070f4c6b3f3fcc1',
             'width': 216},
            {'height': 533,
             'url': 'https://i.redditmedia.com/THJAgRHIOQ7bo58lcGf39dPfnndoQBiIXbvZ1pfP8gg.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=9614f24e886732d3eec05c4feda4478b',
             'width': 320},
            {'height': 1066,
             'url': 'https://i.redditmedia.com/THJAgRHIOQ7bo58lcGf39dPfnndoQBiIXbvZ1pfP8gg.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;s=4ad90a36b11088ca712ab5713f077535',
             'width': 640},
            {'height': 1600,
             'url': 'https://i.redditmedia.com/THJAgRHIOQ7bo58lcGf39dPfnndoQBiIXbvZ1pfP8gg.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=960&amp;s=17b28fd7ec5b0d4b1a5ea4e3561a8ac5',
             'width': 960}],
           'source': {'height': 1617,
            'url': 'https://i.redditmedia.com/THJAgRHIOQ7bo58lcGf39dPfnndoQBiIXbvZ1pfP8gg.jpg?s=16d7b4a2de049b2c8fa5966a9d7eda80',
            'width': 970},
           'variants': {}}]},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 20257,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': True,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'comics',
        'subreddit_id': 't5_2qh0s',
        'subreddit_name_prefixed': 'r/comics',
        'subreddit_subscribers': 926031,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://b.thumbs.redditmedia.com/AhvrFwoiV8uwddrKzrMVMOo1QbT87GrLPE2q45nUHhw.jpg',
        'thumbnail_height': 140,
        'thumbnail_width': 140,
        'title': 'Natural selection',
        'ups': 20257,
        'url': 'https://i.imgur.com/NSMz711.jpg',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'Thund3rbolt',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528068973.0,
        'created_utc': 1528040173.0,
        'distinguished': None,
        'domain': 'i.imgur.com',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o99x7',
        'is_crosspostable': False,
        'is_reddit_media_domain': False,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': None,
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o99x7',
        'no_follow': False,
        'num_comments': 130,
        'num_crossposts': 1,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/aww/comments/8o99x7/efficient_and_appealing/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'link',
        'preview': {'enabled': True,
         'images': [{'id': 'Rd5sWjXFyi0KLYrqmnpmX1iyAT5CPoDwQjznDeK8Ka4',
           'resolutions': [{'height': 156,
             'url': 'https://i.redditmedia.com/hUg46RKbWLGQyh6qyemmMCswbW5YKdMj1wWFIoXIIOE.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;fm=jpg&amp;s=59e89c2e34ee0e06421b623d25c0a897',
             'width': 108},
            {'height': 313,
             'url': 'https://i.redditmedia.com/hUg46RKbWLGQyh6qyemmMCswbW5YKdMj1wWFIoXIIOE.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;fm=jpg&amp;s=2e5e832942a32fbea231401e8b0598c9',
             'width': 216},
            {'height': 464,
             'url': 'https://i.redditmedia.com/hUg46RKbWLGQyh6qyemmMCswbW5YKdMj1wWFIoXIIOE.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;fm=jpg&amp;s=900e52551b174501a7220ae2d1edc64c',
             'width': 320}],
           'source': {'height': 668,
            'url': 'https://i.redditmedia.com/hUg46RKbWLGQyh6qyemmMCswbW5YKdMj1wWFIoXIIOE.gif?fm=jpg&amp;s=1ec6b0b20e5178d79aaf59f504efa752',
            'width': 460},
           'variants': {'gif': {'resolutions': [{'height': 156,
               'url': 'https://g.redditmedia.com/hUg46RKbWLGQyh6qyemmMCswbW5YKdMj1wWFIoXIIOE.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=dad89f0e33313eeb9486200abe8d3924',
               'width': 108},
              {'height': 313,
               'url': 'https://g.redditmedia.com/hUg46RKbWLGQyh6qyemmMCswbW5YKdMj1wWFIoXIIOE.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=2a2540789c09a4e408fa23b33765eaba',
               'width': 216},
              {'height': 464,
               'url': 'https://g.redditmedia.com/hUg46RKbWLGQyh6qyemmMCswbW5YKdMj1wWFIoXIIOE.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=97f55cb9ea3f294095a8855f1eacdc7b',
               'width': 320}],
             'source': {'height': 668,
              'url': 'https://g.redditmedia.com/hUg46RKbWLGQyh6qyemmMCswbW5YKdMj1wWFIoXIIOE.gif?s=4f80bf1b8a1fa39c25ea77bfafa6a424',
              'width': 460}},
            'mp4': {'resolutions': [{'height': 156,
               'url': 'https://g.redditmedia.com/hUg46RKbWLGQyh6qyemmMCswbW5YKdMj1wWFIoXIIOE.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=093f98f89e2f541981855c44016bba2b',
               'width': 108},
              {'height': 313,
               'url': 'https://g.redditmedia.com/hUg46RKbWLGQyh6qyemmMCswbW5YKdMj1wWFIoXIIOE.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=75806675c6268c663adaba4ec85aa71f',
               'width': 216},
              {'height': 464,
               'url': 'https://g.redditmedia.com/hUg46RKbWLGQyh6qyemmMCswbW5YKdMj1wWFIoXIIOE.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=b61195f80cd50056be7c716cdd3dc80e',
               'width': 320}],
             'source': {'height': 668,
              'url': 'https://g.redditmedia.com/hUg46RKbWLGQyh6qyemmMCswbW5YKdMj1wWFIoXIIOE.gif?fm=mp4&amp;mp4-fragmented=false&amp;s=ee8f5c7f44ab810e8c350621da430451',
              'width': 460}}}}],
         'reddit_video_preview': {'dash_url': 'https://v.redd.it/iyk8gfjh2t111/DASHPlaylist.mpd',
          'duration': 19,
          'fallback_url': 'https://v.redd.it/iyk8gfjh2t111/DASH_2_4_M',
          'height': 480,
          'hls_url': 'https://v.redd.it/iyk8gfjh2t111/HLSPlaylist.m3u8',
          'is_gif': True,
          'scrubber_media_url': 'https://v.redd.it/iyk8gfjh2t111/DASH_600_K',
          'transcoding_status': 'completed',
          'width': 330}},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 4769,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': True,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'aww',
        'subreddit_id': 't5_2qh1o',
        'subreddit_name_prefixed': 'r/aww',
        'subreddit_subscribers': 17229875,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://b.thumbs.redditmedia.com/m90NV70LRE8-F14VCTdDe6-JizssnPXVsroQYl2O50I.jpg',
        'thumbnail_height': 140,
        'thumbnail_width': 140,
        'title': 'Efficient and Appealing',
        'ups': 4769,
        'url': 'https://i.imgur.com/UZJPQIh.gifv',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'dumbgringo',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528055838.0,
        'created_utc': 1528027038.0,
        'distinguished': None,
        'domain': 'abcnews.go.com',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o82im',
        'is_crosspostable': False,
        'is_reddit_media_domain': False,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': None,
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o82im',
        'no_follow': False,
        'num_comments': 1860,
        'num_crossposts': 1,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/news/comments/8o82im/officer_fired_after_intentionally_hitting_fleeing/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'link',
        'preview': {'enabled': False,
         'images': [{'id': 'P2GOi16DPjsq1jXfzUmVtVk7WGWyOiYLUQdB67iL1FY',
           'resolutions': [{'height': 60,
             'url': 'https://i.redditmedia.com/PRhKk41jMlcUbE_4XjrPnhXmNRx9oJnlFHl2JE_smc8.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=7ed75651db2907a993a6ff5690fe09b2',
             'width': 108},
            {'height': 121,
             'url': 'https://i.redditmedia.com/PRhKk41jMlcUbE_4XjrPnhXmNRx9oJnlFHl2JE_smc8.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=b721da9be1930fb6fa5271c33db6fd2f',
             'width': 216},
            {'height': 180,
             'url': 'https://i.redditmedia.com/PRhKk41jMlcUbE_4XjrPnhXmNRx9oJnlFHl2JE_smc8.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=135080e17a7d2afbe04b3c173126a184',
             'width': 320},
            {'height': 360,
             'url': 'https://i.redditmedia.com/PRhKk41jMlcUbE_4XjrPnhXmNRx9oJnlFHl2JE_smc8.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;s=32e34eb00c994170b5eae061036329cd',
             'width': 640},
            {'height': 540,
             'url': 'https://i.redditmedia.com/PRhKk41jMlcUbE_4XjrPnhXmNRx9oJnlFHl2JE_smc8.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=960&amp;s=231becf5a6772ae59460e757ff848a2a',
             'width': 960}],
           'source': {'height': 558,
            'url': 'https://i.redditmedia.com/PRhKk41jMlcUbE_4XjrPnhXmNRx9oJnlFHl2JE_smc8.jpg?s=a5632c88d0e2ced5c91ba43e791655f9',
            'width': 992},
           'variants': {}}]},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 18516,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': False,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'news',
        'subreddit_id': 't5_2qh3l',
        'subreddit_name_prefixed': 'r/news',
        'subreddit_subscribers': 16093316,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'default',
        'thumbnail_height': 78,
        'thumbnail_width': 140,
        'title': 'Officer fired after intentionally hitting fleeing suspect with his police car.',
        'ups': 18516,
        'url': 'https://abcnews.go.com/US/officer-fired-intentionally-hitting-fleeing-suspect-police-car/story?id=55613845',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'getjiggywithit69',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528057374.0,
        'created_utc': 1528028574.0,
        'distinguished': None,
        'domain': 'i.redd.it',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o86pt',
        'is_crosspostable': False,
        'is_reddit_media_domain': True,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': 'flash1',
        'link_flair_text': 'r/all',
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o86pt',
        'no_follow': False,
        'num_comments': 314,
        'num_crossposts': 1,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': None,
        'permalink': '/r/holdmyredbull/comments/8o86pt/hmrb_literally_as_i_submit_my_last_massive/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'image',
        'preview': {'enabled': True,
         'images': [{'id': '7lWynpH8DBF6-wBPyUwj4T00TvqQQ8NctzMCLtCAtKU',
           'resolutions': [{'height': 80,
             'url': 'https://i.redditmedia.com/bm9pCjQnkL1hhqqnAhRuvQ70NzGmSkcnrm34z5nA3LA.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=c2220d16ef5fb538b50cf35b8f72177e',
             'width': 108},
            {'height': 161,
             'url': 'https://i.redditmedia.com/bm9pCjQnkL1hhqqnAhRuvQ70NzGmSkcnrm34z5nA3LA.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=7c63997acb545f24a8dbacbcccc5908e',
             'width': 216},
            {'height': 238,
             'url': 'https://i.redditmedia.com/bm9pCjQnkL1hhqqnAhRuvQ70NzGmSkcnrm34z5nA3LA.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=748b6017e1ab99e612c46475ada0706b',
             'width': 320},
            {'height': 477,
             'url': 'https://i.redditmedia.com/bm9pCjQnkL1hhqqnAhRuvQ70NzGmSkcnrm34z5nA3LA.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;s=1f0f5550eeb02991ea0d2f4352602e44',
             'width': 640},
            {'height': 716,
             'url': 'https://i.redditmedia.com/bm9pCjQnkL1hhqqnAhRuvQ70NzGmSkcnrm34z5nA3LA.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=960&amp;s=f4eb8a952983289e05a1e3e0d5b300d9',
             'width': 960},
            {'height': 806,
             'url': 'https://i.redditmedia.com/bm9pCjQnkL1hhqqnAhRuvQ70NzGmSkcnrm34z5nA3LA.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=1080&amp;s=14336fb51bc0924194e23ead20a3a668',
             'width': 1080}],
           'source': {'height': 806,
            'url': 'https://i.redditmedia.com/bm9pCjQnkL1hhqqnAhRuvQ70NzGmSkcnrm34z5nA3LA.jpg?s=467c5f4cd8f5c84ccbf08ff0f1fd717a',
            'width': 1080},
           'variants': {}}]},
        'pwls': None,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 15961,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': False,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'holdmyredbull',
        'subreddit_id': 't5_30dl9',
        'subreddit_name_prefixed': 'r/holdmyredbull',
        'subreddit_subscribers': 243711,
        'subreddit_type': 'public',
        'suggested_sort': 'confidence',
        'thumbnail': 'https://a.thumbs.redditmedia.com/4Ext4xQoFy-yCLIhvYvRej0ynXqc7HnnM5Gvm2GrzV0.jpg',
        'thumbnail_height': 104,
        'thumbnail_width': 140,
        'title': 'HMRB (literally) as I submit my last massive assignment ever for my degree with 7 seconds to spare',
        'ups': 15961,
        'url': 'https://i.redd.it/xcvcnj6y3s111.jpg',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': None,
        'wls': None},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'KadenCG',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528064427.0,
        'created_utc': 1528035627.0,
        'distinguished': None,
        'domain': 'v.redd.it',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o8teq',
        'is_crosspostable': False,
        'is_reddit_media_domain': True,
        'is_self': False,
        'is_video': True,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': None,
        'locked': False,
        'media': {'reddit_video': {'dash_url': 'https://v.redd.it/gbktcawuos111/DASHPlaylist.mpd',
          'duration': 10,
          'fallback_url': 'https://v.redd.it/gbktcawuos111/DASH_600_K',
          'height': 240,
          'hls_url': 'https://v.redd.it/gbktcawuos111/HLSPlaylist.m3u8',
          'is_gif': True,
          'scrubber_media_url': 'https://v.redd.it/gbktcawuos111/DASH_600_K',
          'transcoding_status': 'completed',
          'width': 242}},
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o8teq',
        'no_follow': False,
        'num_comments': 78,
        'num_crossposts': 0,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/tippytaps/comments/8o8teq/excited_boy_plays_fetch/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'hosted:video',
        'preview': {'enabled': False,
         'images': [{'id': '1cGi_mzP69Dq1mAzf12IwqlodGe72fXBfrj3BmAUclk',
           'resolutions': [{'height': 107,
             'url': 'https://i.redditmedia.com/I8IeSRsZCSbS0aMK9aPjwWXfpitTr536YBTBqFawoyE.png?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=fc2d907b8da62fe18dfa64bb2af8fea3',
             'width': 108},
            {'height': 214,
             'url': 'https://i.redditmedia.com/I8IeSRsZCSbS0aMK9aPjwWXfpitTr536YBTBqFawoyE.png?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=d90ca8a6175443a28db9a85f4be9d3dd',
             'width': 216},
            {'height': 318,
             'url': 'https://i.redditmedia.com/I8IeSRsZCSbS0aMK9aPjwWXfpitTr536YBTBqFawoyE.png?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=1731e742e6964678e6ec47bd475fe5a8',
             'width': 320}],
           'source': {'height': 318,
            'url': 'https://i.redditmedia.com/I8IeSRsZCSbS0aMK9aPjwWXfpitTr536YBTBqFawoyE.png?s=4bfc5970fcd0bd14462279b5a1d584c6',
            'width': 320},
           'variants': {}}]},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 6154,
        'secure_media': {'reddit_video': {'dash_url': 'https://v.redd.it/gbktcawuos111/DASHPlaylist.mpd',
          'duration': 10,
          'fallback_url': 'https://v.redd.it/gbktcawuos111/DASH_600_K',
          'height': 240,
          'hls_url': 'https://v.redd.it/gbktcawuos111/HLSPlaylist.m3u8',
          'is_gif': True,
          'scrubber_media_url': 'https://v.redd.it/gbktcawuos111/DASH_600_K',
          'transcoding_status': 'completed',
          'width': 242}},
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': True,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'tippytaps',
        'subreddit_id': 't5_3frqi',
        'subreddit_name_prefixed': 'r/tippytaps',
        'subreddit_subscribers': 235396,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://a.thumbs.redditmedia.com/TuO9nDmEHOHngJ8KW7nq65YewQuMaAE3j9mAz5pQ3m8.jpg',
        'thumbnail_height': 139,
        'thumbnail_width': 140,
        'title': 'Excited boy plays fetch',
        'ups': 6154,
        'url': 'https://v.redd.it/gbktcawuos111',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'guyi567',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528058570.0,
        'created_utc': 1528029770.0,
        'distinguished': None,
        'domain': 'i.imgur.com',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o8a3w',
        'is_crosspostable': False,
        'is_reddit_media_domain': False,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': 'approve',
        'link_flair_text': '/r/ALL',
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o8a3w',
        'no_follow': False,
        'num_comments': 117,
        'num_crossposts': 4,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/interestingasfuck/comments/8o8a3w/magnet_collisions_in_slo_mo_look_like_iron_man/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'link',
        'preview': {'enabled': True,
         'images': [{'id': 'mUZ818lqHroBxgM0TP5ji0X57rLFEieYy5LtD9ngW8o',
           'resolutions': [{'height': 60,
             'url': 'https://i.redditmedia.com/IBcyyT3eMi1y2qeEmFNuqdO4qcKNIsrrz2uPZQbLwDc.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;fm=jpg&amp;s=278e11156f8374f8ca2639c4be23b967',
             'width': 108},
            {'height': 121,
             'url': 'https://i.redditmedia.com/IBcyyT3eMi1y2qeEmFNuqdO4qcKNIsrrz2uPZQbLwDc.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;fm=jpg&amp;s=758975979a4166945c4ff3a5f7c29630',
             'width': 216},
            {'height': 179,
             'url': 'https://i.redditmedia.com/IBcyyT3eMi1y2qeEmFNuqdO4qcKNIsrrz2uPZQbLwDc.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;fm=jpg&amp;s=f22a834a75be70698c823f732bcf1b80',
             'width': 320},
            {'height': 358,
             'url': 'https://i.redditmedia.com/IBcyyT3eMi1y2qeEmFNuqdO4qcKNIsrrz2uPZQbLwDc.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;fm=jpg&amp;s=23899714041f46507e235be32bce65e1',
             'width': 640}],
           'source': {'height': 408,
            'url': 'https://i.redditmedia.com/IBcyyT3eMi1y2qeEmFNuqdO4qcKNIsrrz2uPZQbLwDc.gif?fm=jpg&amp;s=f56090b5e6f0b50d64b3d0a9fbc20394',
            'width': 728},
           'variants': {'gif': {'resolutions': [{'height': 60,
               'url': 'https://g.redditmedia.com/IBcyyT3eMi1y2qeEmFNuqdO4qcKNIsrrz2uPZQbLwDc.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=00148cf1087c4d3583581d63e5cd5b11',
               'width': 108},
              {'height': 121,
               'url': 'https://g.redditmedia.com/IBcyyT3eMi1y2qeEmFNuqdO4qcKNIsrrz2uPZQbLwDc.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=3b8f3b6846f668d36ef64b54cf75dae5',
               'width': 216},
              {'height': 179,
               'url': 'https://g.redditmedia.com/IBcyyT3eMi1y2qeEmFNuqdO4qcKNIsrrz2uPZQbLwDc.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=d927cd054722a70515703f89f309c053',
               'width': 320},
              {'height': 358,
               'url': 'https://g.redditmedia.com/IBcyyT3eMi1y2qeEmFNuqdO4qcKNIsrrz2uPZQbLwDc.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;s=bacf5ee9185d5533ae5f1adb47bb5379',
               'width': 640}],
             'source': {'height': 408,
              'url': 'https://g.redditmedia.com/IBcyyT3eMi1y2qeEmFNuqdO4qcKNIsrrz2uPZQbLwDc.gif?s=8f3e74e74d65020de264afd98c2f1a49',
              'width': 728}},
            'mp4': {'resolutions': [{'height': 60,
               'url': 'https://g.redditmedia.com/IBcyyT3eMi1y2qeEmFNuqdO4qcKNIsrrz2uPZQbLwDc.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=bc51c6abd8629f80f2bc6ad37fcc74fd',
               'width': 108},
              {'height': 121,
               'url': 'https://g.redditmedia.com/IBcyyT3eMi1y2qeEmFNuqdO4qcKNIsrrz2uPZQbLwDc.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=c72a328796291153df835778f54921e7',
               'width': 216},
              {'height': 179,
               'url': 'https://g.redditmedia.com/IBcyyT3eMi1y2qeEmFNuqdO4qcKNIsrrz2uPZQbLwDc.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=80fb5eadf4ed6d6ce0fbf9d53478235c',
               'width': 320},
              {'height': 358,
               'url': 'https://g.redditmedia.com/IBcyyT3eMi1y2qeEmFNuqdO4qcKNIsrrz2uPZQbLwDc.gif?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;fm=mp4&amp;mp4-fragmented=false&amp;s=cadb059d7e7048cdcdc85c9d053bfdaa',
               'width': 640}],
             'source': {'height': 408,
              'url': 'https://g.redditmedia.com/IBcyyT3eMi1y2qeEmFNuqdO4qcKNIsrrz2uPZQbLwDc.gif?fm=mp4&amp;mp4-fragmented=false&amp;s=ed424fe7641ebb6fc24b57044508f786',
              'width': 728}}}}],
         'reddit_video_preview': {'dash_url': 'https://v.redd.it/1fwbegsi7s111/DASHPlaylist.mpd',
          'duration': 10,
          'fallback_url': 'https://v.redd.it/1fwbegsi7s111/DASH_1_2_M',
          'height': 360,
          'hls_url': 'https://v.redd.it/1fwbegsi7s111/HLSPlaylist.m3u8',
          'is_gif': True,
          'scrubber_media_url': 'https://v.redd.it/1fwbegsi7s111/DASH_600_K',
          'transcoding_status': 'completed',
          'width': 640}},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 11085,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': True,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'interestingasfuck',
        'subreddit_id': 't5_2qhsa',
        'subreddit_name_prefixed': 'r/interestingasfuck',
        'subreddit_subscribers': 2193400,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://b.thumbs.redditmedia.com/LtmkDHEcbKGdHFkurcfiPl3xY4X148NozuF73hGg6EA.jpg',
        'thumbnail_height': 78,
        'thumbnail_width': 140,
        'title': 'Magnet collisions in Slo Mo look like Iron Man suiting up.',
        'ups': 11085,
        'url': 'https://i.imgur.com/1BbjqY8.gifv',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'priyankerrao',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528056059.0,
        'created_utc': 1528027259.0,
        'distinguished': None,
        'domain': 'youtube.com',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o833g',
        'is_crosspostable': False,
        'is_reddit_media_domain': False,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': None,
        'locked': False,
        'media': {'oembed': {'author_name': '212 Degrees of Love ***FAN PAGE***',
          'author_url': 'https://www.youtube.com/channel/UC2FyMc8pafrXsH6mOc9ha-w',
          'height': 338,
          'html': '&lt;iframe width="600" height="338" src="https://www.youtube.com/embed/ctq-Gz9r15g?feature=oembed&amp;enablejsapi=1" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen&gt;&lt;/iframe&gt;',
          'provider_name': 'YouTube',
          'provider_url': 'https://www.youtube.com/',
          'thumbnail_height': 360,
          'thumbnail_url': 'https://i.ytimg.com/vi/ctq-Gz9r15g/hqdefault.jpg',
          'thumbnail_width': 480,
          'title': 'Baby leopard lets out mighty roar',
          'type': 'video',
          'version': '1.0',
          'width': 600},
         'type': 'youtube.com'},
        'media_embed': {'content': '&lt;iframe width="600" height="338" src="https://www.youtube.com/embed/ctq-Gz9r15g?feature=oembed&amp;enablejsapi=1" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen&gt;&lt;/iframe&gt;',
         'height': 338,
         'scrolling': False,
         'width': 600},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o833g',
        'no_follow': False,
        'num_comments': 258,
        'num_crossposts': 7,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/videos/comments/8o833g/baby_leopard_lets_out_mighty_roar/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'rich:video',
        'preview': {'enabled': False,
         'images': [{'id': 'IiT_ZZGh1etw_qNtP7NXi1NLgCDS_a-X3JPCwQwCLYk',
           'resolutions': [{'height': 81,
             'url': 'https://i.redditmedia.com/fGWbJEUHcoTRst8wMOjkAlwKYk7-LyLne0kZnqA52Fc.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=b415b2d3d99b6ae57d498237d9a97d0c',
             'width': 108},
            {'height': 162,
             'url': 'https://i.redditmedia.com/fGWbJEUHcoTRst8wMOjkAlwKYk7-LyLne0kZnqA52Fc.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=6642c34cfc7efeae485c73f6ce99b831',
             'width': 216},
            {'height': 240,
             'url': 'https://i.redditmedia.com/fGWbJEUHcoTRst8wMOjkAlwKYk7-LyLne0kZnqA52Fc.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=21b95733d10658406ca656f569bcd38f',
             'width': 320}],
           'source': {'height': 360,
            'url': 'https://i.redditmedia.com/fGWbJEUHcoTRst8wMOjkAlwKYk7-LyLne0kZnqA52Fc.jpg?s=ddc16595b56f3b725598d609c95a6b80',
            'width': 480},
           'variants': {}}]},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 11942,
        'secure_media': {'oembed': {'author_name': '212 Degrees of Love ***FAN PAGE***',
          'author_url': 'https://www.youtube.com/channel/UC2FyMc8pafrXsH6mOc9ha-w',
          'height': 338,
          'html': '&lt;iframe width="600" height="338" src="https://www.youtube.com/embed/ctq-Gz9r15g?feature=oembed&amp;enablejsapi=1" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen&gt;&lt;/iframe&gt;',
          'provider_name': 'YouTube',
          'provider_url': 'https://www.youtube.com/',
          'thumbnail_height': 360,
          'thumbnail_url': 'https://i.ytimg.com/vi/ctq-Gz9r15g/hqdefault.jpg',
          'thumbnail_width': 480,
          'title': 'Baby leopard lets out mighty roar',
          'type': 'video',
          'version': '1.0',
          'width': 600},
         'type': 'youtube.com'},
        'secure_media_embed': {'content': '&lt;iframe width="600" height="338" src="https://www.youtube.com/embed/ctq-Gz9r15g?feature=oembed&amp;enablejsapi=1" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen&gt;&lt;/iframe&gt;',
         'height': 338,
         'media_domain_url': 'https://www.redditmedia.com/mediaembed/8o833g',
         'scrolling': False,
         'width': 600},
        'selftext': '',
        'selftext_html': None,
        'send_replies': True,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'videos',
        'subreddit_id': 't5_2qh1e',
        'subreddit_name_prefixed': 'r/videos',
        'subreddit_subscribers': 17820535,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://a.thumbs.redditmedia.com/hcwIAqN0kR3uSryh9_X010EtAC4KwA-mOXuYXbJULg4.jpg',
        'thumbnail_height': 105,
        'thumbnail_width': 140,
        'title': 'Baby leopard lets out mighty roar',
        'ups': 11942,
        'url': 'https://www.youtube.com/watch?v=ctq-Gz9r15g',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'marcsa',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528057370.0,
        'created_utc': 1528028570.0,
        'distinguished': None,
        'domain': 'i.redd.it',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o86pe',
        'is_crosspostable': False,
        'is_reddit_media_domain': True,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': '',
        'link_flair_text': 'r/all choosy begging',
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o86pe',
        'no_follow': False,
        'num_comments': 515,
        'num_crossposts': 0,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'promo_adult_nsfw',
        'permalink': '/r/ChoosingBeggars/comments/8o86pe/just_in_write_me_1_million_word_essay_for_the/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'image',
        'preview': {'enabled': True,
         'images': [{'id': 's_oME7Ypl8hmk7sFLdRp2gwcWEI8boBsNm9NdnwFnKw',
           'resolutions': [{'height': 73,
             'url': 'https://i.redditmedia.com/DZk62gCn66kidaSnriFKPEkiIKm7917yBzGDQxBfdEQ.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=8c7108eeaba2bc2c67011a2a673c778f',
             'width': 108},
            {'height': 147,
             'url': 'https://i.redditmedia.com/DZk62gCn66kidaSnriFKPEkiIKm7917yBzGDQxBfdEQ.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=87f7197679e2823b001f66d50dfc4e02',
             'width': 216},
            {'height': 218,
             'url': 'https://i.redditmedia.com/DZk62gCn66kidaSnriFKPEkiIKm7917yBzGDQxBfdEQ.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=65f2e072ec59060ffa7dccf384e0504f',
             'width': 320},
            {'height': 436,
             'url': 'https://i.redditmedia.com/DZk62gCn66kidaSnriFKPEkiIKm7917yBzGDQxBfdEQ.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=640&amp;s=b28691a25772aca98e943337fd3fe933',
             'width': 640}],
           'source': {'height': 574,
            'url': 'https://i.redditmedia.com/DZk62gCn66kidaSnriFKPEkiIKm7917yBzGDQxBfdEQ.jpg?s=e3baf1796d655c1c5eaefb842a9bea58',
            'width': 842},
           'variants': {}}]},
        'pwls': 3,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 10349,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': True,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'ChoosingBeggars',
        'subreddit_id': 't5_35fmc',
        'subreddit_name_prefixed': 'r/ChoosingBeggars',
        'subreddit_subscribers': 386041,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://b.thumbs.redditmedia.com/IR3i9RgL3QY667N76bpXrrEeBWBIaAUehoWAeSXQM4w.jpg',
        'thumbnail_height': 95,
        'thumbnail_width': 140,
        'title': 'Just in: Write me 1 million word essay for the generous amount of $22',
        'ups': 10349,
        'url': 'https://i.redd.it/1xv4odho3s111.jpg',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'promo_adult_nsfw',
        'wls': 3},
       'kind': 't3'},
      {'data': {'approved_at_utc': None,
        'approved_by': None,
        'archived': False,
        'author': 'webox',
        'author_flair_css_class': None,
        'author_flair_template_id': None,
        'author_flair_text': None,
        'banned_at_utc': None,
        'banned_by': None,
        'can_gild': False,
        'can_mod_post': False,
        'clicked': False,
        'contest_mode': False,
        'created': 1528062803.0,
        'created_utc': 1528034003.0,
        'distinguished': None,
        'domain': 'i.redd.it',
        'downs': 0,
        'edited': False,
        'gilded': 0,
        'hidden': False,
        'hide_score': False,
        'id': '8o8nsm',
        'is_crosspostable': False,
        'is_reddit_media_domain': True,
        'is_self': False,
        'is_video': False,
        'likes': None,
        'link_flair_css_class': None,
        'link_flair_text': None,
        'locked': False,
        'media': None,
        'media_embed': {},
        'media_only': False,
        'mod_note': None,
        'mod_reason_by': None,
        'mod_reason_title': None,
        'mod_reports': [],
        'name': 't3_8o8nsm',
        'no_follow': False,
        'num_comments': 128,
        'num_crossposts': 1,
        'num_reports': None,
        'over_18': False,
        'parent_whitelist_status': 'all_ads',
        'permalink': '/r/WatchPeopleDieInside/comments/8o8nsm/how_did_i_get_here/',
        'pinned': False,
        'post_categories': None,
        'post_hint': 'image',
        'preview': {'enabled': True,
         'images': [{'id': '5DYrQ38N8vcGD3x0Ycw1Sf5Phii0dUqQbTwhCtCYXY8',
           'resolutions': [{'height': 72,
             'url': 'https://i.redditmedia.com/pY74qq731M6V2heLvt3YWgGUlfurjmiVPSEycwFJvSk.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=108&amp;s=7f4aee2b6b4d19287627f16aa6b7c786',
             'width': 108},
            {'height': 145,
             'url': 'https://i.redditmedia.com/pY74qq731M6V2heLvt3YWgGUlfurjmiVPSEycwFJvSk.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=216&amp;s=a84fd16f2c7155142802056474bd645d',
             'width': 216},
            {'height': 214,
             'url': 'https://i.redditmedia.com/pY74qq731M6V2heLvt3YWgGUlfurjmiVPSEycwFJvSk.jpg?fit=crop&amp;crop=faces%2Centropy&amp;arh=2&amp;w=320&amp;s=eef3310c0d89eea795e08effc8f9b4a2',
             'width': 320}],
           'source': {'height': 403,
            'url': 'https://i.redditmedia.com/pY74qq731M6V2heLvt3YWgGUlfurjmiVPSEycwFJvSk.jpg?s=7060ebb8b16c936e8bb485d9acda7172',
            'width': 600},
           'variants': {}}]},
        'pwls': 6,
        'quarantine': False,
        'removal_reason': None,
        'report_reasons': None,
        'saved': False,
        'score': 6019,
        'secure_media': None,
        'secure_media_embed': {},
        'selftext': '',
        'selftext_html': None,
        'send_replies': True,
        'spoiler': False,
        'stickied': False,
        'subreddit': 'WatchPeopleDieInside',
        'subreddit_id': 't5_3h4zq',
        'subreddit_name_prefixed': 'r/WatchPeopleDieInside',
        'subreddit_subscribers': 403162,
        'subreddit_type': 'public',
        'suggested_sort': None,
        'thumbnail': 'https://b.thumbs.redditmedia.com/WjteoMo8IJ0uvbRZte3wOcM_WiRYbvL1WqJh4n0oGYc.jpg',
        'thumbnail_height': 94,
        'thumbnail_width': 140,
        'title': 'How did I get here',
        'ups': 6019,
        'url': 'https://i.redd.it/orqpr3q0vl111.jpg',
        'user_reports': [],
        'view_count': None,
        'visited': False,
        'whitelist_status': 'all_ads',
        'wls': 6},
       'kind': 't3'}],
     'dist': 25,
     'modhash': ''}



#### It looks like all the data I want is in the 'data' item.
For each post, I want to get its name, time posted, score (~ number of upvotes), the subreddit it came from, and how many comments it got.

As can be seen above, the keys for each of these those categories are:
- 'num_comments'
- 'score'
- 'title'
- 'subreddit'
- 'created' (This actually represents when it was posted, and I'm interested in the time elapsed since it was created, so I'll later subtract the time it was created from the current time to get this info.)


### Getting the data I want for an individual post:


```python
# import the datetime library to deal with the time posted
import datetime

# instantiate dictionary of the data about one post
post_scrape = {}

# get one post to demo on
demo_post = json_data['data']['children'][0]

# get the timestamp from a 10-digit format representing the number of seconds since 1970, subtract the current time, and make the results into minutes
post_scrape['mins_since_post'] = round((datetime.datetime.fromtimestamp(demo_post['data']['created'])-datetime.datetime.now()).seconds/60)
# get the rest of the data
post_scrape['num_comments'] = demo_post['data']['num_comments']
post_scrape['score'] = demo_post['data']['score']
post_scrape['title'] = demo_post['data']['title']
post_scrape['subreddit'] = demo_post['data']['subreddit']

# output the dictionary to see if it looks right
post_scrape
```




    {'num_comments': 2,
     'score': 84,
     'subreddit': 'cactus',
     'timestamp': 55,
     'title': 'Found these cute little guys in Jeonju, South Korea'}



###  Turning the above loop into a function so I can call it for each post once I scrape them:


```python
def scrape_post(post):
    post_scrape = {}
    post_scrape['mins_since_post'] = round((datetime.datetime.fromtimestamp(post['data']['created'])-datetime.datetime.now()).seconds/60)
    post_scrape['num_comments'] = post['data']['num_comments']
    post_scrape['score'] = post['data']['score']
    post_scrape['title'] = post['data']['title']
    post_scrape['subreddit'] = post['data']['subreddit']
    return post_scrape
```

### Scraping data about a bunch of posts to give me data to base my analysis on:


```python
## This code snippet scrapes all 25 posts that display on one reddit page, and uses
## the ID of the last post (contained in the after attribute of any post of that page)
## to make sure the next request goes to posts after that post, then scrapes them.
## It adds all the scraped posts to the list [posts].
## Credit to Riley Dallas https://www.linkedin.com/in/rileydallas/
posts = []
after = None
for i in range(100):
    if after == None:
        params = {}
    else:
        params = {'after': after}
    URL = "http://www.reddit.com/hot.json"
    res = requests.get(URL, params=params, headers = headers)
    json_data = res.json()
    posts.extend(json_data['data']['children'])
    after = json_data['data']['after']
    # wait a second between requests to avoid putting excessive load on the servers
    time.sleep(1)
```


```python
# Checking how many posts were scraped
len(posts)
# 2500, as expected since I ran my loop 100 times
```




    2500



### Putting info about all the posts into a big list of dictionaries


```python
# instantiate list
# posts_infodicts_list = []

# call function on each post, add results to [posts_infodicts_list]
for post in posts:
    posts_infodicts_list.append(scrape_post(post))
    
# checking if the list looks right
# posts_infodicts_list[0:5]
```

### Saving my dataframe of information as a CSV
Saving my data to the disk as a Comma-Separated-Values file so that it won't be lost if this notebook crashes


```python
# first I make the list of dictionaries into a DataFrame for easy export
df = pd.DataFrame(posts_infodicts_list)

```


```python
# export to csv in the local directory
df.to_csv('scraped_reddit_posts')
```

# Feature Engineering and Data Prep

#### Using natural language processing to turn the titles into word vectors, which count the occurence of difference words:


```python
# make a corpus of words which includes all words in any of the post's titles
# this will be used to teach the word vectorizer which words to count
corpus = list(df['title']+df['subreddit'])

# import and instantiate CountVectorizer, a word vectorizer which counts the occurence of each word in the corpus
from sklearn.feature_extraction.text import CountVectorizer
cvec = CountVectorizer()

# use the vectorizer to transform the corpus
cvec_corpus = cvec.fit_transform(corpus)

# make a dataframe of the vector counts, after turning them from a sparse matrix to a dense matrix, and label the variables using cvec's get_feature_names() method
vector_counts = pd.DataFrame(cvec_corpus.todense(),columns=cvec.get_feature_names())
```


```python
# do the same as in the above cell, but with just the subreddits
# this will help me make a simpler model below as a proof of concept
subcorpus = list(df['subreddit'])
sub_cvec = CountVectorizer()
sub_cvec_corpus = sub_cvec.fit_transform(subcorpus)
sub_counts = pd.DataFrame(sub_cvec_corpus.todense(),columns=sub_cvec.get_feature_names())
```

#### Making one dataframe out of my scraped data (minus 'title'), and my vectorized word counts:


```python
# dataframe of data without title column or subreddit column since both have been vectorized
df_notitle = df.drop(columns=['title','subreddit'])

# add the dataframes together (concatonate them)
posts_df = pd.concat([df_notitle,vector_counts],  axis = 1)
```

#### How many words are in the corpus anyway?


```python
vector_counts.shape
# apparently there were 6563 unique words in the titles of my posts
```




    (5000, 10790)



#### I want to predict whether the number of comments was low or high. Before I can do that, I need to get the median number of comments and make a variable for whether each post is above or below that. 


```python
import numpy as np
# learning the median number of comments
np.median(posts_df.num_comments)
```




    20.0




```python
# add new column to dataframe
posts_df['above_median'] = posts_df['num_comments'] >= 20

# drop number of comments from the dataframe since using it to predict would be cheating
posts_df = posts_df.drop(columns=['num_comments'])
```

# Predicting 'above_median' Number of Posts With a Random Forest and a Regression Model:

#### Splitting my data into a testing set and a training set so that I can avoid overfitting my model:


```python
# separating the target (y) from the predictors (x). I'm going to make 
# 2 separate x dataframes, one with all the predictors, and one with just subreddit
# so that I can build a simple proof-of-concept model below.

X_subr = sub_counts
X = posts_df.drop(columns=['above_median'])
y = posts_df['above_median']

from sklearn.model_selection import train_test_split
# use train_test_split to make a training a testing set for both Xs
X_train, X_test, y_train, y_test = train_test_split(X,y)
X_train_subr, X_test_subr, y_train_subr, y_test_subr = train_test_split(X_subr,y)
```

#### Creating a Random Forest model to predict High/Low number of comments. Starting with a proof of concept which uses only the subreddit as a feature:


```python
# import packages from scikitlearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# instantiate random forest model
forest = RandomForestClassifier()

# fit model on training data and score it on testing data
forest.fit(X_train_subr, y_train_subr)
forest.score(X_test_subr, y_test_subr)
# I ran this model on ten different train_test_splits to check for overfitting and all scores were >0.6 and <0.67
```




    0.74



#### Making a similar model but using words in the title, minutes since post, and score as predictors as well as subreddit, to see whether considering the words in the title makes the model better:


```python
forest = RandomForestClassifier()

# fit model on training data and score it on testing data
forest.fit(X_train, y_train)
forest.score(X_test, y_test)
# I ran this model on ten different train_test_splits to check for overfitting and all scores were >0.7 and <0.77
```




    0.7664



#### As you can see, the model improved very little when I including the vectorized titles as well as the subreddits. This means that the influence of the words on a post's number of comments appears to be small if it exists. 

Which features were most important to the model?


```python
# make a dataframe which contains the variable names and their relative importances in the model
df_importances = pd.DataFrame({'variable': X_train.columns,
                              'relative_importance': forest.feature_importances_})

# reorder the dataframe based on absolute value - credit for this implementation goes to posted EdChum on this stackoverflow thread: https://stackoverflow.com/questions/30486263/pandas-sort-by-absolute-value-without-changing-the-data#30486411
df_importances.reindex(df_importances['relative_importance'].abs().sort_values(inplace=False, ascending=False).index).head(50)

# below is a table of relative feature importances for the full model. 
# the relative importance is a fraction of 1, where 1 represents the model's total explanatory power
# Score and mins_since_post are clearly the most important variables.
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>relative_importance</th>
      <th>variable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.192166</td>
      <td>score</td>
    </tr>
    <tr>
      <th>0</th>
      <td>0.086844</td>
      <td>mins_since_post</td>
    </tr>
    <tr>
      <th>9682</th>
      <td>0.004962</td>
      <td>to</td>
    </tr>
    <tr>
      <th>9506</th>
      <td>0.004301</td>
      <td>the</td>
    </tr>
    <tr>
      <th>9578</th>
      <td>0.003135</td>
      <td>this</td>
    </tr>
    <tr>
      <th>4840</th>
      <td>0.002912</td>
      <td>in</td>
    </tr>
    <tr>
      <th>5048</th>
      <td>0.002803</td>
      <td>it</td>
    </tr>
    <tr>
      <th>6717</th>
      <td>0.002771</td>
      <td>of</td>
    </tr>
    <tr>
      <th>686</th>
      <td>0.002650</td>
      <td>and</td>
    </tr>
    <tr>
      <th>10541</th>
      <td>0.002598</td>
      <td>with</td>
    </tr>
    <tr>
      <th>10707</th>
      <td>0.002433</td>
      <td>you</td>
    </tr>
    <tr>
      <th>5023</th>
      <td>0.002285</td>
      <td>is</td>
    </tr>
    <tr>
      <th>6410</th>
      <td>0.001926</td>
      <td>my</td>
    </tr>
    <tr>
      <th>3789</th>
      <td>0.001905</td>
      <td>for</td>
    </tr>
    <tr>
      <th>827</th>
      <td>0.001815</td>
      <td>are</td>
    </tr>
    <tr>
      <th>8583</th>
      <td>0.001758</td>
      <td>should</td>
    </tr>
    <tr>
      <th>5218</th>
      <td>0.001718</td>
      <td>just</td>
    </tr>
    <tr>
      <th>5605</th>
      <td>0.001584</td>
      <td>like</td>
    </tr>
    <tr>
      <th>1135</th>
      <td>0.001536</td>
      <td>be</td>
    </tr>
    <tr>
      <th>2665</th>
      <td>0.001528</td>
      <td>de</td>
    </tr>
    <tr>
      <th>4305</th>
      <td>0.001517</td>
      <td>gun</td>
    </tr>
    <tr>
      <th>916</th>
      <td>0.001445</td>
      <td>at</td>
    </tr>
    <tr>
      <th>6779</th>
      <td>0.001440</td>
      <td>on</td>
    </tr>
    <tr>
      <th>3896</th>
      <td>0.001415</td>
      <td>from</td>
    </tr>
    <tr>
      <th>4430</th>
      <td>0.001395</td>
      <td>have</td>
    </tr>
    <tr>
      <th>3605</th>
      <td>0.001377</td>
      <td>femcelsbraincels</td>
    </tr>
    <tr>
      <th>9567</th>
      <td>0.001372</td>
      <td>think</td>
    </tr>
    <tr>
      <th>5981</th>
      <td>0.001369</td>
      <td>me</td>
    </tr>
    <tr>
      <th>7463</th>
      <td>0.001365</td>
      <td>pride</td>
    </tr>
    <tr>
      <th>4050</th>
      <td>0.001358</td>
      <td>get</td>
    </tr>
    <tr>
      <th>2931</th>
      <td>0.001335</td>
      <td>do</td>
    </tr>
    <tr>
      <th>10056</th>
      <td>0.001278</td>
      <td>up</td>
    </tr>
    <tr>
      <th>9616</th>
      <td>0.001269</td>
      <td>through</td>
    </tr>
    <tr>
      <th>4591</th>
      <td>0.001268</td>
      <td>hmmmhmmm</td>
    </tr>
    <tr>
      <th>10446</th>
      <td>0.001259</td>
      <td>when</td>
    </tr>
    <tr>
      <th>4412</th>
      <td>0.001257</td>
      <td>has</td>
    </tr>
    <tr>
      <th>9936</th>
      <td>0.001239</td>
      <td>two</td>
    </tr>
    <tr>
      <th>733</th>
      <td>0.001230</td>
      <td>anon</td>
    </tr>
    <tr>
      <th>484</th>
      <td>0.001223</td>
      <td>actual</td>
    </tr>
    <tr>
      <th>10495</th>
      <td>0.001202</td>
      <td>will</td>
    </tr>
    <tr>
      <th>9497</th>
      <td>0.001158</td>
      <td>that</td>
    </tr>
    <tr>
      <th>6715</th>
      <td>0.001150</td>
      <td>odroid</td>
    </tr>
    <tr>
      <th>5205</th>
      <td>0.001113</td>
      <td>jump</td>
    </tr>
    <tr>
      <th>3933</th>
      <td>0.001109</td>
      <td>furry</td>
    </tr>
    <tr>
      <th>4154</th>
      <td>0.001106</td>
      <td>golden</td>
    </tr>
    <tr>
      <th>9077</th>
      <td>0.001104</td>
      <td>still</td>
    </tr>
    <tr>
      <th>6699</th>
      <td>0.001090</td>
      <td>oc</td>
    </tr>
    <tr>
      <th>1589</th>
      <td>0.001087</td>
      <td>bruce</td>
    </tr>
    <tr>
      <th>1664</th>
      <td>0.001086</td>
      <td>bustedfashionreps</td>
    </tr>
    <tr>
      <th>1028</th>
      <td>0.001085</td>
      <td>back</td>
    </tr>
  </tbody>
</table>
</div>



#### Repeat the model-building process with a non-tree-based method.


```python
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Standard scaler helps prepare data for a logistic regression model
# by putting all numerical predictors on a similar scale
ss = StandardScaler()
ss.fit_transform(X_train)

# instantiate model
logreg = LogisticRegression()

#logreg.fit(X_train, y_train)
#logreg.score(X_test,y_test)

grid_params = {
    'penalty': ['l1','l2'],
    'C': [.8,.9,1]
}
grid = GridSearchCV(logreg, grid_params)
grid.fit(X_train,y_train)
print(grid.score(X_test,y_test))
print(grid.best_params_)
```

    0.7656
    {'C': 0.9, 'penalty': 'l1'}

