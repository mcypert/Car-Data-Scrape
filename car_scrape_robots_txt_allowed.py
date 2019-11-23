# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 18:26:11 2019

@author: mcype

This was my first ever python program to scrape data from https://www.auto-data.net/en.

"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import random

def get_brand_links():
    url = "https://www.auto-data.net/en/allbrands"
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    
    sec_1, = soup.find('div', attrs = {'id' : 'center'})
    sec_2 = sec_1.find('div', attrs = {'class': 'markite'})
    
    links = []
    for link in sec_2.find_all('a', attrs = {'class': 'marki_blok'}):
        links.append(link.get("href"))
        
    return links
    
def get_model_links(agent_list):
    links = get_brand_links(url = "https://www.auto-data.net/en/allbrands")
    full_links = []
    for end_link in links:
        full_links.append("https://www.auto-data.net" + end_link)
        
    # Loop through every url returned by the get_brand_links function
    for full_link in full_links:
        url = full_link
        page = requests.get(url).text
        soup = BeautifulSoup(page, 'html.parser')
        
        sec_1 = soup.find('div', attrs = {'id': 'center'})
        sec_2 = sec_1.find('div', attrs = {'class': 'markite'})
        
        links = []
        for link in sec_2.find_all('a', attrs = {'class': 'modeli'}):
            links.append(link.get('href'))
            
    return full_links

def get_proxies():
    
    url = "https://free-proxy-list.net/"
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser') 
    
    prox_list = []
    for tbody in soup.find_all('tbody'):
        for td in tbody.find_all('td'):
            prox_list.append(td.text)
    
    ip_list = [prox for index, prox in enumerate(prox_list) if index % 8 == 0]
    port_list = [port for index, port in enumerate(prox_list[1:]) if index % 8 == 0]
    anon_list = [anon for index, anon in enumerate(prox_list[4:]) if index % 8 == 0]
    https_list = [https for index, https in enumerate(prox_list[6:]) if index % 8 == 0]
  
    ip_df = pd.DataFrame({'IP': ip_list, 'Port': port_list, 'Anonymity': anon_list, 'HTTPS': https_list})
    ip_https = ip_df[(ip_df['HTTPS'] == 'yes') & (ip_df['Anonymity'] != 'transparent')]
    ip_list = list(ip_https[['IP', 'Port']].apply(lambda x: ':'.join(x), axis = 1).drop_duplicates())
    
    return ip_list

def get_models():
    
    user_agents = pd.read_csv("My Computer Data/user_agents.csv")
    agent_list = user_agents['UserAgents'].tolist()
    
    proxies = get_proxies()
    links = get_brand_links()
    full_links = ["https://www.auto-data.net" + end_link for end_link in links]

    models = []
    for url in full_links:
        while True:
            if len(proxies) == 0:
                proxies = get_proxies()
            
            user_agent = random.choice(agent_list)
            headers = {'User-Agent': user_agent}
            proxie = random.choice(proxies)
            
            try:
                page = requests.get(url, headers = headers, proxies = {'https': proxie, 'http': proxie}, timeout = 5)
                if page.status_code == 200:
                    break
            except requests.Timeout as e:
                print(str(e))
                proxies.remove(proxie)
                print(len(proxies))
            except requests.ConnectionError as e:
                print(str(e))
                proxies.remove(proxie)
                print(len(proxies))
            except requests.RequestException as e:
                print(str(e))
                proxies.remove(proxie)
                print(len(proxies))
            except KeyboardInterrupt:
                print("Someone closed the program")

        page = page.text
        soup = BeautifulSoup(page, 'html.parser')
        sec_1 = soup.find('div', attrs = {'id': 'center'})
        sec_2 = sec_1.find('div', attrs = {'class': 'markite'})
    
        for link in sec_2.find_all('a', attrs = {'class': 'modeli'}):
            models.append(link.get('href'))
        proxies.remove(proxie)
        print(len(proxies))
        
    return models
            
def get_model_details():
    
    urls = pd.read_csv("My Computer Data/full_urls.csv")
    user_agents = pd.read_csv("My Computer Data//user_agents.csv")
    agent_list = user_agents['UserAgents'].tolist()    
    
    cars = []
    photos = []
    for index, url in enumerate(urls['Url'].tolist()):
        while True:

            user_agent = random.choice(agent_list)
            headers = {'User-Agent': user_agent}      
            
            try:
                page = requests.get(url, headers = headers, timeout = 5)
                if page.status_code == 200:
                    break
            except requests.Timeout as e:
                print(str(e))
            except requests.ConnectionError as e:
                print(str(e))
            except requests.RequestException as e:
                print(str(e))
            except KeyboardInterrupt:
                print("Someone closed the program")      
                
        page = page.text
        soup = BeautifulSoup(page, 'html.parser')    
        
        for td in soup.find_all('td'):
            for a in td.find_all('a'):
                line = a.get('href')
                if 'photo' in line.lower():
                    photos.append(line)
                else:
                    cars.append(line)
        
        print("Success on: " + str(index) + " ", url)
        
    return cars, photos
 
def get_final_df():
    
    urls = pd.read_csv("My Computer Data/car_urls.csv")
    bad_urls = []      
    for index, url in enumerate(urls['Car'].tolist()):
        try:
            car_df = pd.read_html(url, index_col = 0)[0].T
            if index == 0:
                print("Data Frame Start : ")
                final_df = car_df
            else:
                final_df = pd.concat([final_df, car_df], axis = 0, sort = False)
        except:
            print("Error " + url)
            bad_urls.append(url)
    
        print("DataFrame Row: " + str(index))
        
    return final_df, bad_urls

car_df = get_final_df()
car_df.to_csv("My Computer Data/car_df.csv")