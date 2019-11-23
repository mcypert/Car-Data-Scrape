# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 19:20:08 2019

@author: mcype
"""
#cars_df.to_csv("C:/Users/mcype/Documents/Data Science Projects/Full Car Project/clean_car_df.csv")

#libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re


#options
pd.set_option('display.max_rows', 130)
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 200)
plt.style.use('bmh')

#functions
def find_non_alpha(dataframe, column):
    """
    Function that returns all the non-alphanumeric characters in a given dataframe column
    """
    identify_nonalpha = [re.findall(r'\W+', str(i)) for i in dataframe[column].unique().tolist()]
    nonalpha_chars = set([j for i in identify_nonalpha for j in i])
    return nonalpha_chars

def get_replace(string, substitutions):
    """
    Function that allows me to replace multiple characters with one function call
    """
    substrings = sorted(substitutions, key=len, reverse=True)
    regex = re.compile('|'.join(map(re.escape, substrings)))
    return regex.sub(lambda match: substitutions[match.group(0)], string)

def clean_avg(dataframe, column, split):
    """
    Function that takes columns with ranges and averages the range per index (row) based on a split character, i.e, '-'
    """
    dataframe[column] = dataframe[column].astype(str)
    num_cols = dataframe.loc[dataframe[column].str.find(split) != -1, column].dropna().str.split(split, expand=True).shape[1]
    for index in dataframe.loc[dataframe[column].str.find(split) != -1, column].dropna().index:
        dataframe.at[index, column] = np.sum([float(i) for i in dataframe.loc[index, column].split(split)]) / num_cols

"""
load the data:  I scraped the data from https://www.auto-data.net during the weekend of March 15, 2018.
"""

full_cars_dataframe = pd.read_csv('C:/Users/mcype/Documents/Full Car Project/car_df.csv')
cars_df = full_cars_dataframe.copy()

#for this study i am going exclude the columns with more than 50% missing values.
cars_df.shape
keep_cols = pd.Series(cars_df.isnull().sum() / cars_df.shape[0]).sort_values(ascending=False)
keep_cols = keep_cols[keep_cols < .50].index.tolist()
cars_df = cars_df[keep_cols]

#for this study i am going to exclude the columns with more than 50% missing rows.
keep_rows = pd.Series(cars_df.T.isnull().sum() / cars_df.shape[1]).sort_values(ascending=False)      
keep_rows = keep_rows[keep_rows < .50].index.tolist()  
cars_df = cars_df.iloc[keep_rows]

"""
i will first attempt to clean the data before handling the rest of the missing values.  
my goal is to find a solution where i can impute missing values across the whole dataframe is 
accurate as possible.  but the first step is cleaning the data into a form that i can work with.  
this data came from an online car database with little to no restrictions on how people input values so 
the data is really messy.    
"""
#investigate the year column and only include cars that are produced in year 1990 or greater.  
cars_df['Year of putting into production'].value_counts()
#clean up the year column
cars_df['Year of putting into production'] = (pd.to_numeric(cars_df['Year of putting into production']
                                                           .fillna(' ').apply(lambda x: re.sub('year', '', x).strip())
                                                           .replace(' ', np.nan), errors='coerce'))

cars_df['Year of putting into production'].value_counts(dropna=False)
cars_df = cars_df.loc[cars_df['Year of putting into production'] >= 1990, :]

#investigate the Brand column.
cars_df['Brand'] 
cars_df['Brand'].isnull().sum() 

#investigate the Model column.
cars_df['Model'] 
cars_df['Brand'].isnull().sum() 

#investigate the generation column.
cars_df['Generation'] 
cars_df['Generation'].isnull().sum() 

#investigate the Modification (Engine) column.
cars_df['Modification (Engine)'] 
cars_df['Modification (Engine)'].isnull().sum()  

#investigate the Fuel Type column.
cars_df['Fuel Type']
cars_df['Fuel Type'].isnull().sum() 
cars_df['Fuel Type'].value_counts(dropna=False)

#investigate the Drive wheel column
cars_df['Drive wheel']
cars_df['Drive wheel'].isnull().sum()
cars_df['Drive wheel'].value_counts(dropna=False)

#investigate the Position of cylinders column
cars_df['Position of cylinders']
cars_df['Position of cylinders'].isnull().sum()
cars_df['Position of cylinders'].value_counts(dropna=False)

#investigate the Number of cylinders column
cars_df['Number of cylinders'].value_counts(dropna=False)
cars_df['Number of cylinders'].describe()

#investigate Fuel System column
cars_df['Fuel System']
cars_df['Fuel System'].value_counts(dropna=False)

#investigate Number of valves per cylinder
cars_df['Number of valves per cylinder']
cars_df['Number of valves per cylinder'].describe()
cars_df['Number of valves per cylinder'].isnull().sum() 

#investigate Position of engine column
cars_df['Position of engine']
cars_df['Position of engine'].value_counts(dropna=False)

#investigate Steering type column
cars_df['Steering type']
cars_df['Steering type'].isnull().sum()
cars_df['Steering type'].value_counts(dropna=False)

#investigate Compression ratio column
cars_df['Compression ratio']
cars_df['Compression ratio'].isnull().sum()
cars_df['Compression ratio'].describe()

#investigate Power steering column
cars_df['Power steering']
cars_df['Power steering'].value_counts(dropna=False)

#investigate Power steering column
cars_df['Power steering']
cars_df['Power steering'].value_counts(dropna=False)

#investigate the Power column.  the Power columns needs to be cleaned.
# i am going to split the Power column into a Power HP and Power RPM.
cars_df['Power']
cars_df['Power'].isnull().sum() 

find_non_alpha(cars_df, 'Power')    
cars_df.loc[cars_df['Power'].str.find('>') != -1, 'Power'].dropna()
cars_df.loc[cars_df['Power'].str.find('+') != -1, 'Power'].dropna() 
cars_df.loc[cars_df['Power'].str.find('-') != -1, 'Power'].dropna()
cars_df.loc[cars_df['Power'].str.find('/') != -1, 'Power'].dropna()

cars_df.at[27282, 'Power'] = str(169+41+76)
cars_df.at[27269, 'Power'] = str(152+44+76)
cars_df.at[19014, 'Power'] = str(333+116)
cars_df.at[20364, 'Power'] = str(135+82+95)
cars_df.loc[cars_df['Power'].str.find('+') != -1, 'Power'].dropna() 
find_non_alpha(cars_df, 'Power')

cars_df['Power'] = cars_df['Power'].apply(lambda x: get_replace(str(x), {' ':'', 'hp': '', 'rpm.': '', '>': ''}))
find_non_alpha(cars_df, 'Power')

#split the hp and rpm into two different columns
cars_df['Power hp'], cars_df['Power rpm'] = (cars_df['Power'].str.split('/', expand=True)[0],
                                             cars_df['Power'].str.split('/', expand=True)[1])

#power hp
find_non_alpha(cars_df, 'Power hp')
cars_df['Power hp'] = pd.to_numeric(cars_df['Power hp'], errors='coerce')
cars_df['Power hp'].describe()
cars_df['Power hp'].isnull().sum() / cars_df.shape[0]

#power rpm
find_non_alpha(cars_df, 'Power rpm')
clean_avg(dataframe=cars_df, column='Power rpm', split='-')
cars_df['Power rpm'] = pd.to_numeric(cars_df['Power rpm'], errors='coerce')
cars_df['Power rpm'].describe()
cars_df['Power rpm'].isnull().sum() / cars_df.shape[0]

#drop the regular power from the dataset
cars_df.drop('Power', axis=1, inplace=True)

#investigate the Torque column 
cars_df['Torque']
cars_df['Torque'].isnull().sum()

find_non_alpha(cars_df, 'Torque')
cars_df.loc[cars_df['Torque'].str.find('>') != -1, 'Torque'].dropna()
cars_df.loc[cars_df['Torque'].str.find('(') != -1, 'Torque'].dropna()
cars_df.loc[cars_df['Torque'].str.find('*') != -1, 'Torque'].dropna()
cars_df.loc[cars_df['Torque'].str.find('\t') != -1, 'Torque'].dropna()
cars_df.loc[cars_df['Torque'].str.find('~') != -1, 'Torque'].dropna()
cars_df.loc[cars_df['Torque'].str.find('−') != -1, 'Torque'].dropna()
cars_df.loc[cars_df['Torque'].str.find('\u2009-\u2009') != -1, 'Torque'].dropna()
cars_df.loc[cars_df['Torque'].str.find('\u2009') != -1, 'Torque'].dropna()
cars_df.loc[cars_df['Torque'].str.find('[') != -1, 'Torque'].dropna()
cars_df.loc[cars_df['Torque'].str.find(']') != -1, 'Torque'].dropna()

#edit by index
cars_df.at[15624, 'Torque'] = '2000'
cars_df.at[24293, 'Torque'] = str((270+285)/2.0) + "/1750"
cars_df.at[5985, 'Torque'] = "370/5200"
cars_df.at[560, 'Torque'] = "206/1750"
cars_df.at[564, 'Torque'] = "145/2000"
cars_df.at[7757, 'Torque'] = "245/" + str((1400+4000)/2.0)
cars_df.at[16811, 'Torque'] = '650/' + str((2000+4000+3500)/3.0)

#getting rid of some of the non-alpha numeric characters
#since the numbers in the brackets do not seem to make sense i am just going to get rid of that data:
cars_df['Torque'] = cars_df['Torque'].apply(lambda x: 
    get_replace(str(x), {' ': '', '−': '-', 'Nm': '', 'rpm.': '', '\u2009': '', '\u2009-\u2009': ''}))
cars_df['Torque'] = cars_df['Torque'].apply(lambda x: re.sub("\[[0-9_]+(])?", '', x))
find_non_alpha(cars_df, 'Torque')

# split the Nm and rpm into two different columns
cars_df['Torque nm'], cars_df['Torque rpm'] = (cars_df['Torque'].str.split('/', expand=True)[0],
                                               cars_df['Torque'].str.split('/', expand=True)[1])

#torque nm
cars_df['Torque nm'].str.split('-', expand=True)
clean_avg(dataframe=cars_df, column='Torque nm', split='-')
cars_df['Torque nm'] = pd.to_numeric(cars_df['Torque nm'], errors='coerce')
cars_df['Torque nm'].describe()
cars_df['Torque nm'].isnull().sum() / cars_df.shape[0]

#torque rpm
cars_df['Torque rpm'].str.split('-', expand=True) 
clean_avg(dataframe=cars_df, column='Torque rpm', split='-')
cars_df['Torque rpm'] = pd.to_numeric(cars_df['Torque rpm'], errors='coerce')
cars_df['Torque rpm'].describe()
find_non_alpha(cars_df, 'Torque rpm')
cars_df['Torque rpm'].isnull().sum() / cars_df.shape[0]

#drop the regular power from the dataset
cars_df.drop('Torque', axis=1, inplace=True)

#investigate the Length column
cars_df['Length']
cars_df['Length'].isnull().sum()

find_non_alpha(cars_df, 'Length')
cars_df.loc[cars_df['Length'].str.find('-') != -1, 'Length'].dropna()
cars_df.loc[cars_df['Length'].str.find('/') != -1, 'Length'].dropna()

cars_df['Length'] = cars_df['Length'].apply(lambda x: get_replace(str(x), {'mm.': '', ' ':''}))
find_non_alpha(cars_df, 'Length')
clean_avg(dataframe=cars_df, column='Length', split='/')
clean_avg(dataframe=cars_df, column='Length', split='-')
cars_df['Length'] = pd.to_numeric(cars_df['Length'], errors='coerce')
cars_df['Length'].describe()
cars_df['Length'].isnull().sum() / cars_df.shape[0]

#investigate the Height column
cars_df['Height']
cars_df['Height'].isnull().sum()

find_non_alpha(cars_df, 'Height')
cars_df.loc[cars_df['Height'].str.find(' - ') != -1, 'Height'].dropna()
cars_df.loc[cars_df['Height'].str.find('-') != -1, 'Height'].dropna()
cars_df.loc[cars_df['Height'].str.find('/') != -1, 'Height'].dropna()

cars_df['Height'] = cars_df['Height'].apply(lambda x: get_replace(str(x), {'mm.': '', ' ':''}))
find_non_alpha(cars_df, 'Height')
clean_avg(dataframe=cars_df, column='Height', split='-')
clean_avg(dataframe=cars_df, column='Height', split='/')
cars_df['Height'] = pd.to_numeric(cars_df['Height'], errors='coerce')
cars_df['Height'].describe()
cars_df['Height'].isnull().sum() / cars_df.shape[0]

#investigate the Width column
cars_df['Width']
cars_df['Width'].isnull().sum()

find_non_alpha(cars_df, 'Width')
cars_df.loc[cars_df['Width'].str.find('-') != -1, 'Width'].dropna()
cars_df.loc[cars_df['Width'].str.find('/') != -1, 'Width'].dropna()

cars_df['Width'] = cars_df['Width'].apply(lambda x: get_replace(str(x), {'mm.': '', ' ':''}))
find_non_alpha(cars_df, 'Width')
clean_avg(dataframe=cars_df, column='Width', split='-')
clean_avg(dataframe=cars_df, column='Width', split='/')
cars_df['Width'] = pd.to_numeric(cars_df['Width'], errors='coerce')
cars_df['Width'].describe()
cars_df['Width'].isnull().sum() / cars_df.shape[0]

#investigate the Kerb Weight columns
cars_df['Kerb Weight']
cars_df['Kerb Weight'].isnull().sum()

find_non_alpha(cars_df, 'Kerb Weight')
cars_df.loc[cars_df['Kerb Weight'].str.find('-') != -1, 'Kerb Weight'].dropna()
cars_df.loc[cars_df['Kerb Weight'].str.find('/') != -1, 'Kerb Weight'].dropna() 

cars_df['Kerb Weight'] = cars_df['Kerb Weight'].apply(lambda x: get_replace(str(x), {'kg.': '', ' ':''}))
find_non_alpha(cars_df, 'Kerb Weight')

# combinations/variations of the value of Kerb Weight
# pattern 1: 970
# pattern 2: 970-1000
# pattern 3: 970-1000/850-900
# pattern 4: 970/1000

#pattern 3
test_list = [_ for _ in cars_df['Kerb Weight'].str.split('-', expand=True)[2].tolist() if _ is not None] # need to get index
index_list = cars_df['Kerb Weight'].str.split('-', expand=True)[2].index.tolist()
value_list = cars_df['Kerb Weight'].str.split('-', expand=True)[2].tolist()
combined_list = [(i, v) for i, v in zip(index_list, value_list)]    
range2_index = [i for i, v in combined_list if v is not None]    
    
for m, i in enumerate(cars_df.loc[range2_index, 'Kerb Weight'].str.split('/')):
    cars_df.at[range2_index[m], 'Kerb Weight'] = (np.sum([int(t) for t in i[0].split('-')])/2 + np.sum([int(t) for t in i[1].split('-')])/2)/2
    
find_non_alpha(cars_df, 'Kerb Weight')
clean_avg(dataframe=cars_df, column='Kerb Weight', split='-') #pattern 2
clean_avg(dataframe=cars_df, column='Kerb Weight', split='/') #pattern 4
cars_df['Kerb Weight'] = pd.to_numeric(cars_df['Kerb Weight'], errors='coerce')
cars_df['Kerb Weight'].describe()
cars_df['Kerb Weight'].isnull().sum() / cars_df.shape[0] 

#investigate the Engine displacement column
cars_df['Engine displacement']
find_non_alpha(cars_df, 'Engine displacement')
cars_df['Engine displacement'] = (pd.to_numeric(cars_df['Engine displacement']
                                                .apply(lambda x: get_replace(str(x), {' ':'', 'cm3':''})), errors='coerce'))
cars_df['Engine displacement'].describe()

#investigate the Wheelbase columns
cars_df['Wheelbase']
cars_df['Wheelbase'].isnull().sum()

find_non_alpha(cars_df, 'Wheelbase')
cars_df.loc[cars_df['Wheelbase'].str.find('-') != -1, 'Wheelbase'].dropna()
cars_df.loc[cars_df['Wheelbase'].str.find('/') != -1, 'Wheelbase'].dropna()
cars_df['Wheelbase'] = cars_df['Wheelbase'].apply(lambda x: get_replace(str(x), {' ':'', 'mm.': ''})) 
clean_avg(dataframe=cars_df, column='Wheelbase', split='/')
clean_avg(dataframe=cars_df, column='Wheelbase', split='-')
cars_df['Wheelbase'] = pd.to_numeric(cars_df['Wheelbase'], errors='coerce') 

cars_df['Wheelbase'].describe()
cars_df['Wheelbase'].isnull().sum() / cars_df.shape[0] 

#investigate Fuel tank volume
cars_df['Fuel tank volume']
cars_df['Fuel tank volume'].isnull().sum()

find_non_alpha(cars_df, 'Fuel tank volume')
cars_df.loc[cars_df['Fuel tank volume'].str.find('+') != -1, 'Fuel tank volume'].dropna()
cars_df.loc[cars_df['Fuel tank volume'].str.find('(') != -1, 'Fuel tank volume'].dropna()
cars_df.loc[cars_df['Fuel tank volume'].str.find(')') != -1, 'Fuel tank volume'].dropna()
cars_df.loc[cars_df['Fuel tank volume'].str.find('-') != -1, 'Fuel tank volume'].dropna()
cars_df.loc[cars_df['Fuel tank volume'].str.find('.') != -1, 'Fuel tank volume'].dropna()

#edit by index
cars_df.at[13360, 'Fuel tank volume'] = str(95+65)
cars_df.at[13361, 'Fuel tank volume'] = str(95+65)
cars_df.at[13358, 'Fuel tank volume'] = str(95+64)
cars_df.at[13356, 'Fuel tank volume'] = str(112+83)
cars_df.at[13359, 'Fuel tank volume'] = str(95+65)
cars_df.at[16390, 'Fuel tank volume'] = str(93+45)

#i am just going to delete the data between the () because it exists in only 23 rows.  the main number i will keep.
cars_df['Fuel tank volume'] = cars_df['Fuel tank volume'].apply(lambda x: re.sub(r'\((.*?)\)', '', str(x)))
cars_df['Fuel tank volume'] = cars_df['Fuel tank volume'].apply(lambda x: get_replace(str(x), {'l': '', ' ': ''})) 

cars_df['Fuel tank volume'].str.split('-', expand=True)
clean_avg(dataframe=cars_df, column='Fuel tank volume', split='-')
cars_df['Fuel tank volume'] = pd.to_numeric(cars_df['Fuel tank volume'], errors='coerce')
cars_df['Fuel tank volume'].describe()
cars_df['Fuel tank volume'].isnull().sum() / cars_df.shape[0]

#investigate Front brakes column
cars_df['Front brakes']
cars_df['Front brakes'].value_counts(dropna=False)

#i am going to reclassify the categories as [Disc, Ventilated discs, Drum] since there does not seem to be enough data in the type + sizes
cars_df['Front brakes'] = cars_df['Front brakes'].apply(lambda x: 
                                                        str(x) if x=='Drum' else re.sub('[,+0-9+mm\.]', '', str(x)).strip())
cars_df['Front brakes'].value_counts(dropna=False)

#investigate Rear brakes column    
cars_df['Rear brakes']
cars_df['Rear brakes'].value_counts(dropna=False)

#i am going to reclassify the categories as [Disc, Ventilated discs, Drum] since there does not seem to be enough data in the type + sizes
cars_df['Rear brakes'] = cars_df['Rear brakes'].apply(lambda x: 
                                                      str(x) if x=='Drum' else re.sub('[,+0-9+mm\.]', '', str(x)).strip())
cars_df['Rear brakes'].value_counts(dropna=False)

#investigate Front track
cars_df['Front track']
cars_df['Front track'].isnull().sum()

find_non_alpha(cars_df, 'Front track')
cars_df.loc[cars_df['Front track'].str.find(' - ') != -1, 'Front track'].dropna()
cars_df.loc[cars_df['Front track'].str.find('-') != -1, 'Front track'].dropna()
cars_df.loc[cars_df['Front track'].str.find('/') != -1, 'Front track'].dropna()

cars_df['Front track'] = cars_df['Front track'].apply(lambda x: get_replace(str(x), {'mm.': '', ' ':''}))
clean_avg(dataframe=cars_df, column='Front track', split='-')
clean_avg(dataframe=cars_df, column='Front track', split='/')
cars_df['Front track'] = pd.to_numeric(cars_df['Front track'], errors='coerce')
cars_df['Front track'].describe()
cars_df['Front track'].isnull().sum() / cars_df.shape[0]

#investigate Rear (Back) track column
cars_df['Rear (Back) track']
cars_df['Rear (Back) track'].isnull().sum()

find_non_alpha(cars_df, 'Rear (Back) track')
cars_df.loc[cars_df['Rear (Back) track'].str.find(' - ') != -1, 'Rear (Back) track'].dropna()
cars_df.loc[cars_df['Rear (Back) track'].str.find('/') != -1, 'Rear (Back) track'].dropna()

cars_df['Rear (Back) track'] = cars_df['Rear (Back) track'].apply(lambda x: get_replace(str(x), {'mm.': '', ' ':''})) 
clean_avg(dataframe=cars_df, column='Rear (Back) track', split='-')
clean_avg(dataframe=cars_df, column='Rear (Back) track', split='/')
cars_df['Rear (Back) track'] = pd.to_numeric(cars_df['Rear (Back) track'], errors='coerce')
cars_df['Rear (Back) track'].describe()
cars_df['Rear (Back) track'].isnull().sum() / cars_df.shape[0]

#investigate ABS column
cars_df['ABS']
cars_df['ABS'].value_counts(dropna=False)
cars_df['ABS'].isnull().sum()

#reclassifying the nan values to no
cars_df['ABS'] = cars_df['ABS'].apply(lambda x: 'no' if x!='yes' else x)

#investigate Tire size column
cars_df['Tire size']
cars_df['Tire size'].isnull().sum()
cars_df['Tire size'].value_counts(dropna=False)
#since there are so many different tire categories and the data is all over the place, i am going to drop the column for right now.
cars_df.drop('Tire size', axis=1, inplace=True)

#investigate Maximum speed column
cars_df['Maximum speed']
cars_df['Maximum speed'].isnull().sum()

find_non_alpha(cars_df, 'Maximum speed')
cars_df.loc[cars_df['Maximum speed'].str.find('+') != -1, 'Maximum speed'].dropna()
cars_df.loc[cars_df['Maximum speed'].str.find('>') != -1, 'Maximum speed'].dropna()
cars_df.loc[cars_df['Maximum speed'].str.find('-') != -1, 'Maximum speed'].dropna()

cars_df['Maximum speed'] = cars_df['Maximum speed'].apply(lambda x: get_replace(str(x), {'+': '', '>': '', 'km/h': '', ' ': ''})) 
clean_avg(dataframe=cars_df, column='Maximum speed', split='-')
cars_df['Maximum speed'] = pd.to_numeric(cars_df['Maximum speed'], errors='coerce')
cars_df['Maximum speed'].describe()
cars_df['Maximum speed'].isnull().sum() / cars_df.shape[0]

#investigate Piston Stroke column
cars_df['Piston Stroke']
cars_df['Piston Stroke'].isnull().sum()

find_non_alpha(cars_df, 'Piston Stroke')
cars_df['Piston Stroke'] = pd.to_numeric(cars_df['Piston Stroke'].apply(lambda x: get_replace(str(x), {'mm.': '', ' ': ''})), errors='coerce')
cars_df['Piston Stroke'].describe()
cars_df['Piston Stroke'].isnull().sum() / cars_df.shape[0]
cars_df.shape[0] - 5544

#investigate Cylinder Bore column
cars_df['Cylinder Bore']
cars_df['Cylinder Bore'].isnull().sum()

find_non_alpha(cars_df, 'Cylinder Bore')
cars_df['Cylinder Bore'] = pd.to_numeric(cars_df['Cylinder Bore'].apply(lambda x: get_replace(str(x), {'mm.': '', ' ': ''})), errors='coerce')
cars_df['Cylinder Bore'].describe()
cars_df['Cylinder Bore'].isnull().sum() / cars_df.shape[0]

#investigate Year of stopping production column
cars_df['Year of stopping production']
cars_df['Year of stopping production'].value_counts(dropna=False)

cars_df['Year of stopping production'] = (pd.to_numeric(cars_df['Year of stopping production'].fillna(' ')
                                                        .apply(lambda x: re.sub('year', '', x).strip()).replace(' ', np.nan)
                                                        , errors='coerce'))
cars_df['Year of stopping production'].value_counts(dropna=False)

#investigate Minimum volume of Luggage (trunk)
cars_df['Minimum volume of Luggage (trunk)']
cars_df['Minimum volume of Luggage (trunk)'].isnull().sum()

find_non_alpha(cars_df, 'Minimum volume of Luggage (trunk)')
cars_df.loc[cars_df['Minimum volume of Luggage (trunk)'].str.find('+') != -1, 'Minimum volume of Luggage (trunk)'].dropna()
cars_df.loc[cars_df['Minimum volume of Luggage (trunk)'].str.find('-') != -1, 'Minimum volume of Luggage (trunk)'].dropna()
cars_df.loc[cars_df['Minimum volume of Luggage (trunk)'].str.find('/') != -1, 'Minimum volume of Luggage (trunk)'].dropna()
cars_df.loc[cars_df['Minimum volume of Luggage (trunk)'].str.find('.') != -1, 'Minimum volume of Luggage (trunk)'].dropna()

#edit by index
cars_df.at[25256, 'Minimum volume of Luggage (trunk)'] = str(125+160)
cars_df.at[25257, 'Minimum volume of Luggage (trunk)'] = str(125+160)
cars_df['Minimum volume of Luggage (trunk)'] = cars_df['Minimum volume of Luggage (trunk)'].apply(lambda x: get_replace(str(x), {' ': '', 'l': ''})) 

# combinations/variations of the value of Kerb Weight
# pattern 1: 970
# pattern 2: 970-1000
# pattern 3: 970-1000/850-900
# pattern 4: 970/1000

#pattern 3
test_list = [_ for _ in cars_df['Minimum volume of Luggage (trunk)'].str.split('-', expand=True)[2].tolist() if _ is not None] # need to get index
index_list = cars_df['Minimum volume of Luggage (trunk)'].str.split('-', expand=True)[2].index.tolist()
value_list = cars_df['Minimum volume of Luggage (trunk)'].str.split('-', expand=True)[2].tolist()
combined_list = [(i, v) for i, v in zip(index_list, value_list)]    
range2_index = [i for i, v in combined_list if v is not None]    
    
for m, i in enumerate(cars_df.loc[range2_index, 'Minimum volume of Luggage (trunk)'].str.split('/')):
    cars_df.at[range2_index[m], 'Minimum volume of Luggage (trunk)'] = (np.sum([int(t) for t in i[0].split('-')])/2 + np.sum([int(t) for t in i[1].split('-')])/2)/2

find_non_alpha(cars_df, 'Minimum volume of Luggage (trunk)')
clean_avg(dataframe=cars_df, column='Minimum volume of Luggage (trunk)', split='-') #pattern 2
clean_avg(dataframe=cars_df, column='Minimum volume of Luggage (trunk)', split='/') #pattern 4
cars_df['Minimum volume of Luggage (trunk)'] = pd.to_numeric(cars_df['Minimum volume of Luggage (trunk)'], errors='coerce')
cars_df['Minimum volume of Luggage (trunk)'].describe()
cars_df['Minimum volume of Luggage (trunk)'].isnull().sum() / cars_df.shape[0] 

#investigate Front suspension column
cars_df['Front suspension']
cars_df['Front suspension'].value_counts(dropna=False)
#since there are so many different tire categories and the data is all over the place, i am going to drop the column for right now.
cars_df.drop('Front suspension', axis=1, inplace=True)

#investigate Rear suspension
cars_df['Rear suspension']
cars_df['Rear suspension'].value_counts(dropna=False)
#since there are so many different tire categories and the data is all over the place, i am going to drop the column for right now.
cars_df.drop('Rear suspension', axis=1, inplace=True)

#investigate Acceleration 0 - 100 km/h column
cars_df['Acceleration 0 - 100 km/h']
cars_df['Acceleration 0 - 100 km/h'].isnull().sum()

find_non_alpha(cars_df, 'Acceleration 0 - 100 km/h')
cars_df.loc[cars_df['Acceleration 0 - 100 km/h'].str.find('>') != -1, 'Acceleration 0 - 100 km/h'].dropna()
cars_df.loc[cars_df['Acceleration 0 - 100 km/h'].str.find('<') != -1, 'Acceleration 0 - 100 km/h'].dropna()
cars_df.loc[cars_df['Acceleration 0 - 100 km/h'].str.find('..') != -1, 'Acceleration 0 - 100 km/h'].dropna()
cars_df.loc[cars_df['Acceleration 0 - 100 km/h'].str.find('-') != -1, 'Acceleration 0 - 100 km/h'].dropna()
cars_df.loc[cars_df['Acceleration 0 - 100 km/h'].str.find('. ') != -1, 'Acceleration 0 - 100 km/h'].dropna()
cars_df.loc[cars_df['Acceleration 0 - 100 km/h'].str.find('.') != -1, 'Acceleration 0 - 100 km/h'].dropna()

#edit by index
cars_df.at[2762, 'Acceleration 0 - 100 km/h'] = str(8.5)
cars_df.at[7482, 'Acceleration 0 - 100 km/h'] = str(11.3)

cars_df['Acceleration 0 - 100 km/h'] = cars_df['Acceleration 0 - 100 km/h'].apply(
        lambda x: get_replace(str(x), {' ': '', '>': '', '<': '', 'sec': ''})
        ) 

clean_avg(dataframe=cars_df, column='Acceleration 0 - 100 km/h', split='-')
cars_df['Acceleration 0 - 100 km/h'] = pd.to_numeric(cars_df['Acceleration 0 - 100 km/h'], errors='coerce')
cars_df['Acceleration 0 - 100 km/h'].describe()
cars_df['Acceleration 0 - 100 km/h'].isnull().sum() / cars_df.shape[0]

#investigate Max. weight column
cars_df['Max. weight']
cars_df['Max. weight'].isnull().sum()

find_non_alpha(cars_df, 'Max. weight')
cars_df.loc[cars_df['Max. weight'].str.find('-') != -1, 'Max. weight'].dropna()
cars_df.loc[cars_df['Max. weight'].str.find('<') != -1, 'Max. weight'].dropna()
cars_df.loc[cars_df['Max. weight'].str.find('..') != -1, 'Max. weight'].dropna()

cars_df['Max. weight'] = cars_df['Max. weight'].apply(lambda x:
    get_replace(str(x), {'kg.': '', ' ': ''}))

clean_avg(dataframe=cars_df, column='Max. weight', split='-')
clean_avg(dataframe=cars_df, column='Max. weight', split='‐')
clean_avg(dataframe=cars_df, column='Max. weight', split='/')
cars_df['Max. weight'] = pd.to_numeric(cars_df['Max. weight'], errors='coerce')
cars_df['Max. weight'].describe()
cars_df['Max. weight'].isnull().sum() / cars_df.shape[0]
    
#investigate Fuel consumption (economy) - urban column
cars_df['Fuel consumption (economy) - urban']
cars_df['Fuel consumption (economy) - urban'].isnull().sum()
    
find_non_alpha(cars_df, 'Fuel consumption (economy) - urban')
cars_df.loc[cars_df['Fuel consumption (economy) - urban'].str.find(' -') != -1, 'Fuel consumption (economy) - urban'].dropna()
cars_df.loc[cars_df['Fuel consumption (economy) - urban'].str.find('.-') != -1, 'Fuel consumption (economy) - urban'].dropna()
cars_df.loc[cars_df['Fuel consumption (economy) - urban'].str.find(' - ') != -1, 'Fuel consumption (economy) - urban'].dropna()
cars_df.loc[cars_df['Fuel consumption (economy) - urban'].str.find('- ') != -1, 'Fuel consumption (economy) - urban'].dropna()
cars_df.loc[cars_df['Fuel consumption (economy) - urban'].str.find('-') != -1, 'Fuel consumption (economy) - urban'].dropna()

cars_df.at[24513, 'Fuel consumption (economy) - urban'] = str((6.0+6.7)/2)
cars_df['Fuel consumption (economy) - urban'] = cars_df['Fuel consumption (economy) - urban'].apply(lambda x: 
    get_replace(str(x), {'l/100 km.': '', ' ': ''}))

clean_avg(dataframe=cars_df, column='Fuel consumption (economy) - urban', split='-')
cars_df['Fuel consumption (economy) - urban'] = pd.to_numeric(cars_df['Fuel consumption (economy) - urban'], errors='coerce')
cars_df['Fuel consumption (economy) - urban'].describe()
cars_df['Fuel consumption (economy) - urban'].isnull().sum() / cars_df.shape[0]

#investigate Fuel consumption (economy) - extra urban column
cars_df['Fuel consumption (economy) - extra urban']
cars_df['Fuel consumption (economy) - extra urban'].isnull().sum()

find_non_alpha(cars_df, 'Fuel consumption (economy) - extra urban')
cars_df.loc[cars_df['Fuel consumption (economy) - extra urban'].str.find(' -') != -1, 
                    'Fuel consumption (economy) - extra urban'].dropna()
cars_df.loc[cars_df['Fuel consumption (economy) - extra urban'].str.find('-\xa0') != -1, 
                    'Fuel consumption (economy) - extra urban'].dropna()
cars_df.loc[cars_df['Fuel consumption (economy) - extra urban'].str.find('. ') != -1, 
                    'Fuel consumption (economy) - extra urban'].dropna()
cars_df.loc[cars_df['Fuel consumption (economy) - extra urban'].str.find(' -') != -1, 
                    'Fuel consumption (economy) - extra urban'].dropna()

cars_df.at[25570, 'Fuel consumption (economy) - extra urban'] = str((5.9 + 5.8)/2)
cars_df.at[14429, 'Fuel consumption (economy) - extra urban'] = str(7.0)
cars_df.at[28868, 'Fuel consumption (economy) - extra urban'] = str(4.9)
cars_df['Fuel consumption (economy) - extra urban'] = cars_df['Fuel consumption (economy) - extra urban'].apply(lambda x: 
    get_replace(str(x), {'l/100 km.': '', ' ': ''}))

clean_avg(dataframe=cars_df, column='Fuel consumption (economy) - extra urban', split='-')
cars_df['Fuel consumption (economy) - extra urban'] = pd.to_numeric(cars_df['Fuel consumption (economy) - extra urban'], errors='coerce')
cars_df['Fuel consumption (economy) - extra urban'].describe()
cars_df['Fuel consumption (economy) - extra urban'].isnull().sum() / cars_df.shape[0]

#investigate Fuel consumption (economy) - combined column
cars_df['Fuel consumption (economy) - combined']
cars_df['Fuel consumption (economy) - combined'].isnull().sum()

find_non_alpha(cars_df, 'Fuel consumption (economy) - combined')
cars_df.loc[cars_df['Fuel consumption (economy) - combined'].str.find('..') != -1, 'Fuel consumption (economy) - combined'].dropna()
cars_df.loc[cars_df['Fuel consumption (economy) - combined'].str.find('.-') != -1, 'Fuel consumption (economy) - combined'].dropna()

cars_df.at[7667, 'Fuel consumption (economy) - combined'] = str(7.0)
cars_df.at[22099, 'Fuel consumption (economy) - combined'] = str((5.5 + 5.4)/2)
cars_df['Fuel consumption (economy) - combined'] = cars_df['Fuel consumption (economy) - combined'].apply(lambda x: 
    get_replace(str(x), {'l/100 km.': '', ' ': ''}))

clean_avg(dataframe=cars_df, column='Fuel consumption (economy) - combined', split='-')
cars_df['Fuel consumption (economy) - combined'] = pd.to_numeric(cars_df['Fuel consumption (economy) - combined'], errors='coerce')
cars_df['Fuel consumption (economy) - combined'].describe()
cars_df['Fuel consumption (economy) - combined'].isnull().sum() / cars_df.shape[0]

#investigate Wheel rims size column
cars_df['Wheel rims size']
cars_df['Wheel rims size'].isnull().sum()
cars_df['Wheel rims size'].value_counts(dropna=False)
#since there are so many different tire categories and the data is all over the place, i am going to drop the column for right now.
cars_df.drop('Wheel rims size', axis=1, inplace=True)

#investigate Valvetrain column
cars_df['Valvetrain']
cars_df['Valvetrain'].value_counts(dropna=False)
#since there are so many different tire categories and the data is all over the place, i am going to drop the column for right now.
cars_df.drop('Valvetrain', axis=1, inplace=True)

#investigate Seats column.
cars_df['Seats']
cars_df['Seats'].value_counts(dropna=False)

#identifying all the rows with Seats in the d-mmm format:
seat_list = cars_df['Seats'].unique().tolist()
date_list = [i for i in seat_list if re.search("[0-9]-[A-z][A-z][A-z]", str(i))]
change_list = [i.split('-')[1] + "-" + i.split('-')[0] for i in date_list]

month_dict = {'Jan': '1', 'Feb': '2', 'Mar': '3', 'Apr': '4',
              'May': '5', 'Jun': '6', 'Jul': '7', 'Aug': '8',
              'Sep': '9', 'Oct': '10', 'Nov': '11', 'Dec': '12',
              }

change_dict = {}
for m in date_list:
    month, day = m.split('-')[1], m.split('-')[0]
    change_dict[m] = month_dict[month]+ '-' + day
    
cars_df['Seats'] = cars_df['Seats'].map(change_dict).fillna(cars_df['Seats'])
cars_df.at[cars_df[cars_df['Seats'].str.contains('2+0', regex=False, na=False)].index[0], 'Seats'] = '2' # reclassifying the 2+0 to 2
for index in cars_df[cars_df['Seats'].str.contains('2/2+2', regex=False, na=False)].index.tolist(): # reclassifying the 2/2+2 to 2+2
    cars_df.at[index, 'Seats'] = '2+2'

#reclassifying all the + seats to the value before the +, i.e., 2+2 would classify as 2:
for index in cars_df.loc[cars_df['Seats'].str.find('+') != -1, 'Seats'].dropna().index:
    cars_df.at[index, 'Seats'] = cars_df.loc[index, 'Seats'].split('+')[0]

#reclassifying all the - seats to the value before the -, i.e., 2-3 would classify as 2:
for index in cars_df.loc[cars_df['Seats'].str.find('-') != -1, 'Seats'].dropna().index:
    cars_df.at[index, 'Seats'] = cars_df.loc[index, 'Seats'].split('-')[0]

cars_df['Seats'] = pd.to_numeric(cars_df['Seats'].apply(lambda x: str(x).strip()), errors='coerce')    
cars_df['Seats'].value_counts(dropna=False)

#investigate Doors column
cars_df['Doors'].value_counts(dropna=False)
door_list = cars_df['Doors'].unique().tolist()
date_list = [i for i in door_list if re.search("[0-9]-[A-z][A-z][A-z]", str(i))]
change_list = [i.split('-')[1] + "-" + i.split('-')[0] for i in date_list]

change_dict = {}
for m in date_list:
    month, day = m.split('-')[1], m.split('-')[0]
    change_dict[m] = month_dict[month]+ '-' + day    
cars_df['Doors'] = cars_df['Doors'].map(change_dict).fillna(cars_df['Doors'])

#reclassifying all the - doors to the value before the -, i.e., 2-3 would classify as 2:
for index in cars_df.loc[cars_df['Doors'].str.find('-') != -1, 'Doors'].dropna().index:
    cars_df.at[index, 'Doors'] = cars_df.loc[index, 'Doors'].split('-')[0]

cars_df['Doors'] = pd.to_numeric(cars_df['Doors'].apply(lambda x: str(x).strip()), errors='coerce')    
cars_df['Doors'].value_counts(dropna=False)


#investigate the Coupe type column
cars_df['Coupe type']
cars_df['Coupe type'].isnull().sum()
cars_df['Coupe type'].value_counts(dropna=False)

#reclassifying some of the Coupe type
cars_df.loc[cars_df['Coupe type'] == 'Sedan, Fastback', 'Coupe type']
cars_df.at[[30292, 30291], 'Coupe type'] = 'Sedan'
cars_df.loc[cars_df['Coupe type'] == 'Coupe, Fastback', 'Coupe type']
cars_df.at[11548, 'Coupe type'] = 'Coupe'

"""
converting the units.  i am going to convert all the units to units that i am used too:
"""
# 'Cylinder Bore', 'Piston Stroke', 'Rear (Back) track', 'Front track', 'Width', 'Wheelbase', 'Height', 'Length' -> from mm. to inches
mm_list = ['Cylinder Bore', 'Piston Stroke', 'Rear (Back) track', 'Front track', 'Width', 'Wheelbase', 'Height', 'Length']
cars_df[mm_list] = cars_df[mm_list].apply(lambda x: x * 0.039370)

# 'Fuel consumption (economy) - combined', 'Fuel consumption (economy) - extra urban', 'Fuel consumption (economy) - urban' -> l / 100km to mpg
kml_list = ['Fuel consumption (economy) - combined', 'Fuel consumption (economy) - extra urban', 'Fuel consumption (economy) - urban']
cars_df[kml_list] = cars_df[kml_list].apply(lambda x: 235.214 / x)

# 'Max. weight', 'Kerb Weight' -> kg. to pounds (lbs)
kg_list = ['Max. weight', 'Kerb Weight']
cars_df[kg_list] = cars_df[kg_list].apply(lambda x: 2.205 * x)

# 'Maximum speed' -> km/h to mph (miles per hour)
cars_df['Maximum speed'] = cars_df['Maximum speed'].apply(lambda x: x / 1.609) 
cars_df.rename({'Acceleration 0 - 100 km/h': 'Acceleration 0 - 60 m/h'}, axis=1, inplace=True)

# 'Engine displacement' -> cm3 to in3
cars_df['Engine displacement'] = cars_df['Engine displacement'].apply(lambda x: x / 16.387)

# 'Fuel tank volume' -> liters to gallons
cars_df['Fuel tank volume'] = cars_df['Fuel tank volume'].apply(lambda x: x / 3.785) 

#fill in missing values for Power hp.  uses the fact that hp is embedded in the Modidication (Engine) column for the missing values of Power hp.  
fill = cars_df.loc[cars_df['Power hp'].isnull(), ['Brand', 'Model', 'Generation', 'Modification (Engine)', 'Power hp']].index.tolist()
missing_dict = cars_df.loc[fill, 'Modification (Engine)'].apply(
        lambda x: pd.to_numeric(re.findall('[0-9]+', re.findall('\(.*?\)',str(x))[0])[0])).to_dict()
cars_df['Power hp'] = cars_df['Power hp'].index.to_series().map(missing_dict).fillna(cars_df['Power hp'])
cars_df['Power hp'].describe()

#dimension and weight columns.  
pd.plotting.scatter_matrix(cars_df.loc[:, ['Width', 'Length', 'Height', 'Max. weight', 'Kerb Weight']], figsize=(10,10));
cars_df.plot.scatter(x='Kerb Weight', y='Height');

#kerb weight has one value that does not make sense.  it says the car is 39601.8 lbs.
#https://www.auto-data.net/en/ssangyong-musso-i-2.3-d-80hp-16003 is wrong
#https://www.ultimatespecs.com/car-specs/Ssangyong/5705/Ssangyong-Musso-23-TD.html states the kerb weight as 4045 lbs
cars_df.loc[cars_df['Kerb Weight'] > 35000, ['Brand', 'Model', 'Generation', 'Modification (Engine)', 'Year of putting into production']]
cars_df.loc[29278,:]
cars_df.at[29278, 'Kerb Weight'] = 4045.0
cars_df.plot.scatter(x='Kerb Weight', y='Max. weight');
pd.plotting.scatter_matrix(cars_df.loc[:, ['Width', 'Length', 'Height', 'Max. weight', 'Kerb Weight']], figsize=(10,10));

#same situation as above
cars_df.loc[cars_df['Kerb Weight'] > 15000, ['Brand', 'Model', 'Generation', 'Modification (Engine)', 'Year of putting into production']]
cars_df.loc[25072,:]
cars_df.at[25072, 'Kerb Weight'] = 1743.86
cars_df.plot.scatter(x='Kerb Weight', y='Length');
pd.plotting.scatter_matrix(cars_df.loc[:, ['Width', 'Length', 'Height', 'Max. weight', 'Kerb Weight']], figsize=(10,10));

cars_df.to_csv("C:/Users/mcype/Documents/Full Car Project/cars_impute_df.csv")





















