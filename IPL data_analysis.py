#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from IPython.display import Image
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly
import plotly.offline as py
import plotly.graph_objs as go
import plotly.io as pio
from collections import defaultdict
import collections
import os
import psutil
import warnings
warnings.filterwarnings('ignore')


# ## reading the datasets

# In[5]:


deli=pd.read_csv(r'deliveries.csv')
mat=pd.read_csv(r'matches.csv')


# ## data pre processing

# In[6]:


##same team
deli=deli.replace(to_replace="Rising Pune Supergiant",value="Rising Pune Supergiants")
mat=mat.replace(to_replace="Rising Pune Supergiant",value="Rising Pune Supergiants")

##delhi daredevils to delhi capitals
deli=deli.replace(to_replace="Delhi Daredevils",value="Delhi Capitals")
mat=mat.replace(to_replace="Delhi Daredevils",value="Delhi Capitals")

##deccan chargers to sunrisers Hyderabad
deli=deli.replace(to_replace="deccan chargers",value="sunrisers Hyderabad")
mat=mat.replace(to_replace="deccan chargers",value="sunrisers Hyderabad")

##same name of stadiums
mat=mat.replace(to_replace="MA Chidambaran Stadium",value="M. A.Chidambaram Stadium")
mat=mat.replace(to_replace="Punjab Cricket Association IS Bindra Stadium,Mohali",value="Punjab Cricket Association Stadium,Mohali")
mat=mat.replace(to_replace="Rajiv Gandhi International Stadium,Uppal",value="Rajiv Gandhi Intl.Cricket Stadium")
mat=mat.replace(to_replace="ACA-VDCA Stadium",value="Dr. Y.S. Rajeskhara reddy ACA-VDCA Stadium")
mat=mat.replace(to_replace="M. Chinnaswamy Stadium",value="M Chinnaswamy Stadium")

mat['winner']=mat['winner'].fillna("No Result")


# In[7]:


deli['bowling_team'].unique()


# In[8]:


deli['batting_team'].unique()


# In[9]:


Teams={'Sunrisers Hyderabad': 'SRH', 'Royal Challengers Bangalore':'RCB',
       'Mumbai Indians': 'MI', 'Rising Pune Supergiants': 'RPS', 'Gujarat Lions': 'GL',
       'Kolkata Knight Riders': 'KKR', 'Kings XI Punjab': 'KXIP', 'Delhi Capitals': 'DD',
       'Chennai Super Kings': 'CSK', 'Rajasthan Royals': 'RR', 'Deccan Chargers':'DC',
       'Kochi Tuskers Kerala': 'KTK', 'Pune Warriors': 'PW'
       }
deli['batting_team']=deli['batting_team'].map(Teams)
deli['bowling_team']=deli['bowling_team'].map(Teams)


# In[10]:


deli


# ## most successfull team

# In[12]:


#asking user whether to save the plot on the disk or not
if(input("Save/download? y/n: ").lower()[0]=='y'):
   save_files=True
   print("\nplots will be downloaded/saved.")
else:
   save_files=False
   print("\nplots will not be downloaded/saved.")    


# In[13]:


mat['winner'].value_counts()


# ## teams that have won most finals

# In[20]:


finals=mat.drop_duplicates(subset=['season'],keep='last')
finals=finals[['id','season','city','team1','team2','toss_winner','toss_decision','winner']]

most_finals=pd.concat([finals['team1'],finals['team2']]).value_counts().reset_index()
most_finals.rename({'index':'team',0:'count'},axis=1,inplace=True)
wins=finals['winner'].value_counts().reset_index()
most_finals=most_finals.merge(wins,left_on='team',right_on='index',how='outer')
most_finals=most_finals.replace(np.NaN,0)
most_finals.drop('index',axis=1,inplace=True)
most_finals.rename({'count':'finals_played','winner':'won_count'},inplace=True,axis=1)
most_finals


# In[22]:


scores=deli.groupby(['match_id','inning','batting_team','bowling_team'])['total_runs'].sum().reset_index()
scores


# In[24]:


scores_1=scores[scores['total_runs']>=200]
scores_1


# In[27]:


fig=plt.figure(figsize=(12,8))
plt.title("teams that have scores 200+",fontsize=30)
sns.countplot(x='batting_team',data=scores_1)

if save_files:
    if not os.path.exists('plots'):
     os.makedirs('plots')
    filename='plots/teams_200s'
    plt.savefig(filename,bbox_inches='tight')


# ## highest score in each season

# In[30]:


matches_list=[]
total_innings=[]
batting_team=[]
runs=[]

for match_no in deli['match_id'].unique():
    for innings in deli[deli['match_id']==match_no]['inning'].unique():
        ad=deli[
                (deli['match_id']==match_no)&
                (deli['inning']==innings)
        ]
        total_runs=ad['total_runs'].sum()
        runs.append(total_runs)
        matches_list.append(match_no)
        total_innings.append(innings)
        batting_team.append(ad['batting_team'].unique()[0])


# In[32]:


ad0=pd.DataFrame()
ad0['match_id']=matches_list
ad0['total_runs']=runs
ad0['season']=[mat[mat['id']==i]['season'].unique()[0] for i in matches_list]
ad0['batting_team']=batting_team
ad0


# In[41]:


season=[]
teamid=[]
max_runs=[]
for year in ad0['season'].unique():
    maximum_run=ad0[ad0['season']==year]['total_runs'].max()
    team=ad0[(ad0['season']==year)& (ad0['total_runs']==maximum_run)]['batting_team'].unique()[0]
    season.append(year)
    teamid.append(team)
    max_runs.append(maximum_run)
max_runs


# In[43]:


plt.style.use('ggplot')
fig=plt.gcf()
fig.set_size_inches(13,7.5)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("Highest team score in each season",fontsize=20)
ax=sns.barplot(x = season, y = max_runs, hue = teamid, dodge=False)
ax.legend(loc = 'center left', bbox_to_anchor=(1,0.7))

# save file
if save_files:
  if not os.path.exists('plots'):
    os.makedirs('plots')
  filename = 'plots/teams_200s'
  plt.savefig(filename, bbox_inches = 'tight')


# ## win percentage

# In[44]:


matches_copy = pd.DataFrame
matches_copy = mat.copy(deep=True)

matches_copy['team1']=matches_copy['team1'].map(Teams)
matches_copy['team2']=matches_copy['team2'].map(Teams)


# In[45]:


teams=(matches_copy['team1'].value_counts()+matches_copy['team2'].value_counts()).reset_index()
teams.columns=['team_name','Matches_played']
teams


# In[46]:


matches_copy['winner']=matches_copy['winner'].map(Teams)


# In[47]:


wins=matches_copy['winner'].value_counts().reset_index()
wins.columns=['team_name','wins']
wins


# In[48]:


player=teams.merge(wins,left_on='team_name',right_on='team_name',how='inner')
player['%win']=(player['wins']/player['Matches_played'])*100
player = player.sort_values('%win', ascending=False)
player


# In[49]:


trace1=go.Bar(x=player['team_name'], y=player['Matches_played'], name='Total Matches')
trace2=go.Bar(x=player['team_name'], y=player['wins'], name='Matches wins')


# In[50]:


matches_data=[trace1,trace2]
fig = py.iplot(matches_data)


# In[53]:


def team1_vs_team2(team1,team2):
    mt1=mat[((mat['team1']==team1)|(mat['team2']==team1))&((mat['team1']==team2)|(mat['team2']==team2))]
    plt.style.use('ggplot')
    fig=plt.gcf()
    fig.set_size_inches(13,7.5)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("ONE ON ONE CLASH",fontsize=20)
    ax=sns.countplot(x = 'season', hue='winner',data=mt1, palette='YlGnBu')
    ax.legend(loc = 'center left', bbox_to_anchor=(1,0.7))

    # save file
    if save_files:
      if not os.path.exists('plots'):
        os.makedirs('plots')
      filename = 'plots/CSKvsMI_EverySeason'
      plt.savefig(filename, bbox_inches = 'tight')


# In[54]:


team1_vs_team2('Chennai Super Kings','Mumbai Indians')


# In[55]:


data = matches_copy
micsk=data[np.logical_or(np.logical_and(data['team1']=='MI',data['team2']=='CSK'),np.logical_and(data['team2']=='MI',data['team1']=='CSK'))]


# In[56]:


sns.set(style='dark')
fig=plt.gcf()
fig.set_size_inches(10,8)
sns.countplot(micsk['winner'],order=micsk['winner'].value_counts().index)
plt.text(-0.1,15,str(micsk['winner'].value_counts()['MI']),size=29,color='white')
plt.text(0.9,9,str(micsk['winner'].value_counts()['CSK']),size=29,color='white')
plt.xlabel('Winner',fontsize=15)
plt.ylabel('Count',fontsize=15)
plt.yticks(fontsize=0)
plt.title('MI vs CSK - head to head')
plt.show()

# save file
if save_files:
    if not os.path.exists('plots'):
        os.makedirs('plots')
    filename = 'plots/CSKvsMI_HeadToHead'
    fig.savefig(filename, bbox_inches='tight')


# ## player analysis

# In[57]:


### Players with most Man of the Match awards in IPL
mom = pd.DataFrame()
mom['Awards']=mat['player_of_match'].value_counts()
mom['Player'] =mom.index
mom=mom[:20]
plt.style.use('ggplot')
fig=plt.gcf()
fig.set_size_inches(18.5,8.5)
plt.xticks(rotation=90,fontsize=15)
plt.yticks(fontsize=16)
plt.title("Most Man Of The Matches in IPL (All Seasons)",fontsize=20)
ax=sns.barplot(x='Player',y='Awards', data=mom)
count=0
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,height + 0.3,mom['Awards'].iloc[count],ha="center") 
    count+=1
    
# save file
if save_files:
  if not os.path.exists('plots'):
    os.makedirs('plots')
  filename = 'plots/most_MOM'
  plt.savefig(filename, bbox_inches = 'tight')


# In[58]:


dic=defaultdict(list)
for i in range(0,len(mat)):
    dic[mat.season.iloc[i]].append(mat.player_of_match.iloc[i])
player=[]
match=[]
year=[]
for i in sorted(dic.keys()):
    ctr=collections.Counter(dic[i])
    d={k: v for k, v in sorted(ctr.items(), key=lambda item: item[1],reverse=True)}
    player.append(list(d.keys())[0])
    match.append(list(d.values())[0])
    year.append(i)
plt.style.use('ggplot')
fig=plt.gcf()
fig.set_size_inches(18.5,7.5)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("Most Man Of The Match awards won by players each season", fontsize=20)
ax=sns.barplot(year, match)
count=0
for p in ax.patches:
    height = p.get_height()
    ax.text(p.get_x()+p.get_width()/2.,height + 0.15,player[count],ha="center") 
    count+=1
    
# save file
if save_files:
  if not os.path.exists('plots'):
    os.makedirs('plots')
  filename = 'plots/most_MOM_EachSeason'
  plt.savefig(filename, bbox_inches = 'tight')


# In[59]:


### TOP RUN GETTERS IN IPL HISTORY ###

top_runGetters = pd.DataFrame(deli.groupby('batsman')['batsman_runs'].sum().sort_values(ascending=False).head(10))
top_runGetters.reset_index(inplace=True)
top_runGetters.columns=['Batsman', 'Total runs']
plt.style.use('ggplot')
fig=plt.gcf()
fig.set_size_inches(18.5,7.5)

ax=sns.barplot(x='Batsman',y='Total runs', data=top_runGetters)

plt.title("Top Run Getters in IPL", fontsize=20, fontweight = 'bold')
plt.xlabel("Batsmen", size = 25)
plt.ylabel("Total Runs Scored", size=25)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# save file
if save_files:
  if not os.path.exists('plots'):
    os.makedirs('plots')
  filename = 'plots/TopRunScorers'
  plt.savefig(filename, bbox_inches = 'tight')


# In[60]:


top_runGetters.head()


# In[62]:


balls=deli.groupby('batsman')['ball'].count().reset_index()
balls


# In[63]:


runs=deli.groupby('batsman')['batsman_runs'].sum().reset_index()
runs


# In[64]:


### FOURS HIT BY BATSMEN IN IPL SO FAR ###

four=deli[deli['batsman_runs']==4]
#four
runs_4=four.groupby('batsman')['batsman_runs'].count().reset_index()
runs_4.columns=['batsman','4s']
runs_4


# In[65]:


### SIXES HIT BY BATSMEN IN IPL SO FAR ###

six=deli[deli['batsman_runs']==6]
#six

runs_6=six.groupby('batsman')['batsman_runs'].count().reset_index()
runs_6.columns=['batsman','6s']
runs_6


# In[66]:


player=pd.concat([runs,balls.iloc[:,1],runs_4.iloc[:,1],runs_6.iloc[:,1]],axis=1)
player

player.fillna(0,inplace=True)
player


# In[67]:


player['strike_rate']=(player['batsman_runs']/player['ball'])*100
player


# In[68]:


grp=deli.groupby(['match_id','batsman','batting_team'])['batsman_runs'].sum().reset_index()
grp


# In[69]:


maximum=grp.groupby('batsman')['batsman_runs'].max().reset_index()
maximum.columns=['batsman','max_runs']
maximum


# In[70]:


player2=pd.concat([player,maximum.iloc[:,1]],axis=1)
player2

player2_df = pd.DataFrame(player2)

player2.fillna(0,inplace=True)
player2


# ## highest individual score

# In[71]:


deli.groupby(['match_id','batsman','batting_team'])['batsman_runs'].sum().reset_index().sort_values(by='batsman_runs',ascending=False).head(10)


# ## bowler analysis

# In[72]:


deli['dismissal_kind'].unique()


# ## highest wickets

# In[73]:


dismissal_kinds = ['caught', 'bowled', 'lbw', 'caught and bowled', 'stumped', 'hit wicket']


# In[75]:


bowlers = deli[deli['dismissal_kind'].isin(dismissal_kinds)]
bowlers = deli.groupby('bowler').apply(lambda x: x['dismissal_kind'].dropna().reset_index(name='wickets'))
bowlers


# In[76]:


bowlers_df = bowlers.groupby('bowler').count().reset_index()
top_bowlers= bowlers_df.sort_values(by='wickets', ascending=False)
top_bowlers_head=top_bowlers[top_bowlers.wickets>=40].head(10)


# In[77]:


top_bowlers_head


# In[78]:


fig=plt.gcf()
fig.set_size_inches(15.5,5.5)
plt.xticks(fontsize=15, rotation=45)
plt.yticks(fontsize=15)
sns.barplot(top_bowlers_head['bowler'],top_bowlers['wickets'])
plt.title("Top Wicket Takers in IPL", fontsize=20, fontweight = 'bold')
plt.xlabel("Bowler", size = 25)
plt.ylabel("Wickets", size=25)

# save file
if save_files:
  if not os.path.exists('plots'):
    os.makedirs('plots')
  filename = 'plots/TopWicketTakers'
  plt.savefig(filename, bbox_inches = 'tight')


# In[80]:


batsmen = deli.groupby('batsman').apply(lambda x:np.sum(x['batsman_runs'])).reset_index(name="runs")
top_batsmen = batsmen.sort_values(by='runs', ascending=False)
top_batsmen=top_batsmen[top_batsmen.runs>=1000]


# In[81]:


top_batsmen.rename(columns = {'batsman': 'player'}, inplace = True)
top_batsmen.head()


# In[82]:


top_bowlers.rename(columns = {'bowler': 'player'}, inplace = True)


# In[89]:


#Adding some more Feature Columns to deliveries Dataset
dic = dict()
for match_id in mat['id'].unique():
    dic[match_id] = mat[mat['id']==match_id]['season'].values[0]
deli['season'] = [dic[i] for i in deli['match_id']]


# In[90]:


# Selecting only the Death Overs i.e. 16-20 overs
deli = deli[deli['over']>=16]
deli.shape


# In[91]:


mat.head(1)


# In[92]:


deli.head(1)


# ## venue analysis

# In[104]:


sta= pd.DataFrame()
sta['Count']=mat['venue'].value_counts()
sta['Venue'] =sta.index
sta=sta[:20]
plt.style.use('ggplot')
fig=plt.gcf()
fig.set_size_inches(18.5,10.5)
plt.ylabel("Venue", size=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=16)
plt.title("Matches Played in Different Stadiums",fontsize=20)
ax=sns.barplot(x='Count', y='Venue', data=sta)
count=0

for i, v in enumerate(sta['Count']):
    ax.text(v + 1.5, i, str(v),va="center", fontdict=dict(fontsize=20))

# save file
if save_files:
  if not os.path.exists('plots'):
    os.makedirs('plots')
  filename = 'plots/matches_In_Diff_Stadiums'
  plt.savefig(filename, bbox_inches = 'tight')


# In[98]:


df = pd.DataFrame()
df1 = pd.DataFrame()
chunks = []
chunks.append(mat['toss_winner'][mat['toss_decision']=='bat'])
chunks.append(mat['team2'][(mat['toss_decision']=='field') & (mat['toss_winner'] == mat['team1'])])
chunks.append(mat['team1'][(mat['toss_decision']=='field') & (mat['toss_winner'] == mat['team2'])])

df = pd.DataFrame(pd.concat(chunks))
df = df.sort_index()
df.columns = ['team']

indexes = df[(df['team'] == mat['winner'])&(df.index == mat.index)].index

df1['venue'] = mat['venue'].value_counts()
df1['win_count'] = mat[mat.index.isin(indexes)]['venue'].value_counts()
df1['win %'] = 100 * df1['win_count']/df1['venue'] 
df1 = df1.sort_values(by = ['win %'],ascending = False)
df1 = df1[:20]
winPercent = df1['win %'].astype('int')
plt.style.use('ggplot')
fig=plt.gcf()
fig.set_size_inches(18.5,10.5)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("Percent of matches won batting first in Different Stadiums(Top 20)",fontsize=19)
ax=sns.barplot(winPercent[:20], df1[:20].index)
plt.xlabel("Win % batting first", size=20)
count=0

for i, v in enumerate(winPercent):
    ax.text(v + 1.5, i, str(v),va="center", fontdict=dict(fontsize=20))

# save file
if save_files:
  if not os.path.exists('plots'):
    os.makedirs('plots')
  filename = 'plots/percent_BatFirstWin_StadiumWise'
  plt.savefig(filename, bbox_inches = 'tight')


# In[ ]:





# In[99]:


df1 = pd.DataFrame()
 
df1['venue'] = mat['venue'].value_counts()
df1['win_count'] = mat[mat.index.isin(indexes)]['venue'].value_counts()
df1['win %'] = 100 * df1['win_count']/df1['venue'] 
df1 = df1.sort_values(by = ['win %'])
df1['win % balling first'] = 100 - df1['win %']
df1 = df1[:20]
df1['win % balling first'] = df1['win % balling first'].astype('int')
plt.style.use('ggplot')
fig=plt.gcf()
fig.set_size_inches(18.5,10.5)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.title("Percent of matches won fielding first in Different Stadiums(Top 20)",fontsize=20)
ax=sns.barplot(df1['win % balling first'],df1.index)
plt.xlabel("Win % fielding first", size=20)
count=0

for i, v in enumerate(df1['win % balling first']):
    ax.text(v + 1.5, i, str(v),va="center", fontdict=dict(fontsize=20))


# save file
if save_files:
  if not os.path.exists('plots'):
    os.makedirs('plots')
  filename = 'plots/percent_FieldFirstWin_StadiumWise'
  plt.savefig(filename, bbox_inches = 'tight')


# ## toss analysis

# In[100]:


temp_series = mat.toss_decision.value_counts()
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
colors = ['gold', 'lightskyblue']
fig = plt.figure(figsize=(5,5))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Toss decision percentage")
plt.show()

# save file
if save_files:
  if not os.path.exists('plots'):
    os.makedirs('plots')
  filename = 'plots/Toss_Decision_Percent_pie'
  fig.savefig(filename, bbox_inches = 'tight')


# In[102]:


num_of_wins = (mat.win_by_wickets>0).sum()
num_of_loss = (mat.win_by_wickets==0).sum()
labels = ["Wins", "Loss"]
total = float(num_of_wins + num_of_loss)
sizes = [(num_of_wins/total)*100, (num_of_loss/total)*100]
colors = ['gold', 'lightskyblue']
fig = plt.figure(figsize=(5,5))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
plt.title("Win percentage batting second")
plt.show()

# save file
if save_files:
  if not os.path.exists('plots'):
    os.makedirs('plots')
  filename = 'plots/WinPercent_Chasing'
  fig.savefig(filename, bbox_inches = 'tight')


# In[ ]:




