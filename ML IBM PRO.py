#!/usr/bin/env python
# coding: utf-8

# In[4]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
df=pd.read_csv('IPL Matches 2008-2020.csv')
df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df['date'] = pd.to_datetime(df.date)
df['year'] = pd.DatetimeIndex(df.date).year
df['month'] = pd.DatetimeIndex(df.date).month
df['day'] = pd.DatetimeIndex(df.date).day
df['weekday'] = pd.DatetimeIndex(df.date).weekday


# In[9]:


df.isnull().sum()


# In[10]:


df[df['city'].isnull()]


# In[11]:


df['city'].fillna('Dubai',inplace=True)
df['city'].isnull().sum()


# In[12]:


df[df['player_of_match'].isnull()]


# In[13]:


df['venue'].replace({'M.Chinnaswamy Stadium':'M Chinnaswamy Stadium'}, inplace=True)
df.replace({'Rising Pune Supergiant':'Rising Pune Supergiants'}, inplace=True)


# In[14]:


df.drop([241,486,511,744], inplace=True)
df.isnull().sum()


# In[16]:


df[df['result_margin'].isnull()]


# In[17]:


df['result_margin'].fillna(0, inplace=True)


# In[18]:


df.method.unique()


# In[19]:


df['method'].fillna('regular', inplace=True)
df.isnull().sum()


# In[20]:


df1=pd.read_csv('ml project.csv')
df1.head()


# In[21]:


df1.shape


# In[22]:


df1.info()


# In[23]:


df1.replace({'Rising Pune Supergiant':'Rising Pune Supergiants'}, inplace=True)


# In[24]:


df1.isnull().sum()


# In[25]:


len(df1[df1.is_wicket==0])


# In[26]:


len(df1[df1.extra_runs==0])


# In[27]:


df1['dismissal_kind'].fillna('NA',inplace=True)
df1['player_dismissed'].fillna('NA',inplace=True)
df1['fielder'].fillna('NA', inplace=True)
df1['extras_type'].fillna('NA', inplace=True)


# In[28]:


df2 = df.merge(df1, on="id")
df2.head()


# In[29]:


len(df1)-len(df2)


# In[30]:


len(df2['id'].unique())==len(df1['id'].unique())


# In[31]:


df2['bowling_team'].isnull().sum()


# In[32]:


df2.isnull().sum()


# In[33]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[35]:


df.describe()


# In[36]:


df1.describe()


# In[37]:


df.describe(include=['O'])


# In[38]:


df1.describe(include=['O'])


# In[39]:


played=list(df['team1'])+list(df['team2'])
sns.histplot(played)
plt.xticks(rotation=90)
plt.xlabel('Teams');


# In[40]:


sns.histplot(df['winner'], color='orange')
plt.xticks(rotation=90);


# In[41]:


sns.histplot(played)
sns.histplot(df['winner'], color='orange')
plt.xticks(rotation=90)
plt.legend(['Matches Played','Matches Won'])
plt.xlabel('Teams');


# In[42]:


player_of_match=df.player_of_match.value_counts().head(10)
player_of_match


# In[43]:


sns.barplot(x=player_of_match.index, y=player_of_match)
plt.xticks(rotation=60)
plt.ylabel('No. of Times Player of the Match')
plt.xlabel('Name Of the Player')
plt.title('Player of the Match Award', fontsize=20);


# In[44]:


df[df.neutral_venue==1]['venue'].unique()


# In[45]:


plt.figure(figsize=(15,6))
city = df.city.value_counts()
sns.barplot(x=city.index, y=city)
plt.xticks(rotation=90);


# In[46]:


for i in df.year.unique():
    print("Year :", i,'<', '='*50,'>')
    xdf=df[df.year==i] 
    plt.pie(xdf.winner.value_counts(),autopct="%1.1f%%" ) 
    plt.legend(xdf.winner.value_counts().index,bbox_to_anchor=(2,1),loc='upper right', title='Teams in {}'.format(i))
    plt.show();


# In[47]:


xd=df1.groupby(['id', 'over','inning'], as_index=False)['total_runs'].sum()
xd['over']=xd['over']+1
sns.boxplot(y=xd['total_runs'], x=xd['over'])
plt.title('Boxplot of Run Scored per Over')
plt.ylabel('Runs per Over')
plt.xlabel('Over');


# In[48]:


minimum=xd.groupby('over', as_index=False)['total_runs'].min()
maximum=xd.groupby('over', as_index=False)['total_runs'].max()
mean=xd.groupby('over', as_index=False)['total_runs'].mean()
sns.lineplot(data=maximum, x='over', y='total_runs')
sns.lineplot(data=mean, x='over', y='total_runs')
sns.lineplot(data=minimum, x='over', y='total_runs')
plt.legend(['Maximum', 'Mean', 'Minimum'])
plt.title('Lineplot of Runs Scored per Over')
plt.xlabel('Over')
plt.ylabel('Runs per Over');


# In[49]:


for i in df.winner.unique():
    xdf=df[df.winner==i] 
    plt.pie(xdf.venue.value_counts().head(5),autopct="%1.1f%%" ) 
    plt.legend(xdf.venue.value_counts().head(5).index, bbox_to_anchor=(1,1.5),loc='upper right', title=i)
    plt.show()
    print("*"*100);


# In[50]:


homeground={}
for i in df.winner.unique():
    xdf=df[df.winner==i]
    homeground[i]=xdf.venue.value_counts().head(1).index[0]
teams_played={}
for i in homeground:
    xd=df[df.venue==homeground[i]]
    teams_played[i]=len(xd)
teams_won={}
for i in homeground:
    xd=df.loc[(df['venue']==homeground[i]) & (df['winner']==i)]
    teams_won[i]=len(xd)
win_percentage={}
for i in homeground:
    win_percentage[i]=(teams_won[i]/teams_played[i])*100
win_percentage


# In[51]:


y=list(win_percentage.values())
x=list(win_percentage.keys())
sns.barplot(x=x,y=y)
plt.xticks(rotation=90);


# In[52]:


xd=df2.groupby(['batsman'], as_index=False)['batsman_runs'].sum().sort_values('batsman_runs', ascending=False).head(10)
xd


# In[53]:


sns.barplot(x=xd['batsman_runs'], y=xd['batsman'])
plt.title('Highest Run Scorers in All Season', fontsize=20)
plt.xlabel('Run Scored', fontsize=15 )
plt.ylabel('Batsman', fontsize=15);


# In[54]:


top_scorers={}
for i in list(df.year.unique()): 
    xd=df2[df2.year==i] 
    sd=xd.groupby(['batsman'], as_index=False)['batsman_runs'].sum().sort_values('batsman_runs', ascending=False).head(1)
    top_scorers[(sd['batsman'].iloc[0]+str(' ')+str(i))]=sd['batsman_runs'].iloc[0]


# In[55]:


top_scorers


# In[56]:


x=list(top_scorers.keys())
y=list(top_scorers.values())
sns.barplot(x=x, y=y )
plt.xticks(rotation=90)
plt.title('Highest Run Scorers in All Season', fontsize=20)
plt.ylabel('Run Scored', fontsize=15 )
plt.xlabel('Batsman', fontsize=15);


# In[57]:


wd=df2[(df2.is_wicket==1)&((df2.dismissal_kind=='caught')|(df2.dismissal_kind=='bowled')|(df2.dismissal_kind=='lbw')|(df2.dismissal_kind=='caught and bowled'))]
wdc=wd['bowler'].value_counts().head(10)
wdc


# In[58]:


sns.barplot(x=wdc.index, y=wdc)
plt.title('Highest Wicket Takers in All Seasons')
plt.xlabel('Name of the Bowler')
plt.ylabel('No. of Wickets')
plt.xticks(rotation=60);


# In[59]:


top_bowler={}
for i in list(wd.year.unique()):
    cwd=wd[wd.year==i]
    cwd=cwd.groupby('bowler', as_index=False)['is_wicket'].sum().sort_values('is_wicket', ascending=False).head(1)
    top_bowler[(cwd['bowler'].iloc[0]+str(' ')+str(i))]=cwd['is_wicket'].iloc[0]
top_bowler


# In[60]:


x=list(top_bowler.keys())
y=list(top_bowler.values())
sns.barplot(x=x, y=y)
plt.title('Highes Wicket take In Each Season')
plt.xlabel('Name of the Bowler')
plt.ylabel('No. of Wickets')
plt.xticks(rotation=90);


# In[61]:


fd=df1[(df1.dismissal_kind=='caught')]['fielder'].value_counts().head(10)
fd


# In[62]:


sns.barplot(x=fd.index, y=fd)
plt.title('Top 10 Fielders')
plt.xlabel('Name of Fielders')
plt.ylabel('No. of Catches / Run Outs')
plt.xticks(rotation=70);


# In[63]:


a=len(df[df.winner==df.toss_winner])
b=len(df)
print('Chances that toss winner will win the match :',(a/b)*100, '%')
plt.pie([a,b-a],autopct="%1.1f%%")
plt.legend(['won','lost']);


# In[64]:


xdf=df[df.toss_decision=='field']
a=len(xdf[xdf.winner==xdf.toss_winner])
b=len(xdf)
print('Chances that toss winner will win the match :',(a/b)*100, '%')
plt.pie([a,b-a],autopct="%1.1f%%")
plt.legend(['won','lost']);


# In[65]:


xdf=df[df.toss_decision=='bat']
a=len(xdf[xdf.winner==xdf.toss_winner])
b=len(xdf)
print('Chances that toss winner will win the match :',(a/b)*100, '%')
plt.pie([a,b-a],autopct="%1.1f%%")
plt.legend(['won','lost']);


# In[74]:


def team_comparison(team1, team2):
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    xd=df[((df.team1==team1)&(df.team2==team2))|((df.team1==team2)&(df.team2==team1))]
    win=xd.winner.value_counts()
    print('Total Clashes :', len(xd))
    print('*'*50)
    print('Wins by Each Team :')
    print(win)
    print('*'*50)
    print('Winning Chances (in %)')
    print((win/len(xd))*100)
    print('*'*50)
    cd=df2[((df2.team1==team1)&(df2.team2==team2))|((df2.team1==team2)&(df2.team2==team1))]
    cd=cd.groupby(['id','inning'], as_index=False)['total_runs'].sum()
    maxi=cd['total_runs'].max()
    maxi_run=cd[cd.total_runs==maxi]
    high_scorer=df2[(df2.id==maxi_run['id'].iloc[0]) &(df2.inning==maxi_run['inning'].iloc[0])]['batting_team'].iloc[0]
    mini=cd['total_runs'].min()
    mini_run=cd[cd.total_runs==mini]
    low_scorer=df2[(df2.id==mini_run['id'].iloc[0]) &(df2.inning==mini_run['inning'].iloc[0])]['batting_team'].iloc[0]
    print(high_scorer,'has the highest score of', maxi,'runs in the clash.')
    print(low_scorer, 'has the lowest score of', mini,'runs in the clash.')
    print('*'*50)
    xdf=df2[(df2.batting_team==team1) & (df2.bowling_team==team2)]
    bat_run_4=len(xdf[xdf.batsman_runs==4])
    bat_run_6=len(xdf[xdf.batsman_runs==6])
    total_runs=xdf['batsman_runs'].sum()
    wicket=len(xdf[xdf.is_wicket==1])
    bf=xdf.groupby(['batsman'], as_index=False)['batsman_runs'].sum().sort_values('batsman_runs', ascending=False).head(1)
    run_bf=bf['batsman_runs'].iloc[0]
    bf=bf['batsman'].iloc[0]
    bwl=xdf.groupby('bowler', as_index=False)['is_wicket'].sum().sort_values('is_wicket', ascending=False).head(1)
    bwl_wick=bwl['is_wicket'].iloc[0]
    bwl=bwl['bowler'].iloc[0]
    print(team1, 'has scored', total_runs, 'runs with',bat_run_4,'4s and',bat_run_6 ,'6s against', team2)
    print('While the top scorer is',bf, 'with',run_bf,'runs.')
    print(team2, 'has taken', wicket,'wickets against',team1)
    print('While the top wicket taker is',bwl,'with',bwl_wick, 'wickets.')
    print('*'*50)
    axes[0,0].set_title('Scores of teams')
    axes[0,0].bar(height=total_runs,x=team1)
    axes[0,1].set_title('Wickets of teams')
    axes[0,1].bar(height=wicket,x=team2)
    axes[0,2].set_title('Highest Run Scorer')
    axes[0,2].bar(height=run_bf,x=bf)
    axes[1,0].set_title('Highes Wicket Taker')
    axes[1,0].bar(height=bwl_wick,x=bwl)
    axes[1,1].set_title('Number of 4s')
    axes[1,1].bar(height=bat_run_4,x=team1)
    axes[1,2].set_title('Number of 6s')
    axes[1,2].bar(height=bat_run_6,x=team1)
    xdf=df2[(df2.batting_team==team2) & (df2.bowling_team==team1)]
    bat_run_4=len(xdf[xdf.batsman_runs==4])
    bat_run_6=len(xdf[xdf.batsman_runs==6])
    total_runs=xdf['batsman_runs'].sum()
    wicket=len(xdf[xdf.is_wicket==1])
    bf=xdf.groupby(['batsman'], as_index=False)['batsman_runs'].sum().sort_values('batsman_runs', ascending=False).head(1)
    run_bf=bf['batsman_runs'].iloc[0]
    bf=bf['batsman'].iloc[0]
    bwl=xdf.groupby('bowler', as_index=False)['is_wicket'].sum().sort_values('is_wicket', ascending=False).head(1)
    bwl_wick=bwl['is_wicket'].iloc[0]
    bwl=bwl['bowler'].iloc[0]
    print(team2, 'has scored', total_runs, 'runs with', bat_run_4,'4s and',bat_run_6 ,'6s against', team1)
    print('While the top scorer is',bf, 'with',run_bf,'runs.')
    print(team1, 'has taken', wicket,'wickets against',team2)
    print('While the top wicket taker is',bwl,'with',bwl_wick, 'wickets.')
    axes[0,0].set_title('Scores of teams')
    axes[0,0].bar(height=total_runs,x=team2)
    axes[0,1].set_title('Wickets of teams')
    axes[0,1].bar(height=wicket,x=team1)
    axes[0,2].set_title('Highest Run Scorer')
    axes[0,2].bar(height=run_bf,x=bf)
    axes[1,0].set_title('Highes Wicket Taker')
    axes[1,0].bar(height=bwl_wick,x=bwl)
    axes[1,1].set_title('Number of 4s')
    axes[1,1].bar(height=bat_run_4,x=team2)
    axes[1,2].set_title('Number of 6s')
    axes[1,2].bar(height=bat_run_6,x=team2)
    plt.tight_layout(pad=2);


# In[73]:


def stats_info(team):
    team_df=df[(df.team1==team)|(df.team2==team)]
    mom_df=pd.DataFrame(team_df.describe(include=['O']))
    print(mom_df.player_of_match.iloc[2],'has won most ({}) player of the match awards for {}.'.format(mom_df.player_of_match.iloc[3], team))
    print('*'*100)
    L=['Kolkata Knight Riders', 'Chennai Super Kings', 'Delhi Daredevils',
       'Royal Challengers Bangalore', 'Rajasthan Royals',
       'Kings XI Punjab', 'Deccan Chargers', 'Mumbai Indians',
       'Pune Warriors', 'Kochi Tuskers Kerala', 'Sunrisers Hyderabad',
       'Rising Pune Supergiants', 'Gujarat Lions', 'Delhi Capitals']
    L.remove(team)
    wins_dict={}
    for i in L:
        c2_df=team_df[(team_df.team1==i)|(team_df.team2==i)]
        team_won=len(c2_df[c2_df.winner==team])
        team_played=len(c2_df)
        try:
            win_ratio=team_won/team_played
        except Exception as e:
            win_ratio=0
        wins_dict[i]=(win_ratio*100)
    sns.barplot(x=list(wins_dict.keys()),y=list(wins_dict.values()))
    plt.title('Winning Percentage of {}'.format(team))
    plt.xlabel('Teams')
    plt.ylabel('Winning Percentage')
    plt.xticks(rotation=90)
    plt.show()
    print('*'*100)
    team_df2=df2[(df2.team1==team)|(df2.team2==team)]
    xd=team_df2.groupby(['batsman'], as_index=False)['batsman_runs'].sum().sort_values('batsman_runs', ascending=False).head(10)
    sns.barplot(data=xd, x='batsman',y='batsman_runs')
    plt.title('Top Run Scorers for {}'.format(team))
    plt.xlabel('Batsman')
    plt.ylabel('Runs Scored')
    plt.xticks(rotation=90)
    plt.show()
    print('*'*100)
    wd=team_df2[(team_df2.is_wicket==1)&((team_df2.dismissal_kind=='caught')|(team_df2.dismissal_kind=='bowled')|(team_df2.dismissal_kind=='lbw')|(team_df2.dismissal_kind=='caught and bowled'))]
    wdc=wd['bowler'].value_counts().head(10)
    sns.barplot(x=wdc.index,y=wdc)
    plt.title('Most Wicket Takers for {}'.format(team))
    plt.xlabel('Bolwer')
    plt.ylabel('No. of Wickets Taken')
    plt.xticks(rotation=90)
    plt.show()


# In[72]:


stats_info('Chennai Super Kings')


# In[69]:


stats_info('Royal Challengers Bangalore')


# In[70]:


stats_info('Mumbai Indians')


# In[ ]:





# In[ ]:





# In[ ]:




