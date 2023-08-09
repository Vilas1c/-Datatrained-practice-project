#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np

teams = ['Sunrisers Hyderabad',
 'Mumbai Indians',
 'Royal Challengers Bangalore',
 'Kolkata Knight Riders',
 'Kings XI Punjab',
 'Chennai Super Kings',
 'Rajasthan Royals',
 'Delhi Capitals',
 'Gujrat Titans',
 'Lucknow Super Giants']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah','Mohali', 'Bengaluru']

pickle.dump(pipe,open('pipe.pkl','wb'))
st.title('TATA IPL WIN PREDICTOR BY DREAM 11')

col1,col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('select the batting team',sorted(teams))

    with col2:
        bowling_team = st.selectbox('select the bowling team',sorted(teams))

    selected_city = st.selectbox('select host city',sorted (cities))

    target = st.numbers_input('Target')

    col3,col4,col5 = st.columns(3)

    with col3:
        score = st.number_input('score')

        with col4:
            overs = st.number_input('overs complated')

            with col5:
                wickets = st.number_input('wickets out')

    if st.button('predict probability'):
        runs_left = target - score
        balls_left = 120 - (overs*6)
        wickets = 10 - wickets
        crr = score/overs
        rrr = (runs_left*6)/balls_left
input_df = pd.dataframe({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[total_runs_x],'crr':[crr],'rrr':[rrr]})

        result = pipe.predict_proba(input_df)
        loss = result+[0][0]
        win = result+[0][1]

st.header(batting_team + "- " + str(round(win*100)) + "%")
st.header(bowling_team + "- " + str(round(loss*100)) + "%")

# In[2]:


match = pd.read_csv('matches1.csv')
delivery = pd.read_csv('delivery1.csv')


# In[3]:


match.head()


# In[4]:


match.shape


# In[5]:


delivery.head(6)


# In[6]:


delivery.groupby(['match_id','inning']).sum()['total_runs']


# In[7]:


total_score_df = delivery.groupby(['match_id','inning']).sum()['total_runs'].reset_index()


# In[8]:


total_score_df = total_score_df[total_score_df['inning'] == 1]
total_score_df


# In[9]:


match_df = match.merge(total_score_df[['match_id','total_runs']],left_on='id',right_on='match_id')
match_df


# In[10]:


match_df['team1'].unique()


# In[11]:


teams = [
    'Sunrisers Hyderabad',
    'Mumbai Indians',
    'Royal Challengers Bangalore',
    'Kolkata Knight Riders',
    'Kings XI Punjab',
    'Chennai Super Kings',
    'Rajasthan Royals',
    'Delhi Capitals',
    'Gujrat Titans',
    'Lucknow Super Giants'
]


# In[12]:


match_df['team1'] = match_df['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match_df['team2'] = match_df['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match_df['team1'] = match_df['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match_df['team2'] = match_df['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')

match_df['team1'] = match_df['team1'].str.replace('Gujrat Lions','Gujrat Titans')
match_df['team2'] = match_df['team2'].str.replace('Gujrat Lions','Gujrat Titans')

match_df['team1'] = match_df['team1'].str.replace('Pune Warriors','Lucknow Super Giants')
match_df['team2'] = match_df['team2'].str.replace('Pune Warriors','Lucknow Super Giants')


# In[13]:


match_df = match_df[match_df['team1'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]
match_df = match_df[match_df['team2'].isin(teams)]


# In[14]:


match_df.shape


# In[15]:


match_df = match_df[match_df['dl_applied'] == 0]


# In[16]:


match_df = match_df[['match_id','city','winner','total_runs']]


# In[17]:


delivery_df = match_df.merge(delivery,on='match_id')


# In[18]:


delivery_df = delivery_df[delivery_df['inning'] == 2]


# In[19]:


delivery_df.head(20)


# In[20]:


delivery_df['current_score'] = delivery_df.groupby('match_id').cumsum()['total_runs_y']


# In[21]:


delivery_df['runs_left'] = delivery_df['total_runs_x'] - delivery_df['current_score']


# In[22]:


delivery_df['balls_left'] = 120 - (delivery_df['over']*6 + delivery_df['ball'])


# In[23]:


delivery_df


# In[24]:


delivery_df['player_dismissed'] = delivery_df['player_dismissed'].fillna("0")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].apply(lambda x:x if x == "0" else "1")
delivery_df['player_dismissed'] = delivery_df['player_dismissed'].astype('int')
wickets = delivery_df.groupby('match_id').cumsum()['player_dismissed'].values
delivery_df['wickets'] = 10 - wickets
delivery_df.head()


# In[25]:


delivery_df.head()


# In[26]:


# crr = runs/overs
delivery_df['crr'] = (delivery_df['current_score']*6)/(120 - delivery_df['balls_left'])


# In[27]:


delivery_df['rrr'] = (delivery_df['runs_left']*6)/delivery_df['balls_left']


# In[28]:


def result(row):
    return 1 if row['batting_team'] == row['winner'] else 0


# In[29]:


delivery_df['result'] = delivery_df.apply(result,axis=1)


# In[30]:


final_df = delivery_df[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr','result']]


# In[31]:


final_df = final_df.sample(final_df.shape[0])


# In[32]:


final_df.sample()


# In[33]:


final_df.dropna(inplace=True)


# In[34]:


final_df = final_df[final_df['balls_left'] != 0]


# In[35]:


X = final_df.iloc[:,:-1]
y = final_df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1)


# In[36]:


X_train


# In[37]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

trf = ColumnTransformer([
    ('trf',OneHotEncoder(sparse=False,drop='first'),['batting_team','bowling_team','city'])
]
,remainder='passthrough')


# In[38]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


# In[39]:


pipe = Pipeline(steps=[
    ('step1',trf),
    ('step2',LogisticRegression(solver='liblinear'))
])


# In[40]:


pipe.fit(X_train,y_train)


# In[41]:


Pipeline(steps=[('step1',
                 ColumnTransformer(remainder='passthrough',
                                   transformers=[('trf',
                                                  OneHotEncoder(drop='first',
                                                                sparse=False),
                                                  ['batting_team',
                                                   'bowling_team', 'city'])])),
                ('step2', LogisticRegression(solver='liblinear'))])


# In[42]:


y_pred = pipe.predict(X_test)


# In[43]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[44]:


pipe.predict_proba(X_test)[10]


# In[45]:


def match_summary(row):
    print("Batting Team-" + row['batting_team'] + " | Bowling Team-" + row['bowling_team'] + " | Target- " + str(row['total_runs_x']))
    


# In[46]:


def match_progression(x_df,match_id,pipe):
    match = x_df[x_df['match_id'] == match_id]
    match = match[(match['ball'] == 6)]
    temp_df = match[['batting_team','bowling_team','city','runs_left','balls_left','wickets','total_runs_x','crr','rrr']].dropna()
    temp_df = temp_df[temp_df['balls_left'] != 0]
    result = pipe.predict_proba(temp_df)
    temp_df['lose'] = np.round(result.T[0]*100,1)
    temp_df['win'] = np.round(result.T[1]*100,1)
    temp_df['end_of_over'] = range(1,temp_df.shape[0]+1)
    
    target = temp_df['total_runs_x'].values[0]
    runs = list(temp_df['runs_left'].values)
    new_runs = runs[:]
    runs.insert(0,target)
    temp_df['runs_after_over'] = np.array(runs)[:-1] - np.array(new_runs)
    wickets = list(temp_df['wickets'].values)
    new_wickets = wickets[:]
    new_wickets.insert(0,10)
    wickets.append(0)
    w = np.array(wickets)
    nw = np.array(new_wickets)
    temp_df['wickets_in_over'] = (nw - w)[0:temp_df.shape[0]]
    
    print("Target-",target)
    temp_df = temp_df[['end_of_over','runs_after_over','wickets_in_over','lose','win']]
    return temp_df,target
    


# In[47]:


temp_df,target = match_progression(delivery_df,5,pipe)
temp_df


# In[48]:


import matplotlib.pyplot as plt
plt.figure(figsize=(18,8))
plt.plot(temp_df['end_of_over'],temp_df['wickets_in_over'],color='yellow',linewidth=3)
plt.plot(temp_df['end_of_over'],temp_df['win'],color='#00a65a',linewidth=4)
plt.plot(temp_df['end_of_over'],temp_df['lose'],color='red',linewidth=4)
plt.bar(temp_df['end_of_over'],temp_df['runs_after_over'])
plt.title('Target-' + str(target))


# In[49]:


teams


# In[50]:


delivery_df['city'].unique()


# In[2]:


import pickle
pickle.dump(pipe,open('pipe.pkl','wb'))


# In[ ]:





# In[ ]:



