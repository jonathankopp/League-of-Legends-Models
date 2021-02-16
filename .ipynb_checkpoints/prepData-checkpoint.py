import pandas as pd
import json
import numpy as np

df      = pd.read_csv('preSweep.csv',index_col='date',parse_dates = True)
prepDat = []
players = dict()
playerC = dict()
print(df.shape)
print(df.columns[37])

# test = pd.DataFrame.from_dict(data=leg,orient='index',columns=['nip','size'])

# create player data array, and then in 'Real Time' use those to create rows for the model

print(df.head(12))

counter  = 1
team     = 0
exp      = 0
gameAmnt = 6
realCount = 1
teamR   = np.array([0 for i in range(39)])
teamB   = np.array([0 for i in range(39)])
# 12
skip = False

sCount = 0

# print(list(df[df['position']=='team']['opp_dragons']).count(-99),len(list(df[df['position']=='team']['dragons'])))
# print(list(set(list(df[df['position']!='team']['csat10']))))
# change to a type of moving avg later on for emphasis on new data
for i,row in df.iterrows():
    if(skip and sCount < 2):
        if(row['position'] != 'team'):
            print(realCount,'fucker')
            break
        realCount += 1
        sCount+=1
        continue
    elif(skip and sCount >= 2):
        skip = False
        sCount = 0
    if(row['position']=='team'):
        print(realCount,'bitch')
    player = row['player']
    playI  = counter%10 if counter%10 !=0 else 10
    if player not in players.keys():
        players[player] = np.array([0 for i in range(39)])
        playerC[player] = 1
    if(counter%5 != 0):
        if(team == 0):
            teamR = np.add(teamR,players[player])/2.0
        else:
            teamB = np.add(teamB,players[player])/2.0
    elif(counter%5 == 0):
        if(team == 0):
            teamR = np.add(teamR,players[player])/2.0
        else:
            teamB = np.add(teamB,players[player])/2.0
        if(counter%10 == 0):
            # every player has >= gameAmnt games]
            if(exp >= 8):
                side = 'Red' if row['side'] == 0 else 'Blue'
                if(row['result']==1):
                    prepDat.append(np.append(np.concatenate((teamR, teamB), axis=None),[side]))
                else:
                    side = 'Red' if row['side'] == 1 else 'Blue'
                    prepDat.append(np.append(np.concatenate((teamR, teamB), axis=None),[side]))
            exp = 0
            skip=True

    # logic for if the amount of games played is over 10(or whatever we decide) create the rows
    if(playerC[player] >= gameAmnt):
        exp += 1
    # logic for adding the current data to the playerdict
    location = 0
    if(team==0):
        location=10
    else:
        location=11
    location -= playI

    teamData = df.iloc[realCount+(location)]
    if(teamData['position']!='team'):
        print(realCount,'bfuckitch',row['player'])
        break
    # print(teamData['position'],teamData['team'],playI,realCount+(location),team)
    # print(df.columns[7:17],df.columns[37:])

    teamDataList1 = list(teamData[7:17])
    teamDataList2 = list(teamData[37:])
    comp = list(row[5:7])+teamDataList1+list(row[17:37])+teamDataList2
    
    
    # add add then average it
    players[player] = np.add(np.array(comp),players[player])/2.0
    playerC[player] += 1
    if(counter%5==0):
        team = 1 if team == 0 else 0
    counter += 1
    realCount +=1

# print(players['Rascal'])



# print(df.head(10))
# print(prepDat[0])
retColsR = []
retColsB = []

for i in list(df.columns)[5:]:
    retColsR.append('R_'+i)
    retColsB.append('B_'+i)
retCols = retColsR + retColsB
ret = pd.DataFrame(data=prepDat,columns= retCols + ['res'])
print(ret.shape)
with open("preproData.csv","w",newline='\n') as f:
    ret.to_csv(f,index=False,sep=',',header=True)