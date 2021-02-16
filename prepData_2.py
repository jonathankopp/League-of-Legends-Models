import pandas as pd
import json
import numpy as np

df      = pd.read_csv('preSweep.csv',index_col='date',parse_dates = True)
prepTeamDat = []
prepPlayerDat = []
rpm = []
players = dict()
playerC = dict()
sPlayer = dict()
tPlayer = dict()
print(df.shape)
# print(df.columns,len(df.columns))

# test = pd.DataFrame.from_dict(data=leg,orient='index',columns=['nip','size'])

# create player data array, and then in 'Real Time' use those to create rows for the model

# print(df.head(12))
gameAmntAdd = 0

counter  = 1
team     = 0
exp      = 0
gameAmnt = 6
realCount = 1
teamR   = np.array([0 for i in range(17)])
teamB   = np.array([0 for i in range(17)])
temp   = {
    'top' : [],
    'jng' : [],
    'mid' : [],
    'bot' : [],
    'sup' : []
}
skip = False
sCount = 0

# print(list(df[df['position']=='team']['opp_dragons']).count(-99),len(list(df[df['position']=='team']['dragons'])))
# print(list(set(list(df[df['position']!='team']['csat10']))))
# change to a type of moving avg later on for emphasis on new data
for i,row in df.iterrows():
    if(skip and sCount < 2):
        if(row['position'] != 'team'):
            print('fuck')
            print(counter)
            break
        realCount += 1
        sCount+=1
        continue
    elif(skip and sCount >= 2):
        skip = False
        sCount = 0
    if(row['position']=='team'):
        print('heredafuq')
        break
    player = row['player']
    playI  = counter%10 if counter%10 !=0 else 10
    if player not in players.keys():
        tPlayer[player] = np.array([0 for i in range(17)])
        players[player] = np.array([0 for i in range(39)])
        sPlayer[player] = np.array([0 for i in range(22)])
        playerC[player] = 1
    if(counter%5 != 0):
        if(team == 0):
            teamR = np.add(teamR,tPlayer[player])/2.0
        else:
            teamB = np.add(teamB,tPlayer[player])/2.0
        temp[row['position']] = np.append(temp[row['position']], sPlayer[player])

    elif(counter%5 == 0):
        if(team == 0):
            teamR = np.add(teamR,tPlayer[player])/2.0
        else:
            teamB = np.add(teamB,tPlayer[player])/2.0
        temp[row['position']] = np.append(temp[row['position']], sPlayer[player])

        if(counter%10 == 0):
            # every player has >= gameAmnt games]
            if(exp >= 8):
                side = 'Red' if row['side'] == 0 else 'Blue'
                if(row['result']==1):
                    prepTeamDat.append(np.append([gameAmntAdd],np.append(np.concatenate((teamR, teamB), axis=None),[side])))
                    prepPlayerDat.append({
                        gameAmntAdd : {
                            'top': np.append(temp['top'],side),
                            'jng': np.append(temp['jng'],side),
                            'mid': np.append(temp['mid'],side),
                            'bot': np.append(temp['bot'],side),
                            'sup': np.append(temp['sup'],side)
                        },
                    })
                    rpm.append(np.append(gameAmntAdd,np.append('top', np.append(temp['top'],side))))
                    rpm.append(np.append(gameAmntAdd,np.append('jng', np.append(temp['jng'],side))))
                    rpm.append(np.append(gameAmntAdd,np.append('mid', np.append(temp['mid'],side))))
                    rpm.append(np.append(gameAmntAdd,np.append('bot', np.append(temp['bot'],side))))
                    rpm.append(np.append(gameAmntAdd,np.append('sup', np.append(temp['sup'],side))))

                else:
                    side = 'Red' if row['side'] == 1 else 'Blue'
                    prepTeamDat.append(np.append([gameAmntAdd],np.append(np.concatenate((teamR, teamB), axis=None),[side])))
                    prepPlayerDat.append({
                        gameAmntAdd : {
                            'top': np.append(temp['top'],side),
                            'jng': np.append(temp['jng'],side),
                            'mid': np.append(temp['mid'],side),
                            'bot': np.append(temp['bot'],side),
                            'sup': np.append(temp['sup'],side)
                        },
                    })
                    rpm.append(np.append(gameAmntAdd,np.append('top', np.append(temp['top'],side))))
                    rpm.append(np.append(gameAmntAdd,np.append('jng', np.append(temp['jng'],side))))
                    rpm.append(np.append(gameAmntAdd,np.append('mid', np.append(temp['mid'],side))))
                    rpm.append(np.append(gameAmntAdd,np.append('bot', np.append(temp['bot'],side))))
                    rpm.append(np.append(gameAmntAdd,np.append('sup', np.append(temp['sup'],side))))
                gameAmntAdd += 1  
                # print(len(prepTeamDat[0][0]))
                # break
            exp = 0
            skip=True
            temp   = {
                'top' : [],
                'jng' : [],
                'mid' : [],
                'bot' : [],
                'sup' : []
            }
            
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
    teamDataList1 = list(teamData[7:17])
    teamDataList2 = list(teamData[37:])
    comp = list(row[5:7])+teamDataList1+list(row[17:37])+teamDataList2
    
    fullTeam  = teamDataList1 + teamDataList2
    # add add then average it
    tPlayer[player] = np.add(np.array(fullTeam),tPlayer[player])/2.0
    players[player] = np.add(np.array(comp),players[player])/2.0
    sPlayer[player] = np.add(np.array(list(row[5:7])+list(row[17:37])),sPlayer[player])/2.0
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
ret = pd.DataFrame(data=prepTeamDat,columns= ['gamenum','position'] + retColsR[2:12] + retColsR[32:] + retColsB[2:12] + retColsB[33:] + ['res'])


# print(ret)
print(prepPlayerDat[0],'\n',len(prepPlayerDat))
print(ret.shape)
with open("objectiveData.csv","w",newline='\n') as f:
    ret.to_csv(f,index=False,sep=',',header=True)




for i in list(df.columns)[5:]:
    retColsR.append('R_'+i)
    retColsB.append('B_'+i)
retCols = retColsR + retColsB
retPM = pd.DataFrame(data=rpm,columns= ['gamenum','position'] + retColsR[0:2] + retColsR[12:32] + retColsB[0:2] + retColsB[12:32] + ['res'])

with open("playerMatchUps.csv","w",newline='\n') as f:
    retPM.to_csv(f,index=False,sep=',',header=True)