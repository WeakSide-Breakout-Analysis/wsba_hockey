
## GLOBAL VARIABLES ##
seasons = [
    '20072008',
    '20082009',
    '20092010',
    '20102011',
    '20112012',
    '20122013',
    '20132014',
    '20142015',
    '20152016',
    '20162017',
    '20172018',
    '20182019',
    '20192020',
    '20202021',
    '20212022',
    '20222023',
    '20232024',
    '20242025'
]

convert_seasons = {'2007': '20072008', 
                   '2008': '20082009', 
                   '2009': '20092010', 
                   '2010': '20102011', 
                   '2011': '20112012', 
                   '2012': '20122013', 
                   '2013': '20132014', 
                   '2014': '20142015', 
                   '2015': '20152016', 
                   '2016': '20162017', 
                   '2017': '20172018', 
                   '2018': '20182019', 
                   '2019': '20192020', 
                   '2020': '20202021', 
                   '2021': '20212022', 
                   '2022': '20222023', 
                   '2023': '20232024', 
                   '2024': '20242025'}

convert_team_abbr = {'L.A':'LAK',
                     'N.J':'NJD',
                     'S.J':'SJS',
                     'T.B':'TBL',
                     'PHX':'ARI'}

per_sixty = ['Fi','xGi','Gi','A1','A2','P1','P','OZF','NZF','DZF','FF','FA','xGF','xGA','GF','GA','CF','CA','HF','HA','Give','Take','Penl','Penl2','Penl5','Draw','Block']

events = ['faceoff','hit','giveaway','takeaway','blocked-shot','missed-shot','shot-on-goal','goal','penalty']

#Some games in the API are specifically known to cause errors in scraping.
#This list is updated as frequently as necessary
known_probs ={
    '2007020011':'Missing shifts data for game between Chicago and Minnesota.',
    '2007021178':'Game between the Bruins and Sabres is missing data after the second period, for some reason.',
    '2008020259':'HTML data is completely missing for this game.',
    '2008020409':'HTML data is completely missing for this game.',
    '2008021077':'HTML data is completely missing for this game.',
    '2009020081':'HTML pbp for this game between Pittsburgh and Carolina is missing all but the period start and first faceoff events, for some reason.',
    '2009020658':'Missing shifts data for game between New York Islanders and Dallas.',
    '2009020885':'Missing shifts data for game between Sharks and Blue Jackets.',
    '2010020124':'Game between Capitals and Hurricanes is sporadically missing player on-ice data',
    '2012020018':'HTML events contain mislabeled events.',
    '2013020971':'On March 10th, 2014, Stars forward Rich Peverley suffered from a cardiac episode midgame and as a result, the remainder of the game was postponed.  \nThe game resumed on April 9th, and the only goal scorer in the game, Blue Jackets forward Nathan Horton, did not appear in the resumed game due to injury.  Interestingly, Horton would never play in the NHL again.',
    '2018021133':'Game between Lightning and Capitals has incorrectly labeled event teams (i.e. WSH TAKEAWAY - #71 CIRELLI (Cirelli is a Tampa Bay skater in this game)).',
    '2019020876':'Due to the frightening collapse of Blues defensemen Jay Bouwmeester, a game on February 2nd, 2020 between the Ducks and Blues was postponed.  \nWhen the game resumed, Ducks defensemen Hampus Lindholm, who assisted on a goal in the inital game, did not play in the resumed match.'
}

shot_types = ['wrist','deflected','tip-in','slap','backhand','snap','wrap-around','poke','bat','cradle','between-legs']

new = 2024

standings_end = {
    '20072008':'04-06',
    '20082009':'04-12',
    '20092010':'04-11',
    '20102011':'04-10',
    '20112012':'04-07',
    '20122013':'04-28',
    '20132014':'04-13',
    '20142015':'04-11',
    '20152016':'04-10',
    '20162017':'04-09',
    '20172018':'04-08',
    '20182019':'04-06',
    '20192020':'03-11',
    '20202021':'05-19',
    '20212022':'04-01',
    '20222023':'04-14',
    '20232024':'04-18',
    '20242025':'04-17'
}

fenwick_events = ['missed-shot','shot-on-goal','goal']
