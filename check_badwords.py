import os
import re
import math

from collections import defaultdict
from nltk.corpus import stopwords
import pandas as pd
from pprint import pprint
import ipdb

dir_path = os.path.dirname(os.path.realpath(__file__))


############################################################################################################################
## import badwords list
############################################################################################################################

'''
## remove 'mother'
long_badwords = ['*damn', '*dyke', '*fuck*', '*shit*', 'Assface', 'Biatch', 'Carpet', 'Cock*', 'CockSucker', 'Ekrem*', 'Felcher', 'Flikker', 'Fotze', 'Fudge', 'Fukah', 'Fuken', 'Fukin', 'God-damned', 'Huevon', 'Kurac', 'Lesbian', 'Lezzian', 'Lipshits', 'Lipshitz', 'Motha', 'Mutha', 'Phukker', 'Poonani', 'Shitty', 'Shity', 'Shyte', 'Skanky', 'Slutty', 'a-hole', 'ahole', 'amcik', 'andskota', 'arschloch', 'arse*', 'ash0le', 'asholes', 'ass-hole', 'assh0le', 'asshole', 'assholz', 'assrammer', 'asswipe', 'azzhole', 'b!+ch', 'b!tch', 'b*tch', 'b00b*', 'b00bs', 'b17ch', 'b1tch', 'bassterds', 'bastard', 'basterds', 'basterdz', 'bi+ch', 'bi7ch', 'bitch', 'blowjob', 'boffing', 'boiolas', 'bollock*', 'boobs', 'breasts', 'buceta', 'butt-pirate', 'butthole', 'buttwipe', 'cabron', 'cazzo', 'chink', 'chraa', 'clits', 'cock-head', 'cock-sucker', 'cockhead', 'cocks', 'daygo', 'dick*', 'dild0', 'dildo', 'dilld0', 'dirsa', 'dominatricks', 'dominatrics', 'dominatrix', 'dziwka', 'ejackulate', 'ejakulate', 'enculer', 'enema', 'fag1t', 'faget', 'fagg1t', 'faggit', 'faggot', 'fanculo', 'fanny', 'fatass', 'feces', 'ficken', 'fitt*', 'flipping', 'foreskin', 'fucker', 'fuckin', 'fucking', 'futkretzn', 'fux0r', 'gayboy', 'gaygirl', 'gayness', 'guiena', 'h4x0r', 'helvete', 'honkey', 'hoorem', 'hoorm', 'idi0t', 'injun', 'jackoff', 'jerk-off', 'jisim', 'kanker*', 'klootzak', 'knobs', 'knulle', 'kraut', 'kuksuger', 'kurwa', 'kusi*', 'kyrpa*', 'l3i+ch', 'l3itch', 'lesbian', 'lesbo', 'mamhoon', 'masochist', 'masokist', 'massterbait', 'masstrbait', 'masstrbate', 'masterbaiter', 'masterbat*', 'masterbat3', 'masterbate', 'masturbat*', 'masturbate', 'merd*', 'mibun', 'monkleigh', 'mother-fucker', 'motherfucker', 'mouliewop', 'mulkku', 'muschi', 'nastt', 'nepesaurio', 'nigga', 'nigger', 'nigur', 'niiger', 'niigr', 'nutsack', 'orafis', 'orgasim', 'orgasm', 'orgasum', 'oriface', 'orifice', 'orifiss', 'orospu', 'packi', 'packy', 'paska*', 'pecker', 'peeenus', 'peenus', 'peinus', 'pen1s', 'penas', 'penis', 'penis-breath', 'penus', 'penuus', 'perse', 'phuck', 'picka', 'pierdol*', 'pillu*', 'pimmel', 'pimpis', 'piss*', 'pizda', 'polac', 'polak', 'poontsee', 'pr1ck', 'preteen', 'pusse', 'pussee', 'pussy', 'puuke', 'qahbeh', 'queef*', 'queer', 'qweers', 'qweerz', 'qweir', 'rautenberg', 'recktum', 'rectum', 'retard', 's.o.b', 'sadist', 'scank', 'schaffer', 'scheiss*', 'schlampe', 'schlong', 'schmuck', 'screw', 'screwing', 'scrotum', 'semen', 'sh1tter', 'sharmuta', 'sharmute', 'shemale', 'shipal', 'shitter', 'skanck', 'skank', 'skankee', 'skribz', 'skurwysyn', 'son-of-a-bitch', 'sphencter', 'spierdalaj', 'splooge', 'teets', 'testical', 'testicle', 'titt*', 'va1jina', 'vag1na', 'vagiina', 'vagina', 'vaj1na', 'vajina', 'vittu', 'vullva', 'vulva', 'w00se', 'wetback*', 'wh00r.', 'wh0re', 'whoar', 'whore', 'wichser', 'xrated', 'zabourah']
## remove 'Blow'
short_4_badwords = ['Clit', 'Ekto', 'Fu(*', 'Fukk', 'Phuc', 'Phuk', 'Sh!t', 'Shyt', 'anus', 'ayir', 'c0ck', 'cawk', 'chuj', 'cipa', 'clit', 'cnts', 'cntz', 'cock', 'crap', 'cunt', 'd!ck', 'd*ck', 'd4mn', 'dego', 'dick', 'dupa', 'dyke', 'f.ck', 'faen', 'fagz', 'faig', 'fart', 'fcuk', 'g00k', 'gook', 'h00r', 'h0ar', 'hell', 'hoar', 'hoer', 'hore', 'jism', 'jiss', 'jizm', 'jizz', 'kawk', 'kike', 'knob', 'kunt', 'mofo', 'muie', 'n1gr', 'nazi', 'p0rn', 'paki', 'paky', 'poop', 'porn', 'pr0n', 'pr1c', 'pr1k', 'pula', 'pule', 'puta', 'puto', 'sh!+', 'sh!t', 'sh#t', 'sh1t', 'shi+', 'shit', 'shiz', 'slut', 'smut', 'spic', 'suka', 'teez', 'tits', 'turd', 'twat', 'wank', 'wop*']
short_badwords = ['8ss', '@$$', '@ss', 'a$$', ' ass ', ' azz ', 'c0k ', ' cum ', ' dik ', ' fag ', 'fck', ' feg ', ' fuc ', ' fuk ', 'h0r', ' hui ', ' jap ', ' kuk ', ' sux ', ' tit ', 'w0p', 'wtf', ' yed']
new_words_list = ['duckish', 'asscrack','bullshitted']

EP_PROFANITIES = [' ass ', ' asses ', 'asshole', 'assholes', 'buttfuck', 'cocksucker', 'cocksucking', 'cunt', 'cuntlicker', 'cunts', 'cyberfuc', 'cyberfuck', 'cyberfucked', 'cyberfucker', 'cyberfuckers', 'cyberfucking', 'dick', ' fag ', 'fagging', 'faggot', 'faggs', 'fagot', 'fagots', 'fags', 'fingerfuck', 'fingerfucked', 'fingerfucker', 'fingerfuckers', 'fingerfucking', 'fingerfucks', 'fisting', 'fistfuck', 'fistfucked', 'fistfucker', 'fistfuckers', 'fistfucking', 'fistfuckings', 'fistfucks', 'footfuck', 'fuck', 'fucked', 'fucker', 'fuckers', 'fuckfest', 'fuckin', 'fucking', 'fuckings', 'fuckme', 'f u c k', 'fu ck', 'fuc k', 'f uck', 'fucks', 'fucktard', 'fucktards', 'f\\*uck', 'fu\\*ck', 'fuc\\*k', 'fuk', 'fuks', 'gangbang', 'gangbanged', 'gangbangs', 'gaysex', 'gay sex', 'hores', 'incest', 'jackoff', 'jack-off', 'jerk-off', 'kunt', 'mothafuck', 'mothafucka', 'mothafuckas', 'mothafuckaz', 'mothafucked', 'mothafucker', 'mothafuckers', 'mothafuckin', 'mothafucking', 'mothafuckings', 'mothafucks', 'motherfuck', 'motherfucked', 'motherfucker', 'motherfuckers', 'motherfuckin', 'motherfucking', 'motherfuckings', 'motherfucks', 'nigger', 'niggers', 'pecker', 'pedofile', 'pedofilia', 'pedophile', 'pedophilia', 'p e n i s', 'phuk', 'phuked', 'phuking', 'phukked', 'phukking', 'phuks', 'phuq', 'rimjob', 'rimming', 'semen', 'shagged', 'shit', 'shited', 'shitfull', 'shiting', 'shitings', 'shits', 'shitted', 'shitter', 'shitters', 'shitting', 'shittings', 'shitty', 'skank', 'skanks', 'slut', 'sluts', 'spermed', 'sperming', 'titfuck', 'titfucker', 'titjob', 'titties', 'titty', 'twat', 'wank', 'wanker', 'whore', 'whores']
'''

## combine all badwords:
badwords_total = [' ass ', ' asses ', ' azz ', ' cum ', ' dik ', ' fag ', ' feg ', ' fuc ', ' fuk ', ' hui ', ' jap ', ' kuk ', ' sux ', ' tit ', ' yed', '*damn', '*dyke', '*fuck*', '*shit*', '8ss', '@$$', '@ss', 'Assface', 'Biatch', 'Carpet', 'Clit', 'Cock*', 'CockSucker', 'Ekrem*', 'Ekto', 'Felcher', 'Flikker', 'Fotze', 'Fu(*', 'Fudge', 'Fukah', 'Fuken', 'Fukin', 'Fukk', 'God-damned', 'Huevon', 'Kurac', 'Lesbian', 'Lezzian', 'Lipshits', 'Lipshitz', 'Motha', 'Mutha', 'Phuc', 'Phuk', 'Phukker', 'Poonani', 'Sh!t', 'Shitty', 'Shity', 'Shyt', 'Shyte', 'Skanky', 'Slutty', 'a$$', 'a-hole', 'ahole', 'amcik', 'andskota', 'anus', 'arschloch', 'arse*', 'ash0le', 'asholes', 'ass-hole', 'asscrack', 'assh0le', 'asshole', 'assholes', 'assholz', 'assrammer', 'asswipe', 'ayir', 'azzhole', 'b!+ch', 'b!tch', 'b*tch', 'b00b*', 'b00bs', 'b17ch', 'b1tch', 'bassterds', 'bastard', 'basterds', 'basterdz', 'bi+ch', 'bi7ch', 'bitch', 'blowjob', 'boffing', 'boiolas', 'bollock*', 'boobs', 'breasts', 'buceta', 'bullshitted', 'butt-pirate', 'buttfuck', 'butthole', 'buttwipe', 'c0ck', 'c0k ', 'cabron', 'cawk', 'cazzo', 'chink', 'chraa', 'chuj', 'cipa', 'clit', 'clits', 'cnts', 'cntz', 'cock', 'cock-head', 'cock-sucker', 'cockhead', 'cocks', 'cocksucker', 'cocksucking', 'crap', 'cunt', 'cuntlicker', 'cunts', 'cyberfuc', 'cyberfuck', 'cyberfucked', 'cyberfucker', 'cyberfuckers', 'cyberfucking', 'd!ck', 'd*ck', 'd4mn', 'daygo', 'dego', 'dick', 'dick*', 'dild0', 'dildo', 'dilld0', 'dirsa', 'dominatricks', 'dominatrics', 'dominatrix', 'duckish', 'dupa', 'dyke', 'dziwka', 'ejackulate', 'ejakulate', 'enculer', 'enema', 'f u c k', 'f uck', 'f.ck', 'f\\*uck', 'faen', 'fag1t', 'faget', 'fagg1t', 'fagging', 'faggit', 'faggot', 'faggs', 'fagot', 'fagots', 'fags', 'fagz', 'faig', 'fanculo', 'fanny', 'fart', 'fatass', 'fck', 'fcuk', 'feces', 'ficken', 'fingerfuck', 'fingerfucked', 'fingerfucker', 'fingerfuckers', 'fingerfucking', 'fingerfucks', 'fistfuck', 'fistfucked', 'fistfucker', 'fistfuckers', 'fistfucking', 'fistfuckings', 'fistfucks', 'fisting', 'fitt*', 'flipping', 'footfuck', 'foreskin', 'fu ck', 'fu\\*ck', 'fuc k', 'fuc\\*k', 'fuck', 'fucked', 'fucker', 'fuckers', 'fuckfest', 'fuckin', 'fucking', 'fuckings', 'fuckme', 'fucks', 'fucktard', 'fucktards', 'fuk', 'fuks', 'futkretzn', 'fux0r', 'g00k', 'gangbang', 'gangbanged', 'gangbangs', 'gay sex', 'gayboy', 'gaygirl', 'gayness', 'gaysex', 'gook', 'guiena', 'h00r', 'h0ar', 'h0r', 'h4x0r', 'hell', 'helvete', 'hoar', 'hoer', 'honkey', 'hoorem', 'hoorm', 'hore', 'hores', 'idi0t', 'incest', 'injun', 'jack-off', 'jackoff', 'jerk-off', 'jisim', 'jism', 'jiss', 'jizm', 'jizz', 'kanker*', 'kawk', 'kike', 'klootzak', 'knob', 'knobs', 'knulle', 'kraut', 'kuksuger', 'kunt', 'kurwa', 'kusi*', 'kyrpa*', 'l3i+ch', 'l3itch', 'lesbian', 'lesbo', 'mamhoon', 'masochist', 'masokist', 'massterbait', 'masstrbait', 'masstrbate', 'masterbaiter', 'masterbat*', 'masterbat3', 'masterbate', 'masturbat*', 'masturbate', 'merd*', 'mibun', 'mofo', 'monkleigh', 'mothafuck', 'mothafucka', 'mothafuckas', 'mothafuckaz', 'mothafucked', 'mothafucker', 'mothafuckers', 'mothafuckin', 'mothafucking', 'mothafuckings', 'mothafucks', 'mother-fucker', 'motherfuck', 'motherfucked', 'motherfucker', 'motherfuckers', 'motherfuckin', 'motherfucking', 'motherfuckings', 'motherfucks', 'mouliewop', 'muie', 'mulkku', 'muschi', 'n1gr', 'nastt', 'nazi', 'nepesaurio', 'nigga', 'nigger', 'niggers', 'nigur', 'niiger', 'niigr', 'nutsack', 'orafis', 'orgasim', 'orgasm', 'orgasum', 'oriface', 'orifice', 'orifiss', 'orospu', 'p e n i s', 'p0rn', 'packi', 'packy', 'paki', 'paky', 'paska*', 'pecker', 'pedofile', 'pedofilia', 'pedophile', 'pedophilia', 'peeenus', 'peenus', 'peinus', 'pen1s', 'penas', 'penis', 'penis-breath', 'penus', 'penuus', 'perse', 'phuck', 'phuk', 'phuked', 'phuking', 'phukked', 'phukking', 'phuks', 'phuq', 'picka', 'pierdol*', 'pillu*', 'pimmel', 'pimpis', 'piss*', 'pizda', 'polac', 'polak', 'poontsee', 'poop', 'porn', 'pr0n', 'pr1c', 'pr1ck', 'pr1k', 'preteen', 'pula', 'pule', 'pusse', 'pussee', 'pussy', 'puta', 'puto', 'puuke', 'qahbeh', 'queef*', 'queer', 'qweers', 'qweerz', 'qweir', 'rautenberg', 'recktum', 'rectum', 'retard', 'rimjob', 'rimming', 's.o.b', 'sadist', 'scank', 'schaffer', 'scheiss*', 'schlampe', 'schlong', 'schmuck', 'screw', 'screwing', 'scrotum', 'semen', 'sh!+', 'sh!t', 'sh#t', 'sh1t', 'sh1tter', 'shagged', 'sharmuta', 'sharmute', 'shemale', 'shi+', 'shipal', 'shit', 'shited', 'shitfull', 'shiting', 'shitings', 'shits', 'shitted', 'shitter', 'shitters', 'shitting', 'shittings', 'shitty', 'shiz', 'skanck', 'skank', 'skankee', 'skanks', 'skribz', 'skurwysyn', 'slut', 'sluts', 'smut', 'son-of-a-bitch', 'spermed', 'sperming', 'sphencter', 'spic', 'spierdalaj', 'splooge', 'suka', 'teets', 'teez', 'testical', 'testicle', 'titfuck', 'titfucker', 'titjob', 'tits', 'titt*', 'titties', 'titty', 'turd', 'twat', 'va1jina', 'vag1na', 'vagiina', 'vagina', 'vaj1na', 'vajina', 'vittu', 'vullva', 'vulva', 'w00se', 'w0p', 'wank', 'wanker', 'wetback*', 'wh00r.', 'wh0re', 'whoar', 'whore', 'whores', 'wichser', 'wop*', 'wtf', 'xrated', 'zabourah']

############################################################################################################################
## remove special characters, numbers
############################################################################################################################
special_character_removal = re.compile(r'[^a-z\d\!\?\- ]', re.IGNORECASE)  # Regex to remove all Non-Alpha Numeric and space
special_character_removal_all = re.compile(r'[^a-z\d\- ]', re.IGNORECASE)  
replace_numbers = re.compile(r'\d+', re.IGNORECASE)    # regex to replace all numerics
stops = set(stopwords.words("english"))

def remove_stopwords(text):
	text = text.split()
	stops = set(stopwords.words("english"))
	text = [w for w in text if not w in stops]
	text = " ".join(text)
	return text

def stem_words(text):
	text = text.split()
	stemmer = SnowballStemmer('english')
	stemmed_words = [stemmer.stem(word) for word in text]
	text = " ".join(stemmed_words)
	return text


def text_to_wordlist(text, all_char=False, lower=False, remove_stopwords=False, stem_words=False):
	if lower:
		text=text.lower()
	if remove_stopwords:
		text = remove_stopwords(text)
	original_text = text
	try:
		if all_char:
			text = special_character_removal_all.sub('', text)
		else:
			text = special_character_removal.sub('', text)
			text = text.replace('!', ' !')
			text = text.replace('?', ' ?')
			# text = text.replace('.', ' .')
			# text = text.replace(',', ' ,')
		text = text.replace('-', ' ')
		text = replace_numbers.sub('0', text)
	except Exception:
		return original_text
	# if str(text) != str(original_text):
	#     ipdb.set_trace()
	if stem_words:
		text = stem_words(text)
	return(text)


############################################################################################################################
## check whether badwords appears in the given text
############################################################################################################################

def check_badwords(text):
	badwords_list = []
	badwords_num = 0
	for w in short_badwords+short_4_badwords+long_badwords+new_words_list:
		if w.lower() in text.lower():
			badwords_num+=1
			badwords_list.append(w)
	return badwords_num, badwords_list


############################################################################################################################
## import data and apply analysis
############################################################################################################################

pred_df_09844_1and0_diff_sorted = pd.read_csv('pred_df_09844_1and0_diff_sorted.csv', index_col=0)
## count badwords, special characters in text
df2 = pred_df_09844_1and0_diff_sorted['comment_text'].apply(check_badwords)
pred_df_09844_1and0_diff_sorted['badwords_num'] = df2.apply(lambda x : x[0])
pred_df_09844_1and0_diff_sorted['badwords_list'] = df2.apply(lambda x : x[1])
pred_df_09844_1and0_diff_sorted['question_mark'] = pred_df_09844_1and0_diff_sorted['comment_text'].apply(lambda txt : txt.count('?'))
pred_df_09844_1and0_diff_sorted['exclaimation_mark'] = pred_df_09844_1and0_diff_sorted['comment_text'].apply(lambda txt : txt.count('!'))

## count unique words in one text
pred_df_09844_1and0_diff_sorted['total_words'] = pred_df_09844_1and0_diff_sorted['comment_text'].apply(lambda txt : len([w for w in text_to_wordlist(txt,all_char=True).split(' ') if w != '']))
pred_df_09844_1and0_diff_sorted['unique_words'] = pred_df_09844_1and0_diff_sorted['comment_text'].apply(lambda txt : len(set([w for w in text_to_wordlist(txt,all_char=True).split(' ') if w != ''])))
pred_df_09844_1and0_diff_sorted['unique_words_percentage'] = pred_df_09844_1and0_diff_sorted['unique_words']/pred_df_09844_1and0_diff_sorted['total_words']

## count how many Capital words in text
pred_df_09844_1and0_diff_sorted['upper_char_num'] = pred_df_09844_1and0_diff_sorted['comment_text'].apply(lambda txt : sum(1 for c in txt if c.isupper()))
pred_df_09844_1and0_diff_sorted['lower_char_num'] = pred_df_09844_1and0_diff_sorted['comment_text'].apply(lambda txt : sum(1 for c in txt if c.islower()))
pred_df_09844_1and0_diff_sorted['capital_char_percentage'] = pred_df_09844_1and0_diff_sorted['upper_char_num']/(pred_df_09844_1and0_diff_sorted['upper_char_num']+pred_df_09844_1and0_diff_sorted['lower_char_num'])


##############################################################
## do the same analysis for train_df to get base rate
train_df = pd.read_csv('train.csv',index_col=['id'])
## count badwords, special characters in text
df = train_df['comment_text'].apply(check_badwords)
train_df['badwords_num'] = df.apply(lambda x : x[0])
train_df['badwords_list'] = df.apply(lambda x : x[1])
train_df['question_mark'] = train_df['comment_text'].apply(lambda txt : txt.count('?'))
train_df['exclaimation_mark'] = train_df['comment_text'].apply(lambda txt : txt.count('!'))

## count unique words in one text
train_df['total_words'] = train_df['comment_text'].apply(lambda txt : len([w for w in text_to_wordlist(txt,all_char=True).split(' ') if w != '']))
train_df['unique_words'] = train_df['comment_text'].apply(lambda txt : len(set([w for w in text_to_wordlist(txt,all_char=True).split(' ') if w != ''])))
train_df['unique_words_percentage'] = train_df['unique_words']/train_df['total_words']

## count how many Capital words in text
train_df['upper_char_num'] = train_df['comment_text'].apply(lambda txt : sum(1 for c in txt if c.isupper()))
train_df['lower_char_num'] = train_df['comment_text'].apply(lambda txt : sum(1 for c in txt if c.islower()))
train_df['capital_char_percentage'] = train_df['upper_char_num']/(train_df['upper_char_num']+train_df['lower_char_num'])
ipdb.set_trace()

# ##############################################################
# name = 'train'
# train_df = pd.read_csv('{}.csv'.format(name))

# ## generate check badwords columns for dataset
# train_df = train_df.dropna(axis=0, how='any')
# train_df_comment = train_df
# for w in short_badwords+short_4_badwords+long_badwords:
# 	train_df_comment[w] = train_df_comment['comment_text'].apply(lambda txt: 1 if w.lower() in txt.lower() else 0)
# 	# ipdb.set_trace()

# ## add sum row
# train_df = train_df.append(train_df.sum(numeric_only=True), ignore_index=True)
# train_df.to_csv('{}_check_badwords.csv'.format(name))

# ## save to summary dataframe
# total= pd.DataFrame(train_df.ix[train_df.shape[0]-1,:])
# total = total.sort(['summary'],axis=0,ascending=False)
# total.to_csv('{}_check_badwords_summary.csv'.format(name))
# ##############################################################



# ## calculate correlation
# train_df_pure = train_df[train_df.columns[2:]]
# train_df_pure=train_df_pure.drop(labels=226997,axis=0)
# cc = train_df_pure.corr()
# cc.to_csv('{}_summary_corr.csv'.format(name))

# cc_dic = defaultdict()
# cc = pd.read_csv(os.path.join(dir_path,'analysis','{}_summary_corr.csv'.format(name)),index_col=0)
# ipdb.set_trace()
# for i in cc.columns: 
# 	for j in cc.columns: 
# 		if not math.isnan(cc.ix[i,j]):
# 			if (j,i) not in cc_dic.keys():
# 				cc_dic[(i,j)] = cc.ix[i,j]

ipdb.set_trace()