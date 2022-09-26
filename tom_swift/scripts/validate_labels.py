import json
import codecs
import pandas as pd
print('loading data...\n')
import sys
import random
import pickle
sample,class2num = pickle.load(open('validate_dependencies.pkl','rb'))
# choose sample file
import os
from os import listdir

import time

username = sys.argv[1]
if username=='-f':
	username = input('Type in your username and hit Enter,please:\n')
if len(username)==0:
	username = input('You forgot to put your username in the argument for the script. \nType in your username and hit Enter,please:\n')
##################
# Instructions
instructions = '''You will be presented with a Facebook Post.
The fundamental task is to evaluate which Category it is related to, if any..
'''
# Or load from file.
#################
print('\nNow we want to label the data. using the following Instructions:\n %s\n\n Keywords should be separated by ",".'%instructions)
input('\nPress any key when you are ready.')
def load_others_done():

    others_done = set()
    for filename in other_files:
        others_done.update(load_done(filename))
    return others_done
import re
line_sep = r'[\n\r]+'
def load_done(filename):
	done = set()
	with codecs.open(filename,'r','utf-8') as f:
		s = f.read()
		l = re.split(line_sep,s)[0:-1]#.split('\n\r')[0:-1]
		for d in l:
			d = json.loads(d)
			done.add(d['id'])
	return done

fname = 'label_%s.csv'%username
other_files = [i for i in os.listdir() if 'label_' in i and i!=fname]
others_done = load_others_done()

if os.path.isfile(fname):

	done = load_done(fname)
else:
	done = set()
print('\n--------------- Congratulations you have already labeled %d samples -------------------\n'%len(done))
fwrite = codecs.open(fname,'a','utf-8')
import os
import pickle
import time
import pandas as pd
import json
# define label_discussion

instructions = '''You will be presented with a Facebook Post.
The fundamental task is to evaluate which Category it is related to, if any..
'''
class2english = {'kunst_musik_museum_kulturpolitik': 'Culture', ' klima_miljø_naturen ': 'Environment',
                 'undervisning_folkeskole_ungdomsuddannelse': 'Education', 'udenrigspolitik_forsvaret_militær': 'Foreign affairs',
                 ' EU': 'EU', 'sundhed': 'Health',
                 'indvandre_flygtninge_udlændinge': 'Immigration', 'beskæftigelse_arbejdsløshed': 'Employment', 'skat': 'Taxes',
                 'forskning_universitet_videregående uddannelser': 'Science',
                 ' religion_værdipolitik': 'Religion', ' økonomi_vækst': 'Economy',
                 'kriminalitet': 'Crime', 'hjemmehjælp_ ældre_ pension_plejehjem': 'Elder care',
                 'fattig_socialt udsatte_socialministeriet_svageste': 'Social policy'}


import time
import random
def label(row):
    if random.random()<0.5:
        cat = row[-1]
        true = input("\n\tIs this related to %s category? press y else press Enter\n"%class2num[cat])
    else:
        cat = random.choice(range(len(class2num)))
        true = input("\n\tIs this related to %s category? press y else press Enter\n"%class2num[cat])
    d = {'t':time.time(),'category':cat,'true':true,'true_cat':row[-1]}
    return d
def label_complex(row):
	if random.random()<0.5:
		cat = row[-1]
		true = input("\n\tIs this related to %s category? press y else press Enter\n"%class2num[cat])
	else:
		cat = random.choice(range(len(class2num)))
		true = input("\n\tIs this related to %s category? press y else press Enter\n"%class2num[cat])
	d = {'t':time.time(),'category':cat,'true':true,'true_cat':row[-1]}
	question = 'Is this related to the following classes:\n'+'\n'.join(['(%d) %s'%(num,i) for num,i in enumerate(class2num)])+'\n\tInput as Comma-separated list, if none just press enter.\n'
	categories = input(question)
	d.update({'t2':time.time(),'classes':categories.split(',')})
	return d
#print('questione
#'''Is this related to the following classes?
#				   (0) Culture\n(1) Environment\n(2) Education\n(3) Foreign affairs\n(4) EU\n(5) Health\n(6) Immigration\n(7) Employment\n(8) Taxes\n(9) Science\n(10) Religion\n(11) Economy\n(12) Crime\n(13) Elder care\n(14) Social policy
#				   Input as Comma-separated list if none just press Enter'''ady to Annotate....')

while True:
	if random.random()<0.1:
		pid = random.choice(list(others_done))
		row = sample[sample['id']==pid].values[0]
	else:
		row = sample.sample(1).values[0]
	pid = row[1]
	if pid in done:
	    continue
	print('\n\n\n\t||||||||Post||||||||\n\t: %s'%row[-2])
	if len(done)>10*len(class2num):
		data = label_complex(row)
		while True:
			inp = input('\nDo you wish to edit the response? press y else just Enter\n').strip()
			if inp=='y':
				data = label_complex(row)
			else:
				break
	else:
		data = label(row)
		while True:
			inp = input('\nDo you wish to edit the response? press y else just Enter\n').strip()
			if inp=='y':
				data = label(row)
			else:
				break

	data['id'] = pid
	fwrite.write(json.dumps(data)+'\n\r')
	fwrite.flush()
	done.add(pid)
	if len(done)%50==0:
		print('\n--------------- Congratulations you have already labeled %d samples -------------------\n'%len(done))
