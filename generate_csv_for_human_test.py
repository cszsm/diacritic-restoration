import re
import pickle

import csv


def deacc(str):
    
    
    accentized=['Á','á','É',"é",'Í','í','Ú','ú','Ü','ü','Ö','ö','Ű','ű','Ó','ó','Ő','ő']
    decentized=['A','a','E',"e",'I','i','U','u','U','u','O','o','U','u','O','o','O','o']

    for j in range(18):
        str=str.replace(accentized[j],decentized[j])
        
    return str


multis=open('humanoid.pickle','rb')
multiformwords=pickle.load(multis)


length=len(multiformwords)

print(length)

words=[]

with open('rejto.txt','r') as f:
    for line in f:
        for word in line.split():
           words.append(word.replace(',','').replace('.','')) 

print(words[100:2000])

excel=open('excel.csv','w')

fieldnames = ['-4', '-3', '-2', '-1', 'WORD', '1', '2', '3', '4']
writer = csv.DictWriter(excel, fieldnames=fieldnames)
writer.writeheader()

i=0
for szo in words:
    if deacc(szo) in multiformwords:
        writer.writerow({'-4':words[i-4], '-3':words[i-3], '-2':words[i-2], '-1':words[i-1], 'WORD':deacc(szo), '1':words[i+1], '2':words[i+2], '3':words[i+3], '4':words[i+4]})
    i+=1
