import re
import pickle

size=10000
out_file_name='webcorpus_above_5_occur.pickle'
lista=[]
orig=[]


# In[ ]:

with open('web2.2-alfa-sorted.txt',encoding='iso-8859-2') as f:
    lines = f.readlines()

print("files opened")
# In[ ]:

for line in lines:
    x=re.split(r'\t+',line)[0]
    lista.append(x)
    orig.append(x)
print("words selected")

# In[ ]:




# In[ ]:

deacc_list=lista
i=0
accentized=['Á','á','É',"é",'Í','í','Ú','ú','Ü','ü','Ö','ö','Ű','ű','Ó','ó','Ő','ő']
decentized=['A','a','E',"e",'I','i','U','u','U','u','O','o','U','u','O','o','O','o']


proc=0
for item in lista:
    for j in range(18):
        deacc_list[i]=deacc_list[i].replace(accentized[j],decentized[j])
    i+=1

print("deaccentized")

# In[ ]:

print(lista[10511000:10512000])
print(len(deacc_list))


# In[ ]:

multiples=[]
matched=[]
proc=0
for i in range(len(deacc_list)):
    for j in range(i+1,i+10):
        if (j<len(deacc_list)) and deacc_list[i]==deacc_list[j] and (not i in matched) and (not j in matched) and orig[i]!=orig[j] and (orig[i]!=deacc_list[i] or orig[j]!=deacc_list[i]):
            multiples.append(deacc_list[i])
            matched.append(i)
            #print(orig[i]," = ", orig[j])
            matched.append(j)
    if i%420000==0:
        print(proc,"% done")
        proc+=2


# In[ ]:

print(len(multiples))
with open('multis.pickle','wb') as f:
    pickle.dump(multiples,f)


# In[ ]:

print(multiples[101000:102000])
print(len(orig),len(deacc_list))




