def deaccentize(text):
    #list all the hungarian accentized characters, and their no-accent version
    accentized=['Á','á','É',"é",'Í','í','Ú','ú','Ü','ü','Ö','ö','Ű','ű','Ó','ó','Ő','ő']
    deaccentized=['A','a','E',"e",'I','i','U','u','U','u','O','o','U','u','O','o','O','o']
    for j in range(18):
        text=text.replace(accentized[j],deaccentized[j])
    return text

def deaccentize_list(original_list,demonstrate=False):
    deacc_list=[]
    for i in range(len(lista)):
        deacc_list.append(deaccentize(lista[i]))
        
    if demonstrate:
        print(original_list[0:20])
        print(deacc_list[0:20])
        
    return deacc_list