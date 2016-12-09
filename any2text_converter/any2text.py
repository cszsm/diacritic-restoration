import sys
import os.path
import os
import codecs
import re
from bs4 import BeautifulSoup
from pyth.plugins.rtf15.reader import Rtf15Reader
from pyth.plugins.xhtml.writer import XHTMLWriter


def rtf2html(pathfilename):
    print 'extracting:' + pathfilename
    #x=open(, "r",encoding='iso-8859-2')
    try:
        x=open(pathfilename, 'r')
        doc = Rtf15Reader.read(x)
        outhtml=XHTMLWriter.write(doc, pretty=True).read()
        return outhtml
    except:
        print "------------ERROR---------------------------"
        return ""   


def html2txt(html_doc):
    soup = BeautifulSoup(html_doc, 'html.parser')
    return soup.get_text()


def extract_rtfs_from_folder(root_dir):
    log=""
    i=0;
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if ".rtf" in file:
                log+=html2txt(rtf2html(os.path.join(subdir, file)))
            if len(log)>100000000:
                text_file = codecs.open("~/textract/All_from_rtf_"+str(i)+".txt", "w","utf8")
                text_file.write(log)
                i+=1
                text_file.close()
                log=""
    text_file = codecs.open("~/textract/All_from_rtf_"+str(i)+".txt", "w","utf8")
    text_file.write(log)
    i+=1
    text_file.close()

def extract_htmls_from_folder(root_dir):
    log=""
    i=0;
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if ".htm" in file:
                log+=html2txt(open(os.path.join(subdir, file),'r'))
            if len(log)>100000000:
                text_file = codecs.open("All_from_html_"+str(i)+".txt", "w","utf8")
                text_file.write(log)
                i+=1
                text_file.close()
                log=""
    text_file = codecs.open("All_from_html_"+str(i)+".txt", "w","utf8")
    text_file.write(log)
    i+=1
    text_file.close()

def remove_csv_from_html_extract(filename):
    text=codecs.open(filename,'r',"utf8").read()

    text=re.sub(r'.*{[^{^}]*}','',text)
    text_file = codecs.open("html_ext_0.txt", "w","utf8")
    text_file.write(text)
    text_file.close()
def collect_text_from_subfolders(root_dir):
    log=""
    i=0;
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if ".txt" in file:
                log+=codecs.open(os.path.join(subdir, file),'r',"iso-8859-2").read()
            if len(log)>100000000:
                text_file = codecs.open("All_txt_from_big_"+str(i)+".txt", "w","utf8")
                text_file.write(log)
                i+=1
                text_file.close()
                log=""
    text_file = codecs.open("All_txt_from_big_"+str(i)+".txt", "w","utf8",errors='ignore')
    text_file.write(log)
    i+=1
    text_file.close()
    
def filter_binary_from_text(root_dir):
    log=""
    i=0;
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if ".txt" in file:
                log+=re.sub(r'[0-9a-f]{20,}','',codecs.open(os.path.join(subdir, file),'r',"utf8").read())
            if len(log)>100000000:
                text_file = codecs.open("textract_"+str(i)+".txt", "w","utf8")
                text_file.write(log)
                i+=1
                text_file.close()
                log=""
    text_file = codecs.open("textract_"+str(i)+".txt", "w","utf8",errors='ignore')
    text_file.write(log)
    i+=1
    text_file.close()    
    
def filter_css_from_text(root_dir):
    log=""
    i=0;
    
    for subdir, dirs, files in os.walk(root_dir):
        
        for file in files:
            if ".txt" in file:
                log+=re.sub(r'.*{[^{^}]*}','',codecs.open(os.path.join(subdir, file),'r',"utf8").read())
            if len(log)>100000000:
                
                text_file = codecs.open("textract_"+str(i)+".txt", "w","utf8")
                text_file.write(log)
                
                i+=1
                text_file.close()
                log=""
    text_file = codecs.open("textract_"+str(i)+".txt", "w","utf8",errors='ignore')
    text_file.write(log)
    i+=1
    text_file.close()