import requests
from bs4 import BeautifulSoup
import sqlite3
import csv
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from math import log10
import numpy as np
from sklearn.metrics import silhouette_samples, silhouette_score
import skfuzzy as fuzz


conn = sqlite3.connect('articles.sqlite')
conn.execute('''CREATE TABLE if not exists ARTICLES
                (TITLE         TEXT     NOT NULL,
                 ISI         TEXT     NOT NULL);''')
conn.commit()

src = "https://www.malasngoding.com/"
n = 1
while n <= 2:
    print("page ",n)
    page = requests.get(src)
    soup = BeautifulSoup(page.content, 'html.parser')
                    
    linkhead = soup.findAll(class_='text-dark')
    pagination = soup.find(class_='next page-numbers')
            
    for links in linkhead:
        try :
            src = links['href']
            page = requests.get(src)
            soup = BeautifulSoup(page.content, 'html.parser')

            konten = soup.find('article')
            title = konten.find(class_='post-title entry-title pb-2').getText()

            temp = konten.findAll('p')
            isi = []
            for j in range(len(temp)):
                isi += [temp[j].getText()]

            isif = ""
            for i in isi:
                isif += i
            conn.execute("INSERT INTO ARTICLES (TITLE, ISI) VALUES (?, ?)", (title, isif));

        except AttributeError:
            continue
    conn.commit()
    src = pagination['href']
        
    n+=1
#function write csv
def write_csv(nama_file, isi, tipe='w'):
    'tipe=w; write; tipe=a; append;'
    with open(nama_file, mode=tipe) as tbl:
        tbl_writer = csv.writer(tbl, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in isi:
            tbl_writer.writerow(row)

#preproccessing - stopword(menghilangkan imbuhan)
cursor = conn.execute("SELECT* from ARTICLES")
isif =''
for row in cursor:
    isif+=row[1]
    ##print(row)

factory = StopWordRemoverFactory() #katadasar
stopword = factory.create_stop_word_remover()

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop = stopword.remove(isif)
#steming - mengambil kata dasar
stem = stemmer.stem(stop)
katadasar = stem.split()

matrix=[]
#cursor = conn.execute("SELECT* from ARTIKEL")

for row in cursor:
    tampung = []
    for i in katadasar:
        tampung.append(row[1].lower().count(i))
    matrix.append(tampung)

#print(katadasar)

write_csv("kata_before_%s.csv"%n, katadasar)

#nganu kbi
#Sharing kata Sesuai KBI Belum VSM
conn = sqlite3.connect('KBI.db')
cur_kbi = conn.execute("SELECT* from KATA")
    

def LinearSearch (kbi,kata):
    found=False
    posisi=0
    while posisi < len (kata) and not found :
        if kata[posisi]==kbi:
            found=True
        posisi=posisi+1
    return found

berhasil=[]
berhasil2=''
for kata in cur_kbi :
    ketemu=LinearSearch(kata[0],katadasar)
    if ketemu :
        kata = kata[0]
        berhasil.append(kata)
        berhasil2=berhasil2+' '+kata
#print(berhasil)
#menghitung tf-idf = menghitung jumlah fitur/kata
        #df = menghitung frekuensi - 
conn = sqlite3.connect('articles.sqlite')
matrix2=[]
cursor = conn.execute("SELECT* from ARTICLES")
for row in cursor:
    tampung = []
    for i in berhasil:
        tampung.append(row[1].lower().count(i))
        #print(tampung)
        #print(row[2])
    matrix2.append(tampung)
#print(matrix2)
#import csv kata yg sesuai dengan KBI

write_csv("kata_after_%s.csv"%n, berhasil)

#tf-idf
df = list()
for d in range (len(matrix2[0])):
    total = 0
    for i in range(len(matrix2)):
        if matrix2[i][d] !=0:
            total += 1
    df.append(total)

idf = list()
for i in df:
    tmp = 1 + log10(len(matrix2)/(1+i))
    idf.append(tmp)

tf = matrix2
tfidf = []
for baris in range(len(matrix2)):
    tampungBaris = []
    for kolom in range(len(matrix2[0])):
        tmp = tf[baris][kolom] * idf[kolom]
        tampungBaris.append(tmp)
    tfidf.append(tampungBaris)


write_csv("tfidf_%s.csv"%n, tfidf)

#seleksi fitur
def pearsonCalculate(data, u,v):
    "i, j is an index"
    atas=0; bawah_kiri=0; bawah_kanan = 0
    for k in range(len(data)):
        atas += (data[k,u] - meanFitur[u]) * (data[k,v] - meanFitur[v])
        bawah_kiri += (data[k,u] - meanFitur[u])**2
        bawah_kanan += (data[k,v] - meanFitur[v])**2
    bawah_kiri = bawah_kiri ** 0.5
    bawah_kanan = bawah_kanan ** 0.5
    return atas/(bawah_kiri * bawah_kanan)
def meanF(data):
    meanFitur=[]
    for i in range(len(data[0])):
        meanFitur.append(sum(data[:,i])/len(data))
    return np.array(meanFitur)
def seleksiFiturPearson(data, threshold, berhasil):
    global meanFitur
    data = np.array(data)
    meanFitur = meanF(data)
    u=0
    while u < len(data[0]):
        dataBaru=data[:, :u+1]
        meanBaru=meanFitur[:u+1]
        seleksikata=berhasil[:u+1]
        v = u
        while v < len(data[0]):
            if u != v:
                value = pearsonCalculate(data, u,v)
                if value < threshold:
                    dataBaru = np.hstack((dataBaru, data[:, v].reshape(data.shape[0],1)))
                    meanBaru = np.hstack((meanBaru, meanFitur[v]))
                    seleksikata = np.hstack((seleksikata, berhasil[v]))
            v+=1
        data = dataBaru
        meanFitur=meanBaru
        berhasil=seleksikata
        if u%50 == 0 : print("proses : ", data.shape)
        u+=1
    return data, seleksikata

xBaru2,kataBaru = seleksiFiturPearson(tfidf, 0.9, berhasil)
xBaru1,kataBaru2 = seleksiFiturPearson(xBaru2, 0.8, berhasil)

write_csv("kata_pearson_%s.csv"%n, kataBaru2)
#clustering
print("Cluster dgn Seleksi Fitur : 0.8")
cntr, u, u0, distant, fObj, iterasi, fpc =  fuzz.cmeans(xBaru1.T, 3, 2, 0.00001, 1000, seed=0)
membership = np.argmax(u, axis=0)

silhouette = silhouette_samples(xBaru1, membership)
s_avg = silhouette_score(xBaru1, membership, random_state=10)

for i in range(len(tfidf)):
    print("c "+str(membership[i]))#+"\t" + str(silhouette[i]))
print(s_avg)
#kmeans = KMeans(n_clusters=3, random_state=0).fit(xBaru)
#print(kmeans.labels_)

write_csv("Cluster%sFS8.csv"%n, [["Cluster"]])
write_csv("Cluster%sFS8.csv"%n, [membership],        "a")
write_csv("Cluster%sFS8.csv"%n, [["silhouette"]],    "a")
write_csv("Cluster%sFS8.csv"%n, [silhouette],        "a")
write_csv("Cluster%sFS8.csv"%n, [["Keanggotaan"]],   "a")
write_csv("Cluster%sFS8.csv"%n, u,                   "a")
write_csv("Cluster%sFS8.csv"%n, [["pusat Cluster"]], "a")
write_csv("Cluster%sFS8.csv"%n, cntr,                "a")
