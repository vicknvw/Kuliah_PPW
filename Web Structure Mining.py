import requests
from bs4 import BeautifulSoup

import pandas as pd

import networkx as nx
import matplotlib.pyplot as plt

def simplifiedURL(url):
    if "www." in url:
        ind = url.index("www.")
        url = url[ind:]
    if not "http" in url:
        url = "http://"+url
    if url[-1] == "/":
        url = url[:-1]

    parts = url.split("/")
    url = ''
    for i in range(3):
        url += parts[i] + "/"
    return url

def crawl(url, max_deep,  show=False, deep=0, done=[]):
    global edgelist
    url = simplifiedURL(url)
    deep += 1
    if not url in done:
        links = getLink(url)
        done.append(url)
        if show:
            if deep == 1:
                print("(%d)%s" %(len(links),url))
            else:
                print("|", end="")
                for i in range(deep-1): print("--", end="")
                print("(%d)%s" %(len(links),url))

        for link in links:
            edge = (url,link)
            if not edge in edgelist:
                edgelist.append(edge)
            if (deep != max_deep):
                crawl(link, max_deep, show, deep)

def getLink(src):
    try:
        ind = src.find(':')+3
        url = src[ind:]
        page = requests.get(src)
        soup = BeautifulSoup(page.content, 'html.parser')
        a = soup.findAll('a')
        temp = []
        for i in a :
            try:
                link = i['href']
                if not link in temp and 'http' in link :
                    temp.append(link)
            except KeyError:
                pass
        return temp
        #print(temp)
    except:
        return list()
                
root = "https://wiraraja.ac.id/"
#root = "https://ui.ac.id/"
s = True
edgelist = []
crawl(root, 3, show=s)
edgeListFrame = pd.DataFrame(edgelist, None, ("From", "To"))
#print(edgelist)
g = nx.from_pandas_edgelist(edgeListFrame, "From", "To", None, nx.DiGraph())
pos = nx.random_layout(g)

pr = nx.pagerank(g)

print("-----")
nodelist = [root]
nodelist = g.nodes
label= {}
data = []
for i, key in enumerate(nodelist):
    data.append((pr[key],key))
    label[key]=i

pd.set_option('display.max_rows', 1000)
tabel = pd.DataFrame(data, columns=("Pagerank", "Links"))
u = tabel.sort_values(by=["Pagerank", "Links"], ascending=[True,False])

print(u)

nx.draw(g, pos)
nx.draw_networkx_labels(g, pos, label)

plt.axis("off")
plt.show()
