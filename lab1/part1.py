import requests

r = requests.post('https://courses.engr.illinois.edu/ece498icc/sp2020/lab1_string.php', data = {'netid':'bcallas2', 'name':'Brendan Callas'})
st = r.text

#for i in range(len(r.text)):
 #   if i % 498 == 0:
 #       print(st[i], end ="")
        
news = (st[i * 498] for i in range(400))
print(''.join(list(news)))