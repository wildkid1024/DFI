import os

dd = {"aa":{"bb":1, "cc": 2}, "dd":3 }
kk = {"aa.bb.ee":1, "aa.cc":2, "dd":3}

tmp = {}
for k,v in kk.items():
    ss = tmp
    for attr in k.split(".")[:-1]:
        if attr not in ss:
            ss[attr] = {}
        ss = ss[attr]
    ss[k.split(".")[-1]] = v
    
print(tmp)