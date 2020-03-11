string = "())(()()"
s_find = "()"
string = string.replace(s_find,"")
if len(string)==0:
    print("Balanced String")
else:
    print("Unbalanced String")
    