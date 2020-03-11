# I am also assuming that strings of the form [(]) are balanced - just wanted to make that clear
string = "()([)[()]{()[]}][(]{)}"

s_find_1 = "()"
s_find_2 = "[]"
s_find_3 = "{}"
 
# Strings that are further mentioned are for the cases mentioned in the starting line comment

s_find_4 = "(]"
s_find_5 = "(}"
s_find_6 = "[)"
s_find_7 = "[}"
s_find_8 = "{]"
s_find_9 = "{)"

string = string.replace(s_find_1,"")
string = string.replace(s_find_2,"")
string = string.replace(s_find_3,"")

string_prev = ""

# Recurrently removing the strings that are mentioned above
while(1):
    string_prev = string
    string = string.replace(s_find_4,"")
    string = string.replace(s_find_5,"")
    string = string.replace(s_find_6,"")
    string = string.replace(s_find_7,"")
    string = string.replace(s_find_8,"")
    string = string.replace(s_find_9,"")
    if string_prev == string:
        break

print(string)

if len(string)==0:
    print("Balanced String")
else:
    print("Unbalanced String")
    