L = 1
R = 1000
def check_arm(num):
    s = 0
    n = num
    while(num):
        dig =  num%10
        num =  int(num/10)
        s = s + dig**3
    if n==s:
        return 1
    else:
        return 0

num_arm = 0     
for num in range(L,R):
    num_arm+=check_arm(num)
    
print(num_arm)
    