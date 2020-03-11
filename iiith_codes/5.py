import math 
a = 1
b = 1
c = 1
s = (a+b+c)/2
sq_a  = s*(s-a)*(s-b)*(s-c)

if sq_a<=0:
    print("Area is zero")
else:
    print("Area is: ", math.sqrt(sq_a))
    
 