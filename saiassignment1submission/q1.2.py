x=int(input("Enter your 1st number:"))
y=int(input("Enter your 2nd number:"))
while x<=y:
    if x%2==0 and x!=2:
        x+=1
        continue
    i=1
    k=0
    while i<=x:
        if x%i==0:
            k+=1
        i+=2
    if k<=2 and x!=1:
        print(x)
    x+=1        

