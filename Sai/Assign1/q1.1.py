i=1
while i<=100:
    if i%3==0:
        if i%5==0:
            print("FizzBuzz")
            i+=1
            continue
        print("Fizz")
        i+=1
        continue
    if i%5==0: 
        print("Buzz")
        i+=1
        continue
    print(i)
    i+=1
    