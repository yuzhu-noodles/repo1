num = int(input("请输入一个数字"))
sum = 0
n = len(str(num))
temp = num

while temp > 0:
    x = temp%10
    sum += x**n
    temp //=10

if sum == num:
    print("是")
else:
    print("否")






