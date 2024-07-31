l1 = list(map(int, input().split()))
l2 = list(map(int, input().split()))
count = 0
for i in l1:
    if i in l2:
        count += 1
print(count)
