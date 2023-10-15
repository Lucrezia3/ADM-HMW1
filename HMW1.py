### HMW1

## PROBLEM 1

## INTRODUCTION
# Print Funcion
if __name__ == '__main__':
    n = int(input())
    string=''
    for i in range(1,n+1):
        string+=str(i)
    print(string)
    
# Say "Hello, World!" With Python
if __name__ == '__main__':
    print("Hello, World!")


# Python If-Else
import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
    if n%2>0 or (n%2==0 and n>=6 and n<=20):
        print('Weird')
    elif (n%2==0 and n>=2 and n<=5) or n>20:
        print('Not Weird')


# Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)

# Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)


# Loops
if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i**2)

# Write a function (C)
def is_leap(year):
    leap = False
    
    if (year%4==0 and year%100!=0) or year%400==0:
        leap = True
        
    else:
        leap = False
    
    
    return leap




## DATA TYPES
# List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    l=[0,0,0]
    d=[]
    L=[l]
    for i in range(x+1):
        for j in range(y+1):
            for k in range(z+1):
                l[0]=i
                l[1]=j
                l[2]=k
                l=list(l)
                L.append(l)

    L[:] = [i for n, i in enumerate(L) if n == L.index(i)]

    for i in L:
        if sum(i) != n:
            d.append(i)

    print(d)


# Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    
    l=list(arr)
    
    m=max(l)
    
    r=max([i for i in l if i < m])
    print(r)


# Nested Lists
if __name__ == '__main__':
    for _ in range(int(input())):
        name = input()
        score = float(input())

# Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()

# Lists
if __name__ == '__main__':
    N = int(input())
    l=[]
    for i in range(N):
        i=input()
        a=i.split()
        if a[0]=='insert':
            l.insert(int(a[1]), int(a[2]))
        elif a[0]=='print':
            print(l)
        elif a[0]=='remove':
            l.remove(int(a[1]))
        elif a[0]=='append':
            l.append(int(a[1]))
        elif a[0]=='sort':
            l.sort()
        elif a[0]=='pop':
            l.pop()
        elif a[0]=='reverse':
            l.reverse()



# Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t=tuple(integer_list)
    print(hash(t))


## STRINGS
# String Split and Join
def split_and_join(line):
    l=line.split(" ")
    line="-".join(l)
    return line
    
if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)
    

# sWAP cASE
def swap_case(s):
    string=''
    for i in range(len(s)):
        if s[i].isupper()==True:
            string+=s[i].lower()
        else:
            string+=s[i].upper()
    return string

# What's Your Name?
def print_full_name(first, last):
    c='Hello {0} {1}! You just delved into python.'.format(first, last)
    print(c)
    

# Mutations
def mutate_string(string, position, character):
    s = string[:position]+character+string[position+1:]
    return(s)


# Find a string
def count_substring(string, sub_string):
    a=0
    for i in range(len(string)):
        if string[i:len(sub_string)+i]==sub_string:
            a+=1
    return a


# String Validators (C)
if __name__ == '__main__':
    s = input()
    print(any(a.isalnum() for a in s))
    print(any(b.isalpha() for b in s))
    print(any(c.isdigit() for c in s))
    print(any(d.islower() for d in s))
    print(any(e.isupper() for e in s))
    

# Text Alignment
#Replace all ______ with rjust, ljust or center. 
thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))



# Text Wrap
def wrap(string, max_width):
    s=textwrap.wrap(string, max_width)
    w=''
    for i in s:
        w+=i+"\n"
    return w

# Capitalize!
def solve(s):
    l=s.split(' ')
    string=''
    for i in l:
        if i.islower()==True:
            i1=i.capitalize()
            string+=i1+' '
        else:
            string+=i+' '
    return string.strip()


# The Minion Game
def minion_game(string):
    v=['a','A','e','E','i','I','o','O','u','U']
    kevin=0
    stuart=0
    
    for i in range(len(string)):
        if string[i] in v:
            kevin+=(len(string)-i)
        else:
            stuart+=(len(string)-i)
    
    if kevin>stuart:
        print('Kevin {0}'.format(kevin))
    elif stuart>kevin:
        print('Stuart {0}'.format(stuart))
    else:
        print('Draw')
        


#SETS
# Introduction to Sets

# Symmetric Difference

# No Idea!

# Set .add()

# Set .discard(), .remove() & .pop()
 
# Set .union() Operation

# Set .intersection() Operation

# Set .difference() Operation

# Set .symmetric_difference() Operation

# Set .symmetric_difference() Operation

# Set Mutations





## COLLECTION
# collections.Counter()
from collections import Counter

X = int(input())
l = input().split()
N = int(input())
count = Counter(l)
r=0
for i in range(0,N):
    i = input().split()
    if count[i[0]] >0 :
        r+=int(i[1])
        count[i[0]]-=1
print(r)



# DefaultDict Tutorial (C)
from collections import defaultdict

d = defaultdict(list)
n, m = input().split()

for i in range(int(n)):
    i=input()
    d['A'].append(i)
    
for i in range(int(m)):
    i=input()
    d['B'].append(i)

for i in d['B']:
    l=[]
    for ind in range(int(n)):
        if i == d['A'][ind]:
            l.append(ind+1)
        elif i not in d['A']:
            l=[-1]
            break
    print(*l)
    
    
# Collections.namedtuple()
from collections import namedtuple

N = int(input())
h = input()
student = namedtuple('Student',h)
s=0
for i in range(N):
    i = student(*input().split())
    s += int(i.MARKS)
print(float(s/N))


# Collections.OrderedDict() (C)
from collections import OrderedDict

N= int(input())
d = OrderedDict()
p=[]
n=''
for i in range(N):
    i = input().split()
    p = [x for x in i if x.isalpha()]
    n= [x for x in i if x.isdigit()]
    
    if len(p)== 2 and (p[0]+' '+p[1]) not in d.keys():
        d[p[0]+' '+p[1]]=int(n[0])
    elif len(p)== 1 and (p[0]) not in d.keys():
        d[p[0]]=int(n[0])
    elif len(p)== 2:
        d[p[0]+' '+p[1]]+=int(n[0])
    elif len(p)==1:
        d[p[0]]+=int(n[0])

for i in d.items():
    print(*i)
    

# Word Order
from collections import defaultdict

n = int(input())
d= defaultdict(list)

for i in range(n):
    i=input()
    if i not in d.keys():
        d[i]=1
    else:
        d[i]+=1
print(len(d.keys()))
print(*d.values())

# Collections.deque()
from collections import deque

N=int(input())
d=deque()

for i in range(N):
    i=input().split()
    if i[0]=='append':
        d.append(i[1])
    elif i[0]=='pop':
        d.pop()
    elif i[0]=='popleft':
        d.popleft()
    elif i[0]=='appendleft':
        d.appendleft(i[1])
print(*d)

# Company Logo
import math
import os
import random
import re
import sys
from collections import Counter


if __name__ == '__main__':
    s = input()
    S=sorted(s)
    count = Counter(S)
    for i, n in count.most_common(3):
        print(i, n)




## DATA and TIME
# Calendar Module
import calendar
import datetime

i = input().split()
MM = int(i[0])
DD = int(i[1])
YYYY = int(i[2])
d = datetime.date(YYYY,MM,DD)
print(calendar.day_name[d.weekday()].upper())


# Time Delta
import math
import os
import random
import re
import sys
from datetime import *

# Complete the time_delta function below.
def time_delta(t1, t2):
    f = '%a %d %b %Y %H:%M:%S %z'
    t1 = datetime.strptime(t1, f)
    t2 = datetime.strptime(t2, f)
    time_delta=str(int(abs((t1-t2).total_seconds())))
    return time_delta


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()




## Exceptions
T = int(input())

for i in range(T):
    i = input().split()
    a = i[0]
    b = i[1]
    try:
        print(int(a)//int(b))
    except Exception as ex:
        print("Error Code: {0}".format(ex))
        
   
   
        
## BUILT-INS
# Zipped!
N, X = input().split()
l=[]

for i in range(int(X)):
    i = map(float,input().split())
    l.append(i)

z = zip(*l)

for i in z:
    print(sum(i)/int(X))
    

# Athlete Sort
import math
import os
import random
import re
import sys


if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    A = sorted(arr, key = lambda x : x[k])
    for i in A:
        print(*i)
        

# Any or All (C)
def is_palindromic_integer(n):
    return str(n) == str(n)[::-1]

def check_condition(n, num_list):
    all_positive = all(x > 0 for x in num_list)
    any_palindromic = any(is_palindromic_integer(x) for x in num_list)
    return all_positive and any_palindromic


n = int(input().strip())
num_list = list(map(int, input().strip().split()))


print(check_condition(n, num_list))




## PYTHON FUNCTIONALS
# Map and Lambda Function
cube = lambda x: x**3

def fibonacci(n):
    l = [0,1]
    for i in range(2,n):
        l.append(l[i-2] + l[i-1])
    return(l[0:n])

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))
    



## Regex and Parsing challenges (C)
# Detect Floating Point Number
from re import *

pattern = compile('^[-+]?[0-9]*\.[0-9]+$')
N = int(input())

for i in range(N):
    i = input()
    mat = pattern.match(i)
    print(bool(mat))


# Re.split()
regex_pattern = r"[.,]+"

import re
print("\n".join(re.split(regex_pattern, input())))


# Group(), Groups() & Groupdict()
import re

S=input()
m = re.search(r'([a-zA-Z0-9])\1', S.strip())
if m:
    print(m.group(1))
else:
    print(-1)



## XML
# XML 1 - Find the Score
import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    count = 0
    for i in node:
        count = count + get_attr_number(i)
    return count + len(node.attrib)


if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))
    

# XML2 - Find the Maximum Depth
import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):
    global maxdepth
    if (level == maxdepth):
        maxdepth += 1
    for i in elem:
        depth(i, level + 1)
        
        
if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)





## Closures and Decorations 
# Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        f(['+91 ' + o[-10:-5] + ' ' + o[-5:] for o in l])
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 
    

# Decorators 2 - Name Directory (C)
import operator

def person_lister(f):
    def inner(people):
         return [f(person) for person in sorted(people, key=lambda x: int(x[2]))]
    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')
    
    
    
    
## Numpy
# Arrays
def arrays(arr):
    return numpy.array(arr[::-1],float)

arr = input().strip().split(' ')
result = arrays(arr)
print(result)


# Shape and reshape
import numpy 

i = numpy.array(list(input().split()),int)
print(i.reshape(3,3))

# Transpose and Flatten
import numpy

N, M = map(int,input().split())
r=[]
for i in range(N):
    i= list(map(int,input().split()))
    r.append(i)

r = numpy.array(r)
print(numpy.transpose(r))
print(r.flatten())

# Concatenate
import numpy

N,M,P = map(int,input().split())
r = []
for i in range(N+M):
    i = list(map(int,input().split()))
    r.append(i)

r= numpy.array(r)
print(r)

# Zeros and Ones
import numpy

i = list(map(int, input().split()))
r0 = nnumpy.zeros(i,int)
r1 = numpy.ones(i,int)
print(r0)
print(r1)




## PROBLEM 2
# Birthday Cake Candles
import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    m = max(candles)
    count = 0
    for i in candles:
        if i == m:
            count+=1
    return count
    
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()
    
    
# Kangaroo
import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    x = 10000
    i=0
    r = 'NO'
    if x1==x2:
        r = 'YES'
    while i < x:
        x1+=v1
        x2+=v2
        
        if x1==x2:
            r = 'YES'
        i+=1
    return r

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()


# Viral Advertising
import math
import os
import random
import re
import sys

#
# Complete the 'viralAdvertising' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER n as parameter.
#

def viralAdvertising(n):
    d={1:[5, int(5/2), int(5/2)]}
    for day in range(2,n+1):
        s = (d[day-1][1])*3
        l = int(s/2)
        c = d[day-1][2]+l
        if day not in d.keys():
            d[day]=[s, l, c]
    return d[n][2]
    
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()
    
# Recursive Digit Sum
import math
import os
import random
import re
import sys

#
# Complete the 'superDigit' function below.
#
# The function is expected to return an INTEGER.
# The function accepts following parameters:
#  1. STRING n
#  2. INTEGER k
#

def superDigit(n, k):
    p = str(k*sum(int(x) for x in n))
    while len(p)>1:
        p=str(sum((int(i) for i in p)))
    return p

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()


# Insertion Sort - Part 1 (C)
import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort1' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort1(n, arr):
    l=n-1
    a = arr[n-1]

    for i in range(l-1, -1, -1):
        if a < arr[i]:
            arr[i+1]=arr[i]
            print(*arr)
        else:
            arr[i+1] = a
            print(*arr)
            break
    
    if arr[0] > a:
        arr[0] = a
        print(*arr)

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)


# Insertion Sort - Part 2 (C)
import math
import os
import random
import re
import sys

#
# Complete the 'insertionSort2' function below.
#
# The function accepts following parameters:
#  1. INTEGER n
#  2. INTEGER_ARRAY arr
#

def insertionSort2(n, arr):
    for i in range(1, n):
        e = arr[i]
        j = i - 1

        while j >= 0 and arr[j] > e:
            arr[j + 1] = arr[j]
            j -= 1
        
        arr[j + 1] = e
        print(*arr)

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)
