import csv
import os.path

def writecsv(x,y,r,clas):
    x = list(map(float,x))
    y = list(map(float,y))
    r = list(map(float,r))
    size = len(x)
    path = 'data.csv'
    values =[] 
    for value in [list(val) for val in zip(x,y,r)]:
        values.extend(value)
    values.append(clas)
    if os.path.isfile(path):
        with open(path,'r') as file:
            reader = csv.reader(file)
            rows = len([row for row in reader])
        with open(path,'a') as file:
            writer = csv.writer(file)
            writer.writerow(values)
    else:
        with open(path,'w') as file:
            writer = csv.writer(file)
            head = []
            for i in range(1,size+1):
                head.extend(['x'+str(i), 'y'+str(i), 'r'+str(i)])
            head.append('class')
            writer.writerows([head,values])
            rows = 1                   
    print(values, rows)
    return rows

