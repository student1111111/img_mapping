# importing csv module
import csv

# initializing the titles and rows list
fields = []
rows = []

# reading csv file
with open('sample-dataset.csv', 'r') as csvfile:
    # creating a csv reader object
    csvreader = csv.reader(csvfile)

    # extracting field names through first row
    fields = next(csvreader)

    # extracting each data row one by one
    for row in csvreader:
        rows.append(row)

# printing the field names
#print(fields)


machine=[]
for row in rows:
    if row[3] not in machine:
        machine.append(row[3])
print("\n>>>>>>>>>>Machine used are :")
for m in machine:
    print("             >>>",m)

apps=[]
for row in rows:
    if row[7] not in apps:
        apps.append(row[7])
print("\n>>>>>>>>>>Applications used by  user  are :")

for a in apps:
    print("             >>>",a)


times=[]
for row in rows:
    if row[4].split(" ")[1][:5] not in times:
        times.append(row[4].split(" ")[1][:5])
print("\n>>>>>>>>>>user uses machine at following time :")

for t in times:
    print("             >>>",t)

print(len(rows))

users=[]
for row in rows:
    if row[1] not in users:
        users.append(row[1])
print("\n>>>>>>>>>>total number of  UUID  are :" , len(list(set(users))))


#MOUSE_CLICK LEFT_MOUSE


right_click=0
left_click=0

for row in rows:
    if row[11]=="MOUSE_CLICK":right_click+=1

    if row[12]=="LEFT_MOUSE":left_click+=1
print("\n>>>>>>>>>>Mouse button Right clicked for :", right_click)
print("\n>>>>>>>>>>Mouse button Left clicked for :", left_click)
