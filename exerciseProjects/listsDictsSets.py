#This scripts are part of a progressive series of exercises used to practice with python

'''3. Lists, Dicts, and Sets Practice
	•	Lists:
	•	Create a list of 5 integers.
	•	Append a new integer, remove one, and print the final list.
	•	Iterate over the list and print each element squared.
	•	Dictionaries:
	•	Create a dictionary of 3 fruits and their prices.
	•	Add a new fruit, update one price, and remove one fruit.
	•	Iterate and print fruit → price.
	•	Sets:
	•	Create a set with 5 numbers, including duplicates.
	•	Add a new number, remove one number.
	•	Iterate over the set and print all elements.
	•	Show the difference between two sets (A − B).'''

#Lists
#used to store multiple items in a single variable
#List is a collection which is ordered and changeable. Allows duplicate members.

#create a list of 5 integers
myList = [1, 4, 8, 2, 3]
print(myList)

#append a new integer, remove one and print the final list
newInteger = 5
myList.append(newInteger)
myList.remove(1)
print(myList)

#iterate over the list and print each element squared
for n in myList:
    print(f"{n**2}")
    
#Dictionaries
#Dictionaries are used to store data values in key:value pairs.

#Create a dictionary of 3 fruits and their prices.

myDict = {
    "apple":1,
    "banana":2,
    "orange":1.5
}

print(myDict)

#Add a new fruit, update one price, and remove one fruit.
myDict["cherry"] = 2
print(myDict)
#or myDict.update({"cherry":2})

myDict["apple"] = 0.75
print(myDict)

myDict.pop("orange")
print(myDict)

#Iterate and print fruit → price.
for x in myDict:
    print(f"{x} -> {myDict[x]}")
    
#Sets
#Sets are used to store multiple items in a single variable.

#Create a set with 5 numbers, including duplicates.

myset = {2,2,4,3,7}
print(myset)
#Add a new number, remove one number.
myset.add(6)
myset.remove(2)
print(myset)
#Iterate over the set and print all elements.

for x in myset:
    print(x)
    
#Show the difference between two sets (A − B).
secondSet = {5,6,1,2}
print(secondSet)
diffSet = myset - secondSet
print(diffSet)