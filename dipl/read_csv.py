
import csv

with open ('C:\\Users\\greg\\Desktop\\workspace\\dip\\oasis_cross-sectional.csv') as csvfile:
  
    readCSV =  csv.reader(csvfile, delimiter = ',')

    pats = []
    genders = []
    for x in readCSV: # each x is a row in readCSV
        pat = x[0] # x[0] is the first element of the row
        gender = x[1] #x[1] etc

        pats.append(pat)
        genders.append(gender) #by the end of the for loop we have readCSV's columns 0 and 5
    
print(pats)
print(genders)

id = input('give the id of the patient of which you need to know the gender')
padex = pats.index(id) #list.index(thing_i_search_for) returns the location (padex is the index or location)

theGender = genders[padex]
print('The gender of patient', id,' is ',theGender)
