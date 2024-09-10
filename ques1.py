vowels = ['a', 'e', 'i', 'o', 'u']
#Write a program to count the number of vowels and consonants present in an input string.
consonants = ['b', 'c', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'v', 'w', 'x', 'y', 'z']
mystr = input("Enter the string: ")
print("You entered a word " + mystr + " with both vowels and consonants")
vowels_count = 0
consonants_count = 0

vowels_in_mystr = [char for char in mystr.lower() if char in vowels]
consonants_in_mystr = [char for char in mystr.lower() if char in consonants]

print("Vowels: ", vowels_in_mystr)
print("Consonants: ", consonants_in_mystr)
for char in mystr:
    if char.isalpha():
        if char in vowels:
            vowels_count += 1
        else:
            consonants_count += 1

print("Number of vowels: ", vowels_count)
print("Number of consonants: ", consonants_count)