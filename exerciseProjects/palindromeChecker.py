#This scripts are part of a progressive series of exercises used to practice with python

#Palindrome checker
'''	•	Write a function is_palindrome(s: str) -> bool.
	•	Input: a string s.
	•	Return True if the string is a palindrome (same forwards and backwards), else False.
	•	Ignore spaces, capitalization, and punctuation.
	•	Example:
	•	"Radar" → True
	•	"A man a plan a canal Panama" → True
	•	"Hello" → False '''

import string

def is_palindrome(s: str) -> bool:
    import numpy

    #remove punctuation
    s = s.replace(",", "")
    s = s.replace(".", "")
    s = s.replace(";", "")

    #remove capitalization
    s = s.lower()

    #remove spaces
    s = "".join(s.split())

    print(f"{s}")

    #check the number of letters
    letterCount = int(len(s))
    print(f"Letter count = {letterCount}")

    if letterCount & 1: #the number is odd, ignore the middle letter
        firstHalf = s[:int(numpy.floor((letterCount/2)))]
        secondHalf = s[int(numpy.floor((letterCount/2)))+1:]
        secondHalf = secondHalf[::-1]
    else: 
        firstHalf = s[0:int(letterCount/2)]
        secondHalf = s[int(letterCount/2):]
        secondHalf = secondHalf[::-1]
    
    print(firstHalf)
    print(secondHalf)

    if firstHalf==secondHalf:
        return True
    else:
        return False

def is_palindrome_pro(s: str) -> bool:
    s = ''.join(ch.lower() for ch in s if ch.isalnum())
    return s == s[::-1]

# this block ensures the game only starts when the script is run directly
if __name__ == "__main__":
    word = input("Enter a string to check for palindrome: ")
    if is_palindrome(word):
        print("palindrome")
    else:
        print("not palindrome")

    if is_palindrome_pro(word):
        print("palindrome pro")
    else:
        print("not palindrome pro")