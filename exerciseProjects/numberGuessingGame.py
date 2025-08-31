#This scripts are part of a progressive series of exercises used to practice with python

#Number guessing game. 
#Program randomly selects an integer between 1 and 100 (use random.randint).
""" 	•	User tries to guess the number.
	•	After each guess, print:
	•	“Too high” if guess > number
	•	“Too low” if guess < number
	•	“Correct!” if guess == number
	•	Game continues until the correct guess.
	•	(Optional) Keep track of the number of attempts and print it at the end. """


def play_game(MAX_NUM = 10):
    import random
    num = random.randint(1,MAX_NUM)
    solved = False
    attempts = 0

    print("Welcome to the number game")
    print(f"Guess a number between 1 and {MAX_NUM}")

    while not solved:

        try:
            guess = int(input("Please enter your guess:"))
        except ValueError:
            print("That's not a number! Try again.")
            continue

        if guess == num:
            attempts += 1
            print(f"Correct, number of attempts: {attempts}")
            solved = 1
        elif guess > num:
            print("Too high!")
            attempts += 1
        elif guess < num:
            print("Too low!")
            attempts += 1


# this block ensures the game only starts when the script is run directly
if __name__ == "__main__":
    play_game()