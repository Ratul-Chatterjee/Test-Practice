print("Enter first number: ")
num1 = input()
print("Enter second number: ")
num2 = input()
print("Enter the operation: ")
op = input()
print("The result is: ")
if op == "+":
    print(int(num1) + int(num2))
elif op == "-":
    print(int(num1) - int(num2))
elif op == "*":
    print(int(num1) * int(num2))
elif op == "/":
    print(int(num1) / int(num2))
else:
    print("Invalid operation")
