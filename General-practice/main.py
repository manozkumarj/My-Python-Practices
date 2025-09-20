def deposit():
    while True:
        amount = input("How much amount do you want to deposit? $")
        if amount.isdigit():
            amount = int(amount)
            if amount > 0:
                break
            else:
                print("Amount must be greater than Zero.")
        else:
            print("Please enter a number.")
    return amount

deposit()