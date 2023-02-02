from interface import *


def main():
    color = int(input("Rate color of flower (1/0): "))
    form = int(input("Rate form of flower (1/0): "))
    flavour = int(input("Rate flavour of flower (1/0): "))
    
    predict = nn.predict([color, form, flavour])
    
    if predict > 0.5:
        print("Flower is excelent! Your girl will like it ;)")
    else:
        print("Nah bro! Choose another flower.")
    
if __name__ == "__main__":
    main()