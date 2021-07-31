# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 18:40:15 2020

@author: MVR
"""


import sentiment_mod as sm
from tkinter import *
from tkinter import font 
from PIL import ImageTk, Image
from tkinter import scrolledtext

global review_txt


#********************************FINAL ANALYSIS***********************************

def analysis(rev):
    
    analysis_window=Toplevel()
    analysis_window.title("Analysis!")
    analysis_window.geometry("500x300")
    analysis_window.configure(bg="#ffcd3c")
    
    review_analysis= sm.sentiment(rev)
    
    if review_analysis[0] == "neg":
        analyzed_review=Label(analysis_window, text="Your review is negative!", bg="#ffcd3c",font="Consolas 20 bold")
        analyzed_review.grid(row="0", column="0", padx="250", pady="100")
    else:
        analyzed_review=Label(analysis_window, text="Your review is positive!", bg="#ffcd3c",font="Consolas 20 bold")
        analyzed_review.grid(row="0", column="0", padx="120", pady="100")
    
    #print(sm.sentiment("the movie was beautiful. i loved it and it was very fun to watch, the acting was the best!")) 
    
    
    
    

#********************************DETAILS PAGE*****************************************

def details():
    
    details_window=Toplevel()
    details_window.title("Enter Details")
    details_window.geometry("1000x700")
    details_window.configure(bg="#ffcd3c")
    
    book_name=Label(details_window, text="Book Name :",bg="#ffcd3c",font="Arial 15 bold")
    book_name.grid(row="0", column="0", padx=(200, 20), pady=(100,50))
    
    book_name_txt= Text(details_window, height="1", width="40",font="Arial 12")
    book_name_txt.grid(row="0", column="1",pady=(100,50))
    
    author_name=book_name=Label(details_window, text="Author :",bg="#ffcd3c",font="Arial 15 bold")
    author_name.grid(row="1", column="0", padx=(200, 20))
    
    author_name_txt= Text(details_window, height="1", width="40",font="Arial 12")
    author_name_txt.grid(row="1", column="1")
    
    review=Label(details_window, text="Review :",bg="#ffcd3c",font="Arial 15 bold")
    review.grid(row="2", column="0", padx=(200, 20), pady="40")
    
    review_txt=scrolledtext.ScrolledText(details_window,height="10", width="40",font="Arial 12")
    review_txt.grid(row="2", column="1",  pady="40")
    
    analyze_btn=Button(details_window, text="Analyze", height="1", width="15", font="Consolas 14 italic", bg="#d92027", fg="#ffcd3c",command=lambda: analysis(review_txt.get("1.0","end-1c")))
    analyze_btn.grid(row="3", column="0", columnspan="2",padx="400", sticky="ne")


#**********************************MAIN PAGE*****************************************
root= Tk()
root.geometry("1000x700")
root.title("Book Review Analyzer")
root.configure(bg="#ffcd3c")

image=Image.open("logo.png")
image= image.resize((300, 250), Image.ANTIALIAS)
logo_img=ImageTk.PhotoImage(image)

logo=Label(root, image=logo_img, bg="#ffcd3c" )
logo.grid(row="0", column="0", padx="230", pady=(80,0))

heading=Label(root, text="Book Review Analyzer", font=("Comic Sans MS", 40, "italic"), bg="#ffcd3c", fg="#d92027")
heading.grid(row="1", column="0", padx="230", pady=(25,0))

start_btn=Button(root, text="Get started!", height="1", width="15", font="Consolas 14 italic", bg="#d92027", fg="#ffcd3c", command=details)
start_btn.grid(row="2", column="0", padx="140", pady="20")


root.mainloop()


#print(sm.sentiment("the movie was very bad. the acting was the worst and i did not like it at all. it was awful and outrageous."))
#print(sm.sentiment("the movie was beautiful. i loved it and it was very fun to watch, the acting was the best!")) 