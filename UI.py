__author__ = 'jennytou'
import Tkinter as tk

class Example(tk.Frame):
    def __init__(self, parent):
        tk.Frame.__init__(self, parent)

        #layout the UI
        tk.Label(self, text = "Classification Method", borderwidth = 5).grid(row = 1, column = 1)
        tk.Label(self, text = "Alzheimer's Audio Files", borderwidth = 5).grid(row = 1, column = 2)
        tk.Label(self, text = "Control's Audio Files", borderwidth = 5).grid(row = 1, column = 4)

        self.master.title("CogID")


    def calculate(self):
        # get the value from the input widget, convert
        # it to an int, and do a calculation
        try:
            i = int(self.entry.get())
            result = "%s*2=%s" % (i, i*2)
        except ValueError:
            result = "Please enter digits only"

        # set the output widget to have our result
        self.output.configure(text=result)