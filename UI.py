__author__ = 'jennytou'
import Tkinter as tk

class Example(tk.Frame):
    def __init__(self, parent):
        classification = tk.IntVar()
        kernel = tk.IntVar()

        tk.Frame.__init__(self, parent)

        #layout the UI
        tk.Label(self, text = "Classification Method", borderwidth = 1).grid(row = 1, column = 1, columnspan = 4)
        tk.Label(self, text = "Alzheimer's Audio Files", borderwidth = 5).grid(row = 1, column = 5)
        tk.Label(self, text = "Control's Audio Files", borderwidth = 5).grid(row = 1, column = 6)

        tk.Radiobutton(self, text = "KNN", variable = classification, value = 1).grid(row = 2, column = 1, columnspan = 2)
        tk.Radiobutton(self, text = "SVM", variable = classification, value = 2).grid(row = 2, column =3, columnspan = 2)

        tk.Label(self, text = "Parameters", borderwidth = 1).grid(row = 3, column = 1)
        tk.Label(self, text = "KNN", borderwidth = 1).grid(row = 4, column = 1)
        tk.Label(self, text = "SVM", borderwidth = 1).grid(row = 5, column = 1)

        tk.Label(self, text = "k = ", borderwidth = 1).grid(row = 4, column = 2)
        tk.Label(self, text = "Kernel", borderwidth = 1).grid(row = 5, column = 2)

        tk.Radiobutton(self, text = "Linear", variable = kernel, value = 1).grid(row = 5, column = 3)
        tk.Radiobutton(self, text = "RBF", variable = kernel, value = 2).grid(row = 5, column = 4)



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