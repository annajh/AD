__author__ = 'jennytou'
import Tkinter as tk
import tkFileDialog as tkFile
import os
import pyglet
from threading import Thread

class Example(tk.Frame):
    dirAlz = ''
    dirCtr = ''
    filename = ''
    soundfile = ''
    #player = pyglet.media.Player()


    def __init__(self, parent):

        classification = tk.IntVar()
        kernel = tk.IntVar()

        labelText = tk.StringVar()

        tk.Frame.__init__(self, parent)

        self.master.title("CogID")

        #layout the UI
        tk.Label(self, text = "Classification Method", borderwidth = 1).grid(row = 1, column = 1, columnspan = 4, sticky = tk.W)
        tk.Label(self, text = "Alzheimer's Audio Files", borderwidth = 5).grid(row = 1, column = 5, sticky = tk.W)
        tk.Label(self, text = "Control's Audio Files", borderwidth = 5).grid(row = 1, column = 6, sticky = tk.W)

        tk.Radiobutton(self, text = "KNN", variable = classification, value = 1).grid(row = 2, column = 1, columnspan = 2, sticky = tk.W)
        tk.Radiobutton(self, text = "SVM", variable = classification, value = 2).grid(row = 2, column =3, columnspan = 2, sticky = tk.W)

        tk.Label(self, text = "Parameters", borderwidth = 1).grid(row = 3, column = 1, sticky = tk.W)
        tk.Label(self, text = "KNN", borderwidth = 1).grid(row = 4, column = 1, sticky = tk.W)
        tk.Label(self, text = "SVM", borderwidth = 1).grid(row = 5, column = 1, sticky = tk.W)

        tk.Label(self, text = "k = ", borderwidth = 1).grid(row = 4, column = 2, sticky = tk.W)
        tk.Label(self, text = "Kernel", borderwidth = 1).grid(row = 5, column = 2, sticky = tk.W)

        tk.Entry(self, width = 3).grid(row = 4, column = 3, columnspan = 1, sticky = tk.W)

        tk.Radiobutton(self, text = "Linear", variable = kernel, value = 1).grid(row = 5, column = 3, sticky = tk.W)
        tk.Radiobutton(self, text = "RBF", variable = kernel, value = 2).grid(row = 5, column = 4, sticky = tk.W)

        tk.Label(self, text = "Selected File", borderwidth = 1).grid(row = 6, column = 1, columnspan = 4, sticky = tk.W)
        tk.Entry(self).grid(row = 7, column = 1, columnspan = 4, sticky = tk.W)

        tk.Button(self, text = "Listen", borderwidth = 1, command = self.listenAlz).grid(row = 8, column = 1, columnspan = 2, sticky = tk.W)
        tk.Button(self, text = "Classify", borderwidth = 1).grid(row = 8, column = 3, columnspan = 2, sticky = tk.W)

        tk.Label(self, text = "Predicted Label").grid(row = 9, column = 1, columnspan = 2, sticky = tk.W)
        resultLabel = tk.Label(self, textvariable = labelText).grid(row = 9, column = 2, columnspan = 2, sticky = tk.W)

        correctLabel = tk.Label(self, textvariable = labelText).grid(row = 10, column = 1, columnspan = 4, sticky = tk.W)

        self.dataAlz = tk.Listbox(self, selectmode = "Single")
        self.dataAlz.bind('<<ListboxSelect>>',self.onselectAlz)
        self.dataAlz.grid(row = 2, column = 5, rowspan = 8)
        self.dataCtr = tk.Listbox(self, selectmode = "Single")
        self.dataCtr.grid(row = 2, column = 6, rowspan = 8)

        self.scrollbarAlz = tk.Scrollbar(self.dataAlz, orient= tk.VERTICAL)
        self.scrollbarCtr = tk.Scrollbar(self.dataCtr, orient = tk.VERTICAL)

        self.dataAlz.config(yscrollcommand=self.scrollbarAlz.set)
        self.dataCtr.config(yscrollcommand=self.scrollbarCtr.set)
        self.scrollbarAlz.config(command=self.dataAlz.yview)
        self.scrollbarCtr.config(command=self.dataCtr.yview)

        tk.Button(self, text = "Select Directory", borderwidth = 1, command=self.askdirectoryAlz).grid(row = 10, column = 5, sticky = tk.W)
        tk.Button(self, text = "Select Directory", borderwidth = 1, command=self.askdirectoryCtr).grid(row = 10, column = 6, sticky = tk.W)

        # defining options for opening a directory
        self.dir_opt = options = {}
        options['initialdir'] = 'C:\\'
        options['mustexist'] = True
        options['parent'] = self
        options['title'] = 'This is a title'


    def askdirectoryAlz(self):
        dirAlz = tkFile.askdirectory(**self.dir_opt)
        #if os.path.isfile(dirAlz):
        for line in os.listdir(dirAlz):
            if line.endswith(".mp3"):
                self.dataAlz.insert(tk.END, line)
                #print line
        self.dataAlz.update()

    def askdirectoryCtr(self):
        dirCtr = tkFile.askdirectory(**self.dir_opt)
        #if os.path.isfile(dirAlz):
        for line in os.listdir(dirCtr):
            if line.endswith(".mp3"):
                self.dataCtr.insert(tk.END, line)
                #print line
        self.dataCtr.update()

    def onselectAlz(self,evt):
        w = evt.widget
        index = self.dataAlz.curselection()
        sound2 = tk.StringVar()
        sound2.set(w.get(index))
        tk.Entry(self, textvariable = sound2).grid(row = 7, column = 1, columnspan = 4, sticky = tk.W)
        print sound2

    def listenAlz(self):
        soundfile = self.dataAlz.get(self.dataAlz.curselection())
        filename = soundfile[0:5]
        global listen_thread
        listen_thread = Thread(target=startPlaying)
        listen_thread.start()

    def listenCtr(self):
        soundfile = self.dataCtr.get(self.dataAlz.curselection())
        filename = soundfile[0:5]
        global listen_thread
        listen_thread = Thread(target=startPlaying)
        listen_thread.start()

    def startPlaying(self):
        sound = pyglet.media.load(file)
        sound.play()
        pyglet.app.run()

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