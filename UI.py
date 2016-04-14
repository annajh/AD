__author__ = 'jennytou'
import Tkinter as tk
import tkFileDialog as tkFile
import os
import pyglet
from threading import Thread
import data
import SVM
import KNN
import visualize_features
import UI
from sklearn.mixture import GMM
import numpy as np
from sklearn import metrics
import tkMessageBox


def classify_function(filename, method, k, kernal,n):
    if filename == '':
        raise data.ValidationError('please select file')
    X, Y, subjectID = data.load_data("control_features_combinedSubject.txt", "dementia_features_combinedSubject.txt")
    X = data.get_useful_features_mat(X)

    alz_count = 0
    for y in Y:
        if y:
            alz_count = alz_count + 1

	#normalize features
	X_scaled, normalizer = data.normalize_features(X)
	#print X_scaled


    #print filename
    testList = []
    filenameSubj = filename[0:3]
    testList.append(filenameSubj)
    try:
        X_train, Y_train, X_test, Y_test, trainID, testID = data.split_train_test(X_scaled,Y,subjectID,testID=testList)
    except ValueError:
        print filenameSubj
        raise data.ValidationError("combined data missing!")

    #PCA
    if n < 1 and n != -1:
        raise data.ValidationError('# features has to be greater than 0')
    elif n > 16:
        raise data.ValidationError('# features has to be less than 16')
    elif (n >= 1 and n <= 16):
        pca, explained_variance_ratio_ = data.reduce_dimension(X_train, n)
        X_train = pca.transform(X_train)

    #load the real testing data! Y_test should remain the same!
    X_test = data.load_testing_visit("control_features_per_visit.txt", "dementia_features_per_visit.txt", filename[0:5])
    #print "visit"
    #print X_test
    X_test = data.get_useful_features_mat(X_test)
    #print "visit"
    #print X_test
    X_test = normalizer.transform(X_test)
    #print "visit"
    #print X_test
    #if use PCA
    if (n >= 1 and n <= 16):
        X_test = pca.transform(X_test)
    #print "visit"
    #print X_test
    if method == 0:
        raise data.ValidationError('please select classification method')
    #SVM
    elif method == 2:
        if kernal == 0:
            raise data.ValidationError('please select kernel')
        clf = SVM.train(X_train,Y_train,kernal)
        result = SVM.test(X_test,clf)
        #print X_train
        #print X_test
        #print result
        #print clf
    #KNN
    elif method == 1:
        if k == '':
            raise data.ValidationError('please select number of neighbors (k)')
        neigh = KNN.train(X_train,Y_train,k)
        result = KNN.test(X_test,neigh)

    return result[0], Y_test[0]


class Example(tk.Frame):

    soundfile = ''
    #player = pyglet.media.Player()


    def __init__(self, parent):

        dirAlz = ''
        dirCtr = ''

        classification = tk.IntVar()
        kernel = tk.IntVar()
        PCA = tk.IntVar()
        sound2 = tk.StringVar()

        labelText = tk.StringVar()
        labelText2 = tk.StringVar()

        tk.Frame.__init__(self, parent)

        self.master.title("CogID")



        def onselectAlz(evt):
            w = evt.widget
            index = self.dataAlz.curselection()
            sound2.set(w.get(index))
            #tk.Entry(self, textvariable = sound2).grid(row = 7, column = 1, columnspan = 4, sticky = tk.W)
            #filename = self.dataAlz.get(self.dataAlz.curselection())[0:3]
            #print sound2

        def onselectCtr(evt):
            w = evt.widget
            index = self.dataCtr.curselection()
            sound2.set(w.get(index))
            #tk.Entry(self, textvariable = sound2).grid(row = 7, column = 1, columnspan = 4, sticky = tk.W)
            #filename = self.dataAlz.get(self.dataAlz.curselection())[0:3]
            #print sound2

        def classification_method():
            class_method = classification.get()
            k = kbox.get()
            ker = kernel.get()
            #print classification.get()
            #print kbox.get()
            #print ker
            if PCA.get() == 1:
                n = int(PCAbox.get())
            else:
                n = -1
            filename = filebox.get()
            #print filename
            try:
                result, truth = classify_function(filename, class_method, k, ker, n)
            except data.ValidationError as e:
                tkMessageBox.showinfo("Error Message",e.message)
                return
            #print result,truth
            if result == 0:
                labelText.set('control')
            elif result == 1:
                labelText.set('probably Alzheimers')
            if result == truth:
                labelText2.set('correct')
            elif result != truth:
                labelText2.set('incorrect')


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

        kbox = tk.Entry(self, width = 3)
        kbox.grid(row = 4, column = 3, columnspan = 1, sticky = tk.W)

        tk.Radiobutton(self, text = "Linear", variable = kernel, value = 1).grid(row = 5, column = 3, sticky = tk.W)
        tk.Radiobutton(self, text = "RBF", variable = kernel, value = 2).grid(row = 5, column = 4, sticky = tk.W)

        #tk.Label(self, text = "PCA", borderwidth = 1).grid(row = 6, column = 1, sticky = tk.W)
        PCAbox = tk.Entry(self, width = 3)
        PCAbox.configure(state = 'disabled', readonlybackground= 'black')
        def naccheck():
            if PCA.get() == 1:
                PCAbox.configure(state='normal', background= 'white')
            else:
                PCAbox.configure(state='disabled', readonlybackground= 'black')
        On = tk.Checkbutton(self, variable = PCA, text = "PCA", onvalue = 1, offvalue = 0, command = lambda: naccheck())
        On.grid(row = 6, column = 1, sticky = tk.W)

        PCAbox.grid(row = 6, column = 3, sticky = tk.W)

        tk.Label(self, text = "Selected File", borderwidth = 1).grid(row = 7, column = 1, columnspan = 4, sticky = tk.W)

        filebox = tk.Entry(self, textvariable = sound2)
        filebox.grid(row = 8, column = 1, columnspan = 4, sticky = tk.W)

        tk.Button(self, text = "Listen", borderwidth = 1, command = self.listenAlz).grid(row = 9, column = 1, columnspan = 2, sticky = tk.W)
        tk.Button(self, text = "Classify", borderwidth = 1, command = classification_method).grid(row = 9, column = 3, columnspan = 2, sticky = tk.W)

        tk.Label(self, text = "Predicted Label").grid(row = 10, column = 1, columnspan = 4, sticky = tk.W)
        resultLabel = tk.Label(self, textvariable = labelText).grid(row = 11, column = 1, columnspan = 4, sticky = tk.W)

        correctLabel = tk.Label(self, textvariable = labelText2).grid(row = 12, column = 1, columnspan = 4, sticky = tk.W)

        self.dataAlz = tk.Listbox(self, selectmode = "Single")
        self.dataAlz.bind('<<ListboxSelect>>',onselectAlz)
        self.dataAlz.grid(row = 2, column = 5, rowspan = 8)
        self.dataCtr = tk.Listbox(self, selectmode = "Single")
        self.dataCtr.bind('<<ListboxSelect>>',onselectCtr)
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


    def listenAlz(self):
        soundfile = self.dataAlz.get(self.dataAlz.curselection())
        filename = soundfile[0:3]
        global listen_thread
        listen_thread = Thread(target=startPlaying)
        listen_thread.start()

    def listenCtr(self):
        soundfile = self.dataCtr.get(self.dataAlz.curselection())
        filename = soundfile[0:3]
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
