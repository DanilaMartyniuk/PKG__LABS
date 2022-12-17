import numpy as np
import cv2 as cv
from tkinter import ttk
from tkinter import *
import mahotas
import mahotas.demos
from PIL import ImageTk, Image


class MainSolution():
    def __init__(self):
        self.image = cv.imread("img.jpg", cv.IMREAD_GRAYSCALE)
        self.imgray = cv.imread('img.jpg', cv.IMREAD_GRAYSCALE)

    def log(self):
        kernel = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0],
                          [-1, -2, 16, -2, -1], [0, -1, -2, -1, 0], [0, 0, -1, 0, 0]])
        log = cv.filter2D(self.imgray, -1, kernel)
        log = Image.fromarray(log)
        img = log.resize((300, 300))
        return ImageTk.PhotoImage(img)

    def laplasian(self):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        laplasian = cv.filter2D(self.imgray, -1, kernel)
        laplasian = Image.fromarray(laplasian)
        img = laplasian.resize((300, 300))
        return ImageTk.PhotoImage(img)

    def adaptive_threshold(self):
        th3 = cv.adaptiveThreshold(
            self.image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
        img = Image.fromarray(th3)
        img = img.resize((300, 300))
        return ImageTk.PhotoImage(img)

    def bernsenthresholding(self):
        img = self.image
        img = mahotas.thresholding.bernsen(img, 5, 15)
        img = Image.fromarray(img)
        img = img.resize((300, 300))
        return ImageTk.PhotoImage(img)

    def niblackthreshholding(self):
        img = cv.imread('img.jpg', cv.IMREAD_GRAYSCALE)
        img = cv.ximgproc.niBlackThreshold(
            img, maxValue=255, type=cv.THRESH_BINARY,  blockSize=15, k=-0.2)
        img = Image.fromarray(img)
        img = img.resize((300, 300))
        return ImageTk.PhotoImage(img)


if __name__ == "__main__":
    root = Tk()
    ms = MainSolution()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry(f"1040x700")
    lbl_text1 = ttk.Label(text="Высокочастотные фильтры")
    lbl_text1.place(x=610, y=10)
    img1 = ms.log()
    lbl1 = ttk.Label(image=img1)
    lbl1.image = img1
    lbl1.place(x=710, y=40, width=300, height=300)
    img2 = ms.laplasian()
    lbl2 = ttk.Label(image=img2)
    lbl2.image = img2
    lbl2.place(x=370, y=40, width=300, height=300)
    lbl_text2 = ttk.Label(text="Адаптивная пороговая обработка")
    lbl_text2.place(x=90, y=360)
    img3 = ms.adaptive_threshold()
    lbl3 = ttk.Label(image=img3)
    lbl3.image = img3
    lbl3.place(x=30, y=390, width=300, height=300)
    lbl_text3 = ttk.Label(text="Локальная пороговая обработка")
    lbl_text3.place(x=580, y=360)
    img4 = ms.niblackthreshholding()
    lbl4 = ttk.Label(image=img4)
    lbl4.image = img4
    lbl4.place(x=370, y=390, width=300, height=300)
    img5 = ms.bernsenthresholding()
    lbl5 = ttk.Label(image=img5)
    lbl5.image = img5
    lbl5.place(x=710, y=390, width=300, height=300)
    lbl_text6 = ttk.Label(text="Оригинал")
    lbl_text6.place(x=150, y=10)
    img6 = Image.open('img.jpg')
    img6 = ImageTk.PhotoImage(img6.resize((300, 300), Image.ANTIALIAS))
    lbl6 = ttk.Label(image=img6)
    lbl6.image = img6
    lbl6.place(x=30, y=40, width=300, height=300)
    root.mainloop()
