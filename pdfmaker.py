#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 17:43:52 2020

@author: luisaweiss
"""

from fpdf import FPDF
#from PIL import image
import import_settings

   
title = 'Your Microgrid Development Plan'
title1 = 'WECKro-Grid 20 Report'

class PDF(FPDF):
   
    def header(self):
        # Logo
        image = 'logo1.jpeg'
        pdf.image(image, 10, 8, 33, 8)
        
        
        #self.image('foo.png', 20, 20, 50)
        # Arial bold 15
        self.set_font('Arial', 'B', 15)
        w = self.get_string_width(title) + 6
        self.set_x((210-w)/2)
        self.set_draw_color(44, 116, 199)
        self.set_fill_color(152, 221, 250)
        self.set_text_color(199, 72, 44)
        self.set_line_width(1)
        # Move to the right
        self.cell(w, 9, title, 1, 1, 'C', 1)
        self.ln(3)
        pdf.cell(65, 10)
        self.cell(11, 9, title1,'C')
     

        # Line break
        self.ln(80)
        self.set_line_width(0.0)
        self.line(5.0,5.0,205.0,5.0) # top one
        self.line(5.0,292.0,205.0,292.0) # bottom one
        self.line(5.0,5.0,5.0,292.0) # left one
        self.line(205.0,5.0,205.0,292.0) # right one
  
        
        
    def chapter_body(self):
        # Read text file
        
        # Times 12
        self.set_font('Times', '', 12)
        w= 160
        h=100
        
        # Output justified text
        self.multi_cell(100, 50, 'hi')
        # Line break
        
        # Mention in italics
        self.set_font('', 'I')
        self.cell(0, 5, '(end of excerpt)')
       
   
        

    # Page footer
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-15)
        # Arial italic 8
        self.set_font('Arial', 'I', 8)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')
     
# Instantiation of inherited class
pdf = PDF()


pdf.alias_nb_pages()
pdf.add_page()
pdf.set_font('Times', '', 12)
pdf.cell(50, 10)
y= 110
pdf.cell(0,y,'The Optimal Actions to take over a 20 year period', 'C')
image1 = 'plot.png'
pdf.image(image1, x=23, y=50, w=w, h=h)
pdf.ln(90)
pdf.cell(60, 80)
pdf.set_fill_color(152, 221, 250)
pdf.cell(70, h = 15, txt = 'Click to go to our website for more info', border = 1, ln = 2, align = 'C', fill = True, link = 'http://127.0.0.1:8000/ ')

pdf.add_page()
image2 = '/Users/luisaweiss/Downloads/Data/table1.png'
pdf.image(image2, x=23, y= 30, w=w, h=h)
pdf.cell(25, 10)
y= 70
pdf.cell(0,y,'The Optimal Actions to take over a 20 year period, including Investment Costs', 'C')


pdf.add_page()
image3 = '/Users/luisaweiss/Downloads/Data/table2.png'
pdf.image(image3, x=23, y=30, w=w, h=h)
pdf.cell(50, 10)
y= 70
pdf.cell(0,y,'Storage Units List and Investment Costs', 'C')


pdf.add_page()
image4 = '/Users/luisaweiss/Downloads/Data/table3.png'
pdf.image(image4, x=23, y= 30, w=w, h=h)
pdf.cell(50, 10)
y= 70
pdf.cell(0,y,'Power Plants List and Investment Costs', 'C')
pdf.output('results.pdf', 'F')
