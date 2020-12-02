#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 22:05:33 2020

@author: luisaweiss
"""
import os
from flask import Flask, request,redirect, jsonify, json
from flask import render_template
from flask import g
app = Flask(__name__)
#import import_settings
#import import_data
#import forecast_demand
#import outage_simulation
#import reinforcement_learning
#api = Api


@app.route('/')
def index():
    
    return render_template('index.html')


@app.route('/test', methods=['GET','POST'])
def example():
    
    person = list(request.form.values())[0]

    email = list(request.form.values())[1]
    
    p_file = open("person.json", "w")
    json.dump(person, p_file)
    json.dump(email, p_file)
    p_file.close()
    #requestJson = request.get_json(force=True)
    
    return render_template('page2.html')

@app.route('/page3.html')
def page3():
    
   return render_template('page3.html')

@app.route('/result',methods=['GET','POST'])
def results():
    
    data = request.json

   
    out_file = open("myfile.json", "w")
    json.dump(data, out_file)
    out_file.close()
    import data_load
    
   
    return render_template('index.html')
    
#@app.route('/run', methods=['GET','POST'])
#def run_script():
#    if request.method == 'POST':
#        file = open(r'/Data/import_settings.py', 'r').read()
#        facilities_list = out_file
#        import_settings.main()
#        import_data.main()
#        forecast_demand.main()
#    #requestJson = request.get_json(force=True)
    
 #       return exec(file)
    
@app.route('/howto.html')
def howto():
    
    return render_template('howto.html')    
@app.route('/about.html')
def about():
    
    return render_template('about.html')    

if __name__ == '__main__':
    app.run(debug=True, port=8000)
 

    