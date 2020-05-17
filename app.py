#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, jsonify, request
import joblib
import numpy as np


# In[2]:


app = Flask(__name__)


# In[ ]:


@app.route('/submit-data', methods=['POST'])
def getdata():
           data = request.get_json()
           x = np.array([[data['1'], data['2']]])
           model = joblib.load('model')
           prediction = model.predict(x)
           result = {
               prediction
              }
           return jsonify(result)

app.run(port = 5000, debug = 'development')

