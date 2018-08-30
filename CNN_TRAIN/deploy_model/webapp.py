import sys
import os
import flask
from flask import render_template, send_from_directory, request, redirect,url_for
from werkzeug import secure_filename
from flask import jsonify
import base64
import StringIO
import tensorflow as tf 
import numpy as np
import cv2
from scipy.misc import imread, imresize

# Obtain the flask app object
app = flask.Flask(__name__)

UPLOAD_FOLDER='static'
def load_graph(trained_model):   
    with tf.gfile.GFile(trained_model, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name=""
            )
    return graph

@app.route('/')
def index():
    return "Webserver is running"

@app.route('/demo',methods=['POST','GET'])
def demo():
    if request.method == 'POST':
        upload_file = request.files['file']
        filename = secure_filename(upload_file.filename)
        upload_file.save(os.path.join(UPLOAD_FOLDER, filename))
        image_size=128
        num_channels=3
        images = []
        # Reading the image using OpenCV
        # image = cv2.imread(os.path.join(UPLOAD_FOLDER, filename))
        currimg = cv2.imread(os.path.join(UPLOAD_FOLDER, filename))
        currimg = currimg.astype('float32')
        grayimg  = currimg
        graysmall = imresize(grayimg, [64, 64])/255.
        grayvec   = np.reshape(graysmall, (1, -1))
        totalimg   = grayvec
        x_batch = totalimg
        graph =app.graph
        y_pred = graph.get_tensor_by_name("modelOutput:0")
        ## Let's feed the images to the input placeholders
        x= graph.get_tensor_by_name("Placeholder:0")
        b_ = graph.get_tensor_by_name("Placeholder_2:0")
        y_test_images = np.zeros((1, 2))
        sess= tf.Session(graph=graph)
        ### Creating the feed_dict that is required to be fed to calculate y_pred 
        feed_dict_testing = {x: x_batch, b_: np.random.rand(1,128)}
        result=sess.run(y_pred, feed_dict=feed_dict_testing)
        print(result)
        ## 61326_0k_back
        out = {"61326_Scratch_Mark":str(result[0][0]), "61326_Slot_Damage":str(result[0][1]), "61326_Thinning":str(result[0][2]), "61326_Wrinkle":str(result[0][3]), "61326_0k_back":str(result[0][4]), "61326_ok_front":str(result[0][5])}
        return jsonify(out)
        #return redirect(url_for('just_upload',pic=filename))
    return  '''
    <!doctype html>
    <html lang="en">
    <head>
      <title>Flask deployment of tensorflow model</title>
    </head>
    <body>
    <div class="site-wrapper">
        <div class="cover-container">
            <nav id="main">
                <a href="http://localhost:5000/demo" >HOME</a>
            </nav>
          <div class="inner cover">

          </div>
          <div class="mastfoot">
          <hr />
            <div class="container">
              <div style="margin-top:5%">
		            <h1 style="color:black">CLASSIFICTAION DEMO</h1>
		            <h4 style="color:black">Upload new Image </h4>
		            <form method=post enctype=multipart/form-data>
	                 <p><input type=file name=file>
        	        <input type=submit style="color:black;" value=Upload>
		            </form>
	            </div>	
            </div>
        	</div>
     </div>
   </div>
</body>
</html>

    '''




app.graph=load_graph('frozen_convnetTFon.pb')  
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int("5000"), debug=True, use_reloader=False)
    