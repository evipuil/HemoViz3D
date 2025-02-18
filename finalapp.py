from flask import Flask, render_template, request, url_for, flash, redirect, send_from_directory
import json
import math
from moviepy.editor import *
from mpl_toolkits import mplot3d
import numpy as np
import os
import pandas
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn import preprocessing
from stl import mesh

# Definition of Web Application and Global Variables
app = Flask(__name__) # defines web application
app.config['SECRET_KEY']='acad04e0262e72720856b95f9b9e74a2990314e0009e8d36' # creates secret key
messages = [] # for image gen
frames = [] # for video gen
videostl='' # name of stl file for video
vectortype='dots' # type of video
videotype='' # 2d vs 3d video

# Arguments Needed for 3D Video
def frame_args(duration):
    return {
            "frame": {"duration": duration},
            "mode": "immediate",
            "fromcurrent": True,
            "transition": {"duration": duration, "easing": "linear"},
        }

# Home Page
@app.route('/')
def index():
    return render_template('index.html', messages=messages)

# Preset One
@app.route('/preset_one/')
def preset_one():
    file="time-2300"
    stl="branchingartery.stl"
    values=np.loadtxt(file, dtype=object, delimiter=',') # reading file
    values=values[1:]
    templst=[]
    lst=[]

    for i in range(len(values)):
        templst.extend(values[i].split(' '))
    for j in range(len(templst)):
        if templst[j]!='':
            lst.append(float(templst[j]))

    # Defining lists
    x=[]
    y=[]
    z=[]
    u=[]
    v=[]
    w=[]
    m=[]

    skip=10*11 # every ___ point, 1st number is skip, 2nd number is columns

    # Entering Data Into the List
    for k in range(len(lst)):
        if k%skip==1:
            x.append(lst[k]*1000) # x 1000 to calibrate
        elif k%skip==2:
            y.append(lst[k]*1000) # x 1000 to calibrate
        elif k%skip==3:
            z.append(lst[k]*1000) # x 1000 to calibrate
        elif k%skip==5:
            m.append(lst[k])
        elif k%skip==6:
            u.append(lst[k])
        elif k%skip==7:
            v.append(lst[k])
        elif k%skip==8:
            w.append(lst[k])

    m = [(math.sqrt(u[q]**2 + v[q]**2 + w[q]**2)) for q in range(len(u))] # recalculating m
    
    # Converting to Arrays
    x_array = np.array(x)
    y_array = np.array(y)
    z_array = np.array(z)
    u_array = np.array(u)
    v_array = np.array(v)
    w_array = np.array(w)
    m_array = np.array(m)

    # Define the Plot
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "box"}, {"type": "scatter3d"}]])

    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(stl)

    # Extract the unique vertices and the face indices
    vertices, ixr = np.unique(stl_mesh.vectors.reshape(-1, 3), return_inverse=True, axis=0)
    I = np.take(ixr, [3*k for k in range(len(stl_mesh))])
    J = np.take(ixr, [3*k+1 for k in range(len(stl_mesh))])
    K = np.take(ixr, [3*k+2 for k in range(len(stl_mesh))])

    # Adding Traces (parts) to Figure
    fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=I, j=J, k=K, color='gray', opacity=0.1, flatshading=True), row=1, col=2)
    fig.add_trace(go.Cone(x=x_array, y=y_array, z=z_array, u=u_array, v=v_array, w=w_array, sizemode='absolute', sizeref=2, colorscale='Rainbow', colorbar=dict(title="Relative VM")), row=1, col=2)
    fig.add_trace(go.Box(y=m_array, name="Relative VM"), row=1, col=1)

    # Configuration of the Plot
    fig.update_layout(
        xaxis=dict(
            showline=False,
            scaleratio=1,
            ticks='outside',
            tickson='boundaries',
            ticklen=0,
            tickwidth=0,
            tickcolor='#000'
        ),
        yaxis=dict(
            showline=False,
            scaleratio=1,
            ticks='outside',
            tickson='boundaries',
            ticklen=0,
            tickwidth=0,
            tickcolor='#000'
        ),
        yaxis2=dict(
            showline=False,
            scaleratio=1,
            ticks='outside',
            tickson='boundaries',
            ticklen=0,
            tickwidth=0,
            tickcolor='#000'
        ),
        scene=dict(
            xaxis_title='',
            yaxis_title='',
            zaxis_title=''
        ),
        title=go.layout.Title(
            text="Preset 1"
        )
    )
    fig.update_scenes(
        xaxis_showaxeslabels=False, yaxis_showaxeslabels=False, zaxis_showaxeslabels=False,
        xaxis_showbackground=False, yaxis_showbackground=False, zaxis_showbackground=False,
        xaxis_showline=False, yaxis_showline=False, zaxis_showline=False,
        xaxis_nticks=False, yaxis_nticks=False, zaxis_nticks=False,
        xaxis_showticklabels=False, yaxis_showticklabels=False, zaxis_showticklabels=False,
        aspectmode='data'
    )

    # Export Plot as JSON
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Return Output Page
    return render_template('image_out.html', plot=graphJSON)

# Preset Two
@app.route('/preset_two/')
def preset_two():
    file="carotid_stenosis-6-03250-wss-3050"
    stl="branchingartery.stl"
    values=np.loadtxt(file, dtype=object, delimiter=',') # reading file
    values=values[1:]
    templst=[]
    lst=[]

    for i in range(len(values)):
        templst.extend(values[i].split(' '))
    for j in range(len(templst)):
        if templst[j]!='':
            lst.append(float(templst[j]))

    # Defining lists
    x=[]
    y=[]
    z=[]
    stress=[]

    skip=8 # every ___ point, 1st number is skip, 2nd number is columns

    # Entering Data Into the List
    for k in range(len(lst)):
        if k%skip==1:
            x.append(lst[k]*1000) # x 1000 to calibrate
        elif k%skip==2:
            y.append(lst[k]*1000) # x 1000 to calibrate
        elif k%skip==3:
            z.append(lst[k]*1000) # x 1000 to calibrate
        elif k%skip==4:
            stress.append(lst[k])
   
    stress=preprocessing.normalize([stress])

    # Define the Plot
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "box"}, {"type": "scatter3d"}]])

    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(stl)

    # Extract the unique vertices and the face indices
    vertices, ixr = np.unique(stl_mesh.vectors.reshape(-1, 3), return_inverse=True, axis=0)
    I = np.take(ixr, [3*k for k in range(len(stl_mesh))])
    J = np.take(ixr, [3*k+1 for k in range(len(stl_mesh))])
    K = np.take(ixr, [3*k+2 for k in range(len(stl_mesh))])

    # Adding Traces (parts) to Figure
    fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=I, j=J, k=K, color='gray', opacity=0.1, flatshading=True), row=1, col=2)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(color=stress[0], colorscale="rainbow", colorbar=dict(title="Relative WSS"), size=5)), row=1, col=2)
    fig.add_trace(go.Box(y=stress[0], name="Relative WSS"), row=1, col=1)

    # Configuration of the Plot
    fig.update_layout(
        xaxis=dict(
            showline=False,
            scaleratio=1,
            ticks='outside',
            tickson='boundaries',
            ticklen=0,
            tickwidth=0,
            tickcolor='#000'
        ),
        yaxis=dict(
            showline=False,
            scaleratio=1,
            ticks='outside',
            tickson='boundaries',
            ticklen=0,
            tickwidth=0,
            tickcolor='#000'
        ),
        yaxis2=dict(
            showline=False,
            scaleratio=1,
            ticks='outside',
            tickson='boundaries',
            ticklen=0,
            tickwidth=0,
            tickcolor='#000'
        ),
        scene=dict(
            xaxis_title='',
            yaxis_title='',
            zaxis_title=''
        ),
        title=go.layout.Title(
            text="Preset 2"
        )
    )
    fig.update_scenes(
        xaxis_showaxeslabels=False, yaxis_showaxeslabels=False, zaxis_showaxeslabels=False,
        xaxis_showbackground=False, yaxis_showbackground=False, zaxis_showbackground=False,
        xaxis_showline=False, yaxis_showline=False, zaxis_showline=False,
        xaxis_nticks=False, yaxis_nticks=False, zaxis_nticks=False,
        xaxis_showticklabels=False, yaxis_showticklabels=False, zaxis_showticklabels=False,
        aspectmode='data'
    )

    # Export Plot as JSON
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Return Output Page
    return render_template('image_out.html', plot=graphJSON)

# Image Input Page
@app.route('/image_in/', methods=('GET', 'POST'))
def image_in():
    if request.method == 'POST': # inputting file
        title = request.form['title'] # get title
        vectorfile = request.files['vectorfile'] # get vector file
        stlfile= request.files['stlfile'] # get stlfile
        vectortype=request.form['vectortype'] # get vector type

        columnnumber=request.form['columnnum'] # get number of columns
        xcol=request.form['xcol'] # get x column
        ycol=request.form['ycol'] # get y column
        zcol=request.form['zcol'] # get z column
        ucol=request.form['ucol'] # get u column
        vcol=request.form['vcol'] # get v column
        wcol=request.form['wcol'] # get w column
        mcol=request.form['mcol'] # get m column

        vectorfile.save(vectorfile.filename) # saving vector file
        stlfile.save(stlfile.filename) # saving STL file

        messages.append({'title': title, 
                         'vectorfile': vectorfile.filename, 
                         'stlfile': stlfile.filename, 
                         'vectortype': vectortype,
                         "columnnumber": columnnumber,
                         "xcol": xcol,
                         "ycol": ycol,
                         "zcol": zcol,
                         "ucol": ucol,
                         "vcol": vcol,
                         "wcol": wcol,
                         "mcol": mcol}) # saving file names
        
        return redirect(url_for('image_out')) # send output page

    return render_template('image_in.html') # send input page

# Image Output Page
@app.route('/image_out/')
def image_out():
    # Input Parameters
    title=messages[-1]['title']
    vectorfile=messages[-1]['vectorfile']
    stlfile=messages[-1]['stlfile']
    vectortype=messages[-1]['vectortype']
    columnnumber=messages[-1]['columnnumber'] # get number of columns
    xcol=int(messages[-1]['xcol']) # get x column
    ycol=int(messages[-1]['ycol']) # get y column
    zcol=int(messages[-1]['zcol']) # get z column
    ucol=int(messages[-1]['ucol']) # get u column
    vcol=int(messages[-1]['vcol']) # get v column
    wcol=int(messages[-1]['wcol']) # get w column
    mcol=int(messages[-1]['mcol']) # get m column

    if '.csv' not in vectorfile: # .txt/misc file format
        values=np.loadtxt(vectorfile, dtype=object, delimiter=',') # reading file
        values=values[1:]
        templst=[]
        lst=[]

        for i in range(len(values)):
            templst.extend(values[i].split(' '))
        for j in range(len(templst)):
            if templst[j]!='':
                lst.append(float(templst[j]))

        # Defining lists
        x=[]
        y=[]
        z=[]
        u=[]
        v=[]
        w=[]
        m=[]

        skip=10*int(columnnumber) # every ___ point, 1st number is skip, 2nd number is columns

        # Entering Data Into the List
        for k in range(len(lst)):
            if k%skip==xcol:
                x.append(lst[k]*1000) # x 1000 to calibrate
            elif k%skip==ycol:
                y.append(lst[k]*1000) # x 1000 to calibrate
            elif k%skip==zcol:
                z.append(lst[k]*1000) # x 1000 to calibrate
            elif k%skip==mcol:
                m.append(lst[k])
            elif k%skip==ucol:
                u.append(lst[k])
            elif k%skip==vcol:
                v.append(lst[k])
            elif k%skip==wcol:
                w.append(lst[k])

        if mcol==0:
            m = [(math.sqrt(u[q]**2 + v[q]**2 + w[q]**2)) for q in range(len(u))] # recalculating m

    else: # csv file format (with columns)
        values=pandas.read_csv(vectorfile)
        x=values['x']
        y=values['y']
        z=values['z']
        u=values['u']
        v=values['v']
        w=values['w']
        m = [(math.sqrt(u[o]**2 + v[o]**2 + w[o]**2)) for o in range(len(u))] # calculating m

    # Converting to Arrays
    x_array = np.array(x)
    y_array = np.array(y)
    z_array = np.array(z)
    u_array = np.array(u)
    v_array = np.array(v)
    w_array = np.array(w)
    m_array = np.array(m)

    # Define the Plot
    fig = go.Figure()

    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(stlfile)

    # Extract the unique vertices and the face indices
    vertices, ixr = np.unique(stl_mesh.vectors.reshape(-1, 3), return_inverse=True, axis=0)
    I = np.take(ixr, [3*k for k in range(len(stl_mesh))])
    J = np.take(ixr, [3*k+1 for k in range(len(stl_mesh))])
    K = np.take(ixr, [3*k+2 for k in range(len(stl_mesh))])

    # Adding Traces (parts) to Figure
    fig.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=I, j=J, k=K, color='gray', opacity=0.1, flatshading=True))
    if vectortype=='arrows':
        fig.add_trace(go.Cone(x=x_array, y=y_array, z=z_array, u=u_array, v=v_array, w=w_array, sizemode='absolute', sizeref=2, colorscale='Rainbow', colorbar=dict(title="Relative Magnitude")))
    else:
        fig.add_trace(go.Scatter3d(x=x_array, y=y_array, z=z_array, mode='markers', marker=dict(color=m_array, colorscale="rainbow", colorbar=dict(title="Relative Magnitude", len=1), size=2)))

    # Configuration of the Plot
    fig.update_layout(
        xaxis=dict(
            showline=False,
            scaleratio=1,
            ticks='outside',
            tickson='boundaries',
            ticklen=0,
            tickwidth=0,
            tickcolor='#000'
        ),
        yaxis=dict(
            showline=False,
            scaleratio=1,
            ticks='outside',
            tickson='boundaries',
            ticklen=0,
            tickwidth=0,
            tickcolor='#000'
        ),
        yaxis2=dict(
            showline=False,
            scaleratio=1,
            ticks='outside',
            tickson='boundaries',
            ticklen=0,
            tickwidth=0,
            tickcolor='#000'
        ),
        scene=dict(
            xaxis_title='',
            yaxis_title='',
            zaxis_title=''
        ),
        title=go.layout.Title(
            text=title
        )
    )
    fig.update_scenes(
        xaxis_showaxeslabels=False, yaxis_showaxeslabels=False, zaxis_showaxeslabels=False,
        xaxis_showbackground=False, yaxis_showbackground=False, zaxis_showbackground=False,
        xaxis_showline=False, yaxis_showline=False, zaxis_showline=False,
        xaxis_nticks=False, yaxis_nticks=False, zaxis_nticks=False,
        xaxis_showticklabels=False, yaxis_showticklabels=False, zaxis_showticklabels=False,
        aspectmode='data'
    )

    # Export Plot as JSON
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    # Return Output Page
    return render_template('image_out.html', plot=graphJSON)
    
# Video Input Page
@app.route('/video_in/', methods=['GET', 'POST'])
def video_in():
    global frames
    if request.method == 'POST': # inputting file
        stlfile= request.files['stlfile'] # get stl file
        vectortype=request.form['vectortype'] # get vector type
        files=request.files.getlist('vectorfiles') # get all vector files
        videotype=request.form['videotype']

        columnnumber=request.form['columnnum'] # get number of columns
        xcol=request.form['xcol'] # get x column
        ycol=request.form['ycol'] # get y column
        zcol=request.form['zcol'] # get z column
        ucol=request.form['ucol'] # get u column
        vcol=request.form['vcol'] # get v column
        wcol=request.form['wcol'] # get w column
        mcol=request.form['mcol'] # get m column

        for file in files: # save all timepoints
            file.save(file.filename)
            frames.append(file.filename)
            
        stlfile.save(stlfile.filename)
        videostl=stlfile.filename
        messages.append({"stlfile": stlfile.filename, 
                         "vectortype": vectortype, 
                         "videotype": videotype, 
                         "columnnumber": columnnumber, 
                         "xcol": xcol, 
                         "ycol": ycol, 
                         "zcol": zcol, 
                         "ucol": ucol, 
                         "vcol": vcol, 
                         "wcol": wcol, 
                         "mcol": mcol})

        if videotype=='2D':
            return redirect(url_for('video_out')) # return 2d video
        else:
            return redirect(url_for('video_3d_out')) # return 3d video

    return render_template('video_in.html') # return input page

# Video Output Page
@app.route('/video_out/')
def video_out():
    global frames
    stlfile=messages[-1]['stlfile']
    vectortype=messages[-1]['vectortype']
    columnnumber=int(messages[-1]['columnnumber']) # get number of columns
    xcol=int(messages[-1]['xcol']) # get x column
    ycol=int(messages[-1]['ycol']) # get y column
    zcol=int(messages[-1]['zcol']) # get z column
    ucol=int(messages[-1]['ucol']) # get u column
    vcol=int(messages[-1]['vcol']) # get v column
    wcol=int(messages[-1]['wcol']) # get w column
    mcol=int(messages[-1]['mcol']) # get m column

    cmin=1000000000000000
    cmax=0
    for a in range(len(frames)):
        currentm=[]
        if mcol!=0:
            for b in range(len(frames[a])):
                if b % int(columnnumber) == mcol:
                    currentm.append(float(frames[a][b]))
                    if frames[a][b] > cmax:
                        cmax=frames[a][b]
                    if frames[a][b] < cmin:
                        cmin=frames[a][b]
        else:
            for b in range(len(frames[b])):
                if b % columnnumber == ucol:
                    currentm.append(math.sqrt(float(frames[a][b])**2 + float(frames[a][b+1])**2 + float(frames[a][b+2])**2))
                    if math.sqrt(float(frames[a][b])**2 + float(frames[a][b+1])**2 + float(frames[a][b+2])**2) > cmax:
                        cmax=math.sqrt(float(frames[a][b])**2 + float(frames[a][b+1])**2 + float(frames[a][b+2])**2)
                    if math.sqrt(float(frames[a][b])**2 + float(frames[a][b+1])**2 + float(frames[a][b+2])**2) < cmin:
                        cmin=math.sqrt(float(frames[a][b])**2 + float(frames[a][b+1])**2 + float(frames[a][b+2])**2)
   
    for frame in frames: #generating each frame
        values=np.loadtxt(frame, dtype=object, delimiter=',') #loading file
        values=values[1:] #removing title row
        templst=[]
        lst=[]
        for l in range(len(values)):
            templst.extend(values[l].split(' '))
        for j in range(len(templst)):
            if templst[j]!='':
                lst.append(float(templst[j]))

        # Defining lists
        x=[]
        y=[]
        z=[]
        u=[]
        v=[]
        w=[]
        m=[]

        skip=10*int(columnnumber) # every ___ point, 1st number is skip, 2nd number is columns
        # Separating Data
        for k in range(len(lst)):
            if k%skip==xcol:
                x.append(lst[k]*1000)
            elif k%skip==ycol:
                y.append(lst[k]*1000)
            elif k%skip==zcol:
                z.append(lst[k]*1000)
            elif k%skip==mcol:
                m.append(lst[k])
            elif k%skip==ucol:
                u.append(lst[k])
            elif k%skip==vcol:
                v.append(lst[k])
            elif k%skip==wcol:
                w.append(lst[k])

        # Load the STL file
        stl_mesh = mesh.Mesh.from_file(stlfile)

        # Extract the unique vertices and the face indices
        vertices, ixr = np.unique(stl_mesh.vectors.reshape(-1, 3), return_inverse=True, axis=0)
        I = np.take(ixr, [3*k for k in range(len(stl_mesh))])
        J = np.take(ixr, [3*k+1 for k in range(len(stl_mesh))])
        K = np.take(ixr, [3*k+2 for k in range(len(stl_mesh))])

        # Defining Figure and Configuring Parameters
        fig=go.Figure(
            data=[go.Cone(x=x, y=y, z=z, u=u, v=v, w=w, sizeref=2, sizemode="scaled", cmin=cmin, cmax=cmax, colorscale="Rainbow"),
                  go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=I, j=J, k=K, color='gray', opacity=0.1, flatshading=True)]
        )
        fig.update_layout(
            xaxis=dict(
                showline=False,
                scaleratio=1,
                ticks='outside',
                tickson='boundaries',
                ticklen=0,
                tickwidth=0,
                tickcolor='#000'
            ),
            yaxis=dict(
                showline=False,
                scaleratio=1,
                ticks='outside',
                tickson='boundaries',
                ticklen=0,
                tickwidth=0,
                tickcolor='#000'
            ),
            yaxis2=dict(
                showline=False,
                scaleratio=1,
                ticks='outside',
                tickson='boundaries',
                ticklen=0,
                tickwidth=0,
                tickcolor='#000'
            ),
            scene=dict(
                xaxis_title='',
                yaxis_title='',
                zaxis_title=''
            ),
            title=go.layout.Title(
                text="Preset 1"
            )
        )

        fig.update_scenes(
            xaxis_showaxeslabels=False, yaxis_showaxeslabels=False, zaxis_showaxeslabels=False,
            xaxis_showbackground=False, yaxis_showbackground=False, zaxis_showbackground=False,
            xaxis_showline=False, yaxis_showline=False, zaxis_showline=False,
            xaxis_nticks=False, yaxis_nticks=False, zaxis_nticks=False,
            xaxis_showticklabels=False, yaxis_showticklabels=False, zaxis_showticklabels=False,
            aspectmode='data'
        )
        fig.write_image("image"+str(i)+".png") # writing image
        i+=1 # next file

    image_files = [f for f in os.listdir('.') if f.endswith('.png')] # every image in directory
    clips = [ImageClip(f).set_duration(0.3) for f in image_files] # making clips
    video = concatenate_videoclips(clips) # concatenating clips
    video.write_videofile('output.mp4', fps=12) # writing video file
    for image in image_files: # removing images
        os.remove(image)
    for frame in frames: # removing files
        os.remove(frame)
    frames=[]

    return send_from_directory('.', 'output.mp4') #return video

# 3D Video Page
@app.route("/video_3d_out/")
def video_3d_out():
    files=frames
    stl=videostl
    dictionary={
        "x":[],
        "y":[],
        "z":[],
        "u":[],
        "v":[],
        "w":[]
    }

    for a in range(len(files)):
        values=np.loadtxt(files[a], dtype=object, delimiter=',') # reading file
        values=values[1:]
        templst=[]
        lst=[]

        for i in range(len(values)):
            templst.extend(values[i].split(' '))
        for j in range(len(templst)):
            if templst[j]!='':
                lst.append(float(templst[j]))

        # Defining lists
        x=[]
        y=[]
        z=[]
        u=[]
        v=[]
        w=[]
        m=[]

        skip=10*12 # every ___ point, 1st number is skip, 2nd number is columns

        # Entering Data Into the List
        for k in range(len(lst)):
            if k%skip==1:
                x.append(lst[k]*1000) # x 1000 to calibrate
            elif k%skip==2:
                y.append(lst[k]*1000) # x 1000 to calibrate
            elif k%skip==3:
                z.append(lst[k]*1000) # x 1000 to calibrate
            elif k%skip==5:
                m.append(lst[k])
            elif k%skip==6:
                u.append(lst[k])
            elif k%skip==7:
                v.append(lst[k])          
            elif k%skip==8:
                w.append(lst[k])

        dictionary["x"].append(x)
        dictionary["y"].append(y)
        dictionary["z"].append(z)
        dictionary["u"].append(u)
        dictionary["v"].append(v)
        dictionary["w"].append(w)

    # Load the STL file
    stl_mesh = mesh.Mesh.from_file(stl)

    # Extract the unique vertices and the face indices
    vertices, ixr = np.unique(stl_mesh.vectors.reshape(-1, 3), return_inverse=True, axis=0)
    I = np.take(ixr, [3*k for k in range(len(stl_mesh))])
    J = np.take(ixr, [3*k+1 for k in range(len(stl_mesh))])
    K = np.take(ixr, [3*k+2 for k in range(len(stl_mesh))])

    # Multiple Frames in Figure
    fig = go.Figure(
        data=[go.Cone(x=dictionary["x"][0], y=dictionary["y"][0], z=dictionary["z"][0], u=dictionary["u"][0], v=dictionary["v"][0], w=dictionary["w"][0], sizeref=2.5, sizemode="scaled", colorscale="Rainbow"), go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=I, j=J, k=K, color='gray', opacity=0.1, flatshading=True)],
        layout=go.Layout(updatemenus=[dict(type="buttons", buttons=[dict(label="Play", method="animate", args=[None])])]),
        frames=[go.Frame(data=[go.Cone(x=dictionary["x"][i], y=dictionary["y"][i], z=dictionary["z"][i], u=dictionary["u"][i], v=dictionary["v"][i], w=dictionary["w"][i], sizeref=2.5, sizemode="scaled", colorscale="Rainbow"), go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2], i=I, j=J, k=K, color='gray', opacity=0.1, flatshading=True)], name=str(i)) for i in range(1, len(dictionary["x"]))])

    #Configuration
    fig.update_layout(
        xaxis=dict(
            showline=False,
            scaleratio=1,
            ticks='outside',
            tickson='boundaries',
            ticklen=0,
            tickwidth=0,
            tickcolor='#000'
        ),
        yaxis=dict(
            showline=False,
            scaleratio=1,
            ticks='outside',
            tickson='boundaries',
            ticklen=0,
            tickwidth=0,
            tickcolor='#000'
        ),
        yaxis2=dict(
            showline=False,
            scaleratio=1,
            ticks='outside',
            tickson='boundaries',
            ticklen=0,
            tickwidth=0,
            tickcolor='#000'
        ),
        scene=dict(
            xaxis_title='',
            yaxis_title='',
            zaxis_title=''
        ),
        title=go.layout.Title(
            text="Preset 1"
        )
    )
    fig.update_scenes(
        xaxis_showaxeslabels=False, yaxis_showaxeslabels=False, zaxis_showaxeslabels=False,
        xaxis_showbackground=False, yaxis_showbackground=False, zaxis_showbackground=False,
        xaxis_showline=False, yaxis_showline=False, zaxis_showline=False,
        xaxis_nticks=False, yaxis_nticks=False, zaxis_nticks=False,
        xaxis_showticklabels=False, yaxis_showticklabels=False, zaxis_showticklabels=False,
        aspectmode='data'
    )
    sliders = [
        {
            "pad": {"b": 10, "t": 60},
            "len": 0.9,
            "x": 0.1,
            "y": 0,
            "steps": [
                {
                    "args": [[f.name], frame_args(0)],
                    "label": str(k),
                    "method": "animate",
                }
                for k, f in enumerate(fig.frames)
            ],
        }
    ]
    fig.update_layout(sliders=sliders)

    # Export Plot as JSON
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    frames = []
    # Return Output Page
    return render_template('image_out.html', plot=graphJSON)

@app.route('/WSS_in/', methods=['GET', 'POST']) #wall shear stress input page
def WSS_in():
    # Same as image_in, but for WSS
    if request.method == 'POST':
        title = request.form['title']
        vectorfile = request.files['vectorfile']
        stlfile= request.files['stlfile']

        vectorfile.save(vectorfile.filename)
        stlfile.save(stlfile.filename)

        messages.append({'title': title, 
                         'vectorfile': vectorfile.filename, 
                         'stlfile': stlfile.filename})
        
        return redirect(url_for('WSS_out'))

    return render_template('WSS_in.html')

@app.route('/WSS_out')
def WSS_out():
    # Same As Image Input, but with WSS
    global messages
    file=messages[-1]
    title=file['title']
    vectorfile=file['vectorfile']
    values=np.loadtxt(vectorfile, dtype=object, delimiter=',')
    values=values[1:]
    templst=[]
    lst=[]

    for a in range(len(values)):
        value=str(values[a])
        templst.extend(value.split(' '))
    for b in range(len(templst)):
        if templst[b] != '':
            lst.append(float(templst[b]))

    x=[]
    y=[]
    z=[]
    stress=[]

    for c in range(0, len(lst)):
        if c%5==1:
            x.append(lst[c])
        elif c%5==2:
            y.append(lst[c])
        elif c%5==3:
            z.append(lst[c])
        elif c%5==4:
            stress.append(lst[c])

    x_array=np.array(x)
    y_array=np.array(y)
    z_array=np.array(z)
    stress_array=np.array(stress)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=dict(color=stress_array, colorscale="rainbow", colorbar=dict(len=1), size=5)))

    fig.update_layout(
        xaxis=dict(
            showline=False,
            scaleratio=1,
            ticks='outside',
            tickson='boundaries',
            ticklen=0,
            tickwidth=0,
            tickcolor='#000'
        ),
        yaxis=dict(
            showline=False,
            scaleratio=1,
            ticks='outside',
            tickson='boundaries',
            ticklen=0,
            tickwidth=0,
            tickcolor='#000'
        ),
        yaxis2=dict(
            showline=False,
            scaleratio=1,
            ticks='outside',
            tickson='boundaries',
            ticklen=0,
            tickwidth=0,
            tickcolor='#000'
        ),
        scene=dict(
            xaxis_title="",
            yaxis_title="",
            zaxis_title=""
        ),
        title=go.layout.Title(
            text=title
        )
    )

    fig.update_scenes(
        xaxis_showaxeslabels=False, yaxis_showaxeslabels=False, zaxis_showaxeslabels=False,
        xaxis_showbackground=False, yaxis_showbackground=False, zaxis_showbackground=False,
        xaxis_showline=False, yaxis_showline=False, zaxis_showline=False,
        xaxis_nticks=False, yaxis_nticks=False, zaxis_nticks=False,
        xaxis_showticklabels=False, yaxis_showticklabels=False, zaxis_showticklabels=False,
        aspectmode="data"
    )

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template("WSS_out.html", plot=graphJSON)