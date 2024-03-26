#====================================
# 
# Code written by Gaurav Mitra and team members
# used here with permission
#
#====================================
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist
import numpy as np
import gsd.hoomd
import argparse
import os
import sys


def distance(r,box_size):
    r=r-box_size*np.floor(r/box_size+0.5)
    return(np.sqrt(r[0]**2+r[1]**2+r[2]**2))

def split_into_clusters(link_mat,thresh,n):
   c_ts=n
   clusters={}
   for row in link_mat:
      if row[2] < thresh:
          n_1=int(row[0])
          n_2=int(row[1])

          if n_1 >= n:
             link_1=clusters[n_1]
             del(clusters[n_1])
          else:
             link_1= [n_1]

          if n_2 >= n:
             link_2=clusters[n_2]
             del(clusters[n_2])
          else:
             link_2= [n_2]

          link_1.extend(link_2)
          clusters[c_ts] = link_1
          c_ts+=1
      else:
          return clusters

def analyze_cluster(filename, cutoff, outputprefix):

    traj = gsd.hoomd.open(filename, "r")
    numsnap = len(traj)
    gsdfile=os.path.basename(filename)

    minframe=0
    maxframe=numsnap-1
    frameinterval=1

    print('minframe used:',minframe)
    print('maxframe used:',maxframe)
    print('frameinterval used:',frameinterval)
    frames_list=np.arange(minframe,maxframe+1,frameinterval) # list of frames to analyze
    print("No of frames used for analysis:",len(frames_list))

    box=traj[0].configuration.box[:3]

    largestclustersize_allframes=[]
    noofclusters_allframes=[]
    timestep_list=[]

    # cutoff=230 #choose the cutoff according to the size of the particles forming the cluster

    for frame in frames_list[::10]:
        snap=traj[int(frame)]
        timestep=snap.configuration.step
        timestep_list.append(timestep)
        points = snap.particles.position
        pairwise_distances=pdist(points,metric='euclidean')
        z = linkage(pairwise_distances, method='single',metric='euclidean')
        N=snap.particles.position.shape[0]
        clustering = split_into_clusters(z,cutoff,N)
        cluster_num = 0
        cluster_sizes=[]
        if clustering==None or len(clustering) == 0:
            print("No clusters! Most likely your distance cutoff is too high", frame)
            cluster_sizes.append(0)
            maxclustersize = 0
            noofclusters_allframes.append(0)
            index=cluster_sizes.index(maxclustersize)
            largestclustersize_allframes.append(maxclustersize)
        else:
            for cluster in clustering:
                cluster_num+=1
                cluster_sizes.append(len(clustering[cluster]))
            maxclustersize=np.max(cluster_sizes)
            noofclusters_allframes.append(len(cluster_sizes))
            index=cluster_sizes.index(maxclustersize)
            largestclustersize_allframes.append(maxclustersize)

    largestclustersize_allframes=np.array(largestclustersize_allframes)
    noofclusters_allframes=np.array(noofclusters_allframes)
    timestep_list=np.array(timestep_list)

    #Saving output file containing clustering analysis data over timesteps into the same directory as simulation trajectory
    output_file_clusteranalysis=F"{outputprefix}.clusteranalysis.data"
    d=np.array([timestep_list,largestclustersize_allframes,noofclusters_allframes],dtype=object)
    # print(d)
    d.dump(output_file_clusteranalysis)
    return [timestep_list, largestclustersize_allframes, noofclusters_allframes]

def plot_clustering_time(ts_list, max_cluster_size_list, num_cluster_list, outputprefix):
    import matplotlib.pyplot as plt
    import matplotlib
    from mpl_toolkits.mplot3d import Axes3D
    matplotlib.rcParams['savefig.dpi'] = 300
    matplotlib.rcParams['lines.linewidth'] = 3.0
    # matplotlib.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.linewidth'] = 3.0
    matplotlib.use('Agg')
    #Plotting the desired quantities as a function of simulation timestep
    plot_path = F"{outputprefix}.plot.png"
    fig, ax1 = plt.subplots(figsize=(8,6))
    ax1.plot(ts_list, max_cluster_size_list, color='red', marker='.',linewidth=3.0,label='Size of largest cluster')
    ax2=ax1.twinx()
    ax2.plot(ts_list, num_cluster_list, color='navy', marker='.',linewidth=3.0,label='Number of clusters')
    #ax1.legend(loc='upper right',ncol=1)
    #ax2.legend(loc='lower right',ncol=1)
    ax1.tick_params(axis='y',colors='red')
    ax2.tick_params(axis='y',colors='navy')
    ax1.set_xlabel('Simulation timestep',fontsize=30)
    ax1.set_ylabel('Size of largest cluster',fontsize=30)
    ax2.set_ylabel('Number of clusters',fontsize=30)
    ax1.yaxis.label.set_color('red')
    ax2.yaxis.label.set_color('navy')
    ax1.spines['left'].set_color('red')
    ax1.spines['right'].set_color('navy')
    fig.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return plot_path

