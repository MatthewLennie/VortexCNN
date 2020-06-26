#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 21:18:15 2019
input: array=numpy array with row wise stacked time series 
output:
@author: Johannes Steenbuck & Matthew Lennie
"""
#Load modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tslearn.barycenters import softdtw_barycenter
from scipy.signal import butter,filtfilt,resample
import dtw # Upload in repo!!
from itertools import combinations
from scipy.signal import argrelextrema
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette,fcluster
from matplotlib import colors
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_samples
from sklearn import neighbors
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
#%% Plot functions
def plotAllRepetitions(preprocessed_data,gamma):
    plt.figure()
    [plt.plot(i,linewidth=0.5,color='grey') for i in preprocessed_data]
    plt.plot(np.mean(preprocessed_data,axis=0),color='r',label='Arithmetic barycentre')
    plt.plot(softdtw_barycenter(preprocessed_data,gamma=gamma),color='b',label='Soft-DTW barycentre')
    plt.ylabel(r'')
    plt.xlabel(r'')
    plt.legend()
    plt.xlim([0,preprocessed_data.shape[1]])

def plotLineBar(x,y):
    plt.ylim([0,np.max(y)+0.1])
    [plt.axvline(x=v, ymin=0, ymax=m/(np.max(y)+.1), lw=8) for v,m in zip(x,y)]
    if np.min(x)==0:
        plt.xlim([-0.1*np.max(x),np.max(x)*1.1])
    else:
        plt.xlim([np.min(x)*0.9,np.max(x)*1.1])
        
    plt.xticks(x)
    plt.tight_layout()
    
def createColorList():
    cmap = plt.get_cmap("tab20")
    cmap2 = plt.get_cmap("tab20b")
    
    cmap_bold=[cmap(i) for i in range(0,20,2)]
    cmap_light=[cmap(i) for i in range(1,20,2)]
    
    bold2list=[0,1,4,5,8,9,12,13,16,17]
    light2list=[i for i in range(0,20) if i not in bold2list]
    cmap_bold2=[cmap2(i) for i in bold2list]    
    cmap_light2=[cmap2(i) for i in light2list]
    colorlist=cmap_bold+cmap_bold2+cmap_light+cmap_light2
    
    col_list_hex=[]
    for i in range(len(colorlist)):
        rgb = colorlist[i][:3] # will return rgba, we take only first 3 so we get rgb
        col_list_hex.append(colors.rgb2hex(rgb))
    
    #colorlist_rgb=colorlist    
    colorlist=col_list_hex 
    
    return colorlist

def visualizeClusterResults(ClusterResults,metric,clusternum,preprocessed_data,centroids,plots=['Dendogram','MDS2','MDS3','Clustered_TS']):
    if 'Dendogram' in plots:
        plotDend(metric[0],clusternum)
    if 'MDS2' in plots:
        MDSPlot(ClusterResults,metric[1],dim=2,sils=True)
    if 'MDS3' in plots:
        MDSPlot(ClusterResults,metric[1],dim=3)
    if 'Clustered_TS' in plots:
        plotClusters(preprocessed_data,ClusterResults,Centroids=centroids)
    
    return None
    
def plotDend(Distances,clusternum):
    plt.figure()
    link=linkage(Distances,method='ward')
    ct=link[-(clusternum-1),2]
    set_link_color_palette(createColorList())
    if np.sqrt(2*len(Distances))>50:
        no_labels=True
    else:
        no_labels=False
    
    dendrogram(link,color_threshold = ct, orientation="left", above_threshold_color='grey',no_labels=no_labels)
    #dend=dendrogram(link,color_threshold = ct, above_threshold_color='grey')
    plt.xlabel('Distance')
    plt.ylabel('Data elements')
    plt.gca().yaxis.set_label_position('right')
    plt.grid(False)
    
def MDSPlot(ClusterResults,distanceMatFull,dim=3,sils=True):    
    reduced = MDS(n_components=dim,dissimilarity="precomputed")
    if dim==3:
        ClusterResults['var1'],ClusterResults['var2'],ClusterResults['var3'] = reduced.fit_transform(distanceMatFull).T
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(ClusterResults.var1,ClusterResults.var2,ClusterResults.var3,c=ClusterResults['colors'])
#        threedee = plt.figure().gca(projection='3d')        
#        threedee.scatter(ClusterResults.var1,ClusterResults.var2,ClusterResults.var3,c=ClusterResults['colors'])
    else:
        if sils:
            fig, (ax1, ax2) = plt.subplots(1, 2)
        else:
            fig=plt.figure()
            ax2=plt.gca()
        ClusterResults['var1'],ClusterResults['var2'] = reduced.fit_transform(distanceMatFull).T
        clf = neighbors.KNeighborsClassifier(3, weights='uniform')
        clf.fit(ClusterResults[['var1','var2']],ClusterResults['Clusters'])
        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = ClusterResults['var1'].min() - 1, ClusterResults['var1'].max() + 1
        y_min, y_max = ClusterResults['var2'].min() - 1, ClusterResults['var2'].max() + 1
        h = 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        
#        cmap_light = ListedColormap(['#FFAAAA', '#66FF66', '#6666FF', '#6666FF', '#66FFFF', '#FF99FF'])
#        cmap_light = ListedColormap([ (0.6823529411764706, 0.7803921568627451, 0.9098039215686274, 1.0), (1.0, 0.7333333333333333, 0.47058823529411764, 1.0)])
        cmap_light = colors.ListedColormap(createColorList()[20:20+ClusterResults.Clusters.max()])
#        plt.figure()                            
        ax2.pcolormesh(xx, yy, Z,cmap=cmap_light)#,**{'zorder':2})
        scaler = MinMaxScaler()
        ax2.scatter(ClusterResults['var1'], ClusterResults['var2'], c=ClusterResults['colors'], cmap=ClusterResults['colors'],
                edgecolor='k', s=15 + scaler.fit_transform(ClusterResults['silSamples'].values.reshape(-1, 1))*50)
#        ax2.set_yticks([])
#        ax2.set_xticks([])
#        ax2.set_title("MDS Plot")
        ClusterResults_sorted=ClusterResults.sort_values(["Clusters","silSamples"],ascending=[True,True]) 
        y_lower=0
        if sils:
            ax1.set_xlim([-0.3, 1])
            for i in range(1,ClusterResults_sorted.Clusters.max()+1):
                cluster_height=len(ClusterResults_sorted[ClusterResults_sorted["Clusters"]==i])
                y_upper=y_lower+cluster_height
                ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, list(ClusterResults_sorted.silSamples[ClusterResults_sorted["Clusters"]==i]),facecolor=createColorList()[i-1], edgecolor='k', alpha=0.7)
                ax1.text(-0.05,(y_lower+y_upper)/2 -0.8, str(i),fontsize=16)
    #             y_lower + cluster_height/2-(i-1)*0.5
                y_lower=y_upper+0.5
            ax1.axvline(x=ClusterResults.silSamples.mean(), color="red", linestyle="--")
            ax1.set_yticks([])
            ax1.grid('false')
            ax1.set_xlabel("Silhouette score")
#            ax1.set_title("Silhouette plot for each Cluster")
            ax1.set_ylabel("Cluster number")
    plt.tight_layout()
    
def plotClusters(StackedDF,ClusterResults,Centroids=None):
    """
    Plots timelines of each cluster with Centroids in own subplot \n
    :param data: DataFrame with repetition in each row
    :param ClusterResults:  dataframe with index Clusters which has cluster number. indexing from 1 (output from RunCluster)
    :param Centroids: List of the Centroids (Output from CalcCentroid) 
    :return: 
    """
    colorlist=createColorList()
    data=np.array(StackedDF)

    
    clusternum=ClusterResults.Clusters.max()
    xlen=data.shape[1]
    xticklabel=['0',r'$\pi$/2',r'$\pi$',r'$3\pi$/2',r'2$\pi$']
    
    
    if clusternum < 4:
        fig, axs = plt.subplots(clusternum,1, sharey='all',sharex='all')
        for clust in range(clusternum):
            trick_phase=np.linspace(0,np.pi*2,xlen)  
#            [axs[clust].plot(trick_phase,i,c=aux.colorlist[20+clust],linewidth=1) for i in data[ClusterResults.Clusters==clust+1]]
            if Centroids is not None:
                [axs[clust].plot(trick_phase,i,c=colorlist[20+clust],linewidth=1) for i in data[ClusterResults.Clusters==clust+1]]
                axs[clust].plot(trick_phase,Centroids[clust],c=colorlist[0+clust])
            else:
                [axs[clust].plot(trick_phase,i,c=colorlist[clust],linewidth=1) for i in data[ClusterResults.Clusters==clust+1]]
            axs[clust].set_title('Cluster {}'.format(clust+1))
            axs[clust].set_ylabel('$c_{P,1}$')    
    elif clusternum < 9:
        if clusternum%2==1:
            plots=clusternum+1
        else:
            plots=clusternum
            
        fig, axs = plt.subplots(int(plots/2),2, sharey='all',sharex='all')
        axs=axs.reshape(-1)
        
            
        for clust in range(clusternum):
                trick_phase=np.linspace(0,np.pi*2,xlen)  
                if Centroids is not None:
                    [axs[clust].plot(trick_phase,i,c=colorlist[20+clust],linewidth=1) for i in data[ClusterResults.Clusters==clust+1]]
                    axs[clust].plot(trick_phase,Centroids[clust],c=colorlist[0+clust])
                else:
                    [axs[clust].plot(trick_phase,i,c=colorlist[clust],linewidth=1) for i in data[ClusterResults.Clusters==clust+1]]
                
                axs[clust].set_title('Cluster {}'.format(clust+1))
                axs[clust].set_ylabel('$c_L$')
        if clusternum%2==1:            
            plt.setp(axs[clust+1], xlim=[0,2*np.pi], xlabel='Phase',
                     xticks=[0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],xticklabels=xticklabel) 
        else:
            plt.setp(axs[clust-1], xlim=[0,2*np.pi], xlabel='Phase',
                     xticks=[0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],xticklabels=xticklabel) 
    else:
        if clusternum%3==1:
            plots=clusternum+2
        elif clusternum%3==2:
            plots=clusternum+1
        else:
            plots=clusternum
            
        fig, axs = plt.subplots(int(plots/3),3, sharey='all',sharex='all')
        axs=axs.reshape(-1)
            
        for clust in range(clusternum):
                trick_phase=np.linspace(0,np.pi*2,xlen)  
                if Centroids is not None:
                    [axs[clust].plot(trick_phase,i,c=colorlist[20+clust],linewidth=1) for i in data[ClusterResults.Clusters==clust+1]]
                    axs[clust].plot(trick_phase,Centroids[clust],c=colorlist[0+clust])
                else:
                    [axs[clust].plot(trick_phase,i,c=colorlist[clust],linewidth=1) for i in data[ClusterResults.Clusters==clust+1]]
                
                axs[clust].set_title('Cluster {}'.format(clust+1),size=13)
                axs[clust].set_ylabel('$c_L$',size=16)
        if clusternum%3==1:            
            plt.setp(axs[clust+1], xlim=[0,2*np.pi], xlabel='Phase',
                     xticks=[0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],xticklabels=xticklabel) 
            plt.setp(axs[clust+2], xlim=[0,2*np.pi], xlabel='Phase', 
                     xticks=[0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],xticklabels=xticklabel) 
        elif clusternum%3==2:
            plt.setp(axs[clust-1], xlim=[0,2*np.pi], xlabel='Phase',
                     xticks=[0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],xticklabels=xticklabel) 
            plt.setp(axs[clust+1], xlim=[0,2*np.pi], xlabel='Phase',
                     xticks=[0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],xticklabels=xticklabel) 
        else:
            plt.setp(axs[clust-1], xlim=[0,2*np.pi], xlabel='Phase',
                     xticks=[0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],xticklabels=xticklabel) 
            plt.setp(axs[clust-2], xlim=[0,2*np.pi], xlabel='Phase',
                     xticks=[0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],xticklabels=xticklabel) 
            
    plt.setp(axs[clust], xlim=[0,2*np.pi], xlabel='Phase',
                 xticks=[0,np.pi/2,np.pi,3*np.pi/2,2*np.pi],xticklabels=xticklabel)    
    fig.tight_layout()

    #%% Analysis functions
def preprocessData(array,downsample=6):
    """
    Filters and resamples the input array
    :param array: np.array containing cycles stacked row wise
    :param downsample: factor to reduce the signal length by. Significantly decreases time comsumption.
    """
    
    resample_value=round(array.shape[1]/downsample)
    # butter generates the butterworth filter. The first input (x) to butter(x,y) is the filter order, the second (y) is the cutoff frequency. 
    # the frequency is calculated by f_cut=y*f_sample/2
    b, a = butter(3, 0.1)
    #preprocessing
    preprocessed_data=resample(filtfilt(b,a,array),resample_value,axis=1)
    return preprocessed_data


def ClustNumSils(distances,distanceMatFull,plot=True,maxclusts=10):
    """
    Calculates Silhouette Scores for Clusternumber 2:maxclust  \n
    :param Distances: metric[0]
    :param num: number of timeseries
    :param plot(bool): To plot or not to plot
    :param maxlust: Number to which the Silscores are calculated
    :return: recommendation for number of clusters 
    """

    sils=[]
    for clust in range(2,maxclusts):
        ClusterResults = RunCluster(distances,distanceMatFull,clusternum=clust,printresult=False)
        sils.append(ClusterResults.silSamples.mean())
    
    
    bestclust=[i+2 for i in argrelextrema(np.asarray(sils), np.greater)[0]]
    if plot:    
        plt.figure()    
#        plt.plot(range(2,maxclusts),sils)    
        plotLineBar(range(2,maxclusts),sils)
        plt.xlabel('Clusters')
        plt.ylabel('Mean of Silhouette Score')
    
    
    
    bestclust=range(2,maxclusts)[sils.index(max(sils))]
    
    return bestclust

def RunCluster(distances,distanceMatFull,clusternum,printresult=True):
    
    clusters = fcluster(linkage(distances,method='ward'),t=clusternum,criterion='maxclust')

    
    ClusterResults = pd.DataFrame()
    ClusterResults['silSamples'] = silhouette_samples(distanceMatFull, labels=clusters, metric='precomputed')
    
    ClusterResults['Clusters'] = clusters
#    ClusterResults['colors'] = [list(colors.values())[x] for x in ClusterResults['Clusters']]
    colorlist=createColorList()
    color = [ colorlist[col-1] for col in ClusterResults.Clusters.values]
    ClusterResults['colors'] = color
    if printresult:
        print(ClusterResults['Clusters'])
      
    return  ClusterResults  

def getDistanceMetric(preprocessData,number_of_repetitions):
    #calculates the distance metric, outputs the distances as a list, a matrix and the warping pathes
    print('Calculating distances, this may take a while')
    DistanceList=[dtw.warping_path(preprocessData[TS1],preprocessData[TS2]) for TS1,TS2 in combinations(range(number_of_repetitions),2)]
    distances = [i[0] for i in DistanceList]
    paths= [i[1] for i in DistanceList]
    
    distanceMatFull = np.zeros((number_of_repetitions,number_of_repetitions))
    for x, (i,j) in zip(distances,combinations(range(0,number_of_repetitions),2)):
        distanceMatFull[int(i),int(j)] = x
        distanceMatFull[int(j),int(i)] = x
    return [distances,distanceMatFull,paths]

def CalcCentroid(StackedDF,ClusterResults,gamma=1):
    """
    Calculates Centroid  
    :param StackedDF: np.array containing cycles stacked row wise
    :param ClusterResults:  dataframe with index Clusters which has cluster number. indexing from 1
    :return: Centroids 2D array
    """

    print("Computing Centroids")
    #Compute centroids with Soft-DTW
    data=np.array(StackedDF)
    
    Centroids = []
    for clust in range(len(ClusterResults['Clusters'].unique())):
        Centroids.append(softdtw_barycenter(data[ClusterResults['Clusters'] == clust+1],gamma=gamma, max_iter=5))
    return Centroids  
    

#%% Main
def main(array,gamma,clusterNum=None):

    number_of_repetitions=array.shape[0]
    #Applying filter and resample
    preprocessed_data=preprocessData(array)
    
    #inspect result as a plot of all repetitions with arithmetik and soft-dtw barycentres
    plotAllRepetitions(preprocessed_data,gamma)
    
    #applying the dtw distance metric
    metric=getDistanceMetric(preprocessed_data,number_of_repetitions)

    # look for maximum in following graph and set clusters accordingly    
    clusternumSils=ClustNumSils(metric[0],metric[1],number_of_repetitions,maxclusts=10)
    
    
    if clusterNum:
        print('Silhouette method suggests '+str(clusternumSils)+' clusters. '+str(clusterNum)+' clusters were chosen in this example')
    else:
        clusterNum=int(input('Silhouette suggests '+str(clusternumSils)+' clusters. Please insert your choice: '))
    
    
    ClusterResults=RunCluster(metric[0],metric[1],clusterNum,printresult=False)
    centroids=CalcCentroid(preprocessed_data,ClusterResults,gamma=gamma)
    plots=['Dendogram','MDS2','MDS3','Clustered_TS']
    visualizeClusterResults(ClusterResults,metric,clusterNum,preprocessed_data,centroids,plots=plots)


#load the example data
df=pd.read_pickle('example_data')#.sample(20)
data=df.values   

gamma=0.001 # relaxation factor of the DTW, the higher the smoother the barycenter
clusterNum= 3 # if you know the nomber of clusters in advance. Else clusterNum=None

main(data,clusterNum,gamma=gamma)
