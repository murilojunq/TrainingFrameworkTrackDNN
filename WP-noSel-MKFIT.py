import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # #Silences unnecessary spam from TensorFlow libraries. Set to 0 for full output
import gpusetter

import pandas as pd
import numpy as np
import awkward as ak
import uproot, os

from glob import glob
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc

import tensorflow as tf
import sys

WFILE=sys.argv[1]
print(WFILE)

'''
tool to convert a ROOT tkNtuple into a input and compute the DNN from a saved pb file (not frozen graph)
'''

columns_to_load = [
    "trk_pt", "trk_eta", "trk_phi",
    "trk_ptErr", "trk_etaErr", "trk_phiErr",
    "trk_px", "trk_py", "trk_pz",
    "trk_inner_px", "trk_inner_py", "trk_inner_pz", "trk_inner_pt",
    "trk_outer_px", "trk_outer_py", "trk_outer_pz", "trk_outer_pt",
    "trk_dxyClosestPV","trk_dzClosestPV", "trk_dxy", "trk_dz", "trk_dxyErr", "trk_dzErr",
    "trk_nChi2", "trk_ndof", "trk_nChi2_1Dmod",
    "trk_nCluster",
    "trk_nPixel", "trk_nStrip", "trk_nPixelLay", "trk_nStripLay","trk_n3DLay",
    "trk_nInnerInactive", "trk_nOuterInactive", "trk_nInnerLost", "trk_nOuterLost",
    "trk_nValid", "trk_nLostLay",
    "trk_originalAlgo", "trk_stopReason", "trk_mva", "trk_simTrkIdx",
]

#DNN established inputs: order must be preserved!
columns_of_regular_inputs = [
    "trk_pt",
    "trk_inner_px", "trk_inner_py", "trk_inner_pz", "trk_inner_pt",
    "trk_outer_px", "trk_outer_py", "trk_outer_pz", "trk_outer_pt",
    "trk_ptErr",
    "trk_dxyClosestPV", "trk_dzClosestPV", "trk_dxy", "trk_dz", "trk_dxyErr", "trk_dzErr",
    "trk_nChi2",
    "trk_eta", "trk_phi", "trk_etaErr", "trk_phiErr",
    "trk_nPixel", "trk_nStrip",
    "trk_ndof",
    "trk_nInnerLost", "trk_nOuterLost", "trk_nInnerInactive", "trk_nOuterInactive", "trk_nLostLay"
]

def load_to_pandas(file_, columns, columns_of_regular_inputs):
        try:
            tree = uproot.open(file_)["trackingNtuple/tree"].arrays(columns)
            flat_tree = {}
            for ele in tree:
                ele = ak.to_awkward0(ele)
                if not bool(flat_tree):
                    flat_tree = ele
                else:
                    for key in ele.keys():
                        flat_tree[key]=np.concatenate((flat_tree[key],ele[key]))

            dataframe = pd.DataFrame(flat_tree, columns=columns) #Explicitly using float32 to save memory during training
            dataframe = label(dataframe) # if no _seed = True during the production of the trackNtuple
            dataframe = dataframe.loc[:, columns_of_regular_inputs+["trk_originalAlgo"]+["trk_isTrue"]].astype('float32')
            return dataframe
        except:
            return

def label(dataframe):
        dataframe.loc[:, "trk_isTrue"] = dataframe.loc[:, "trk_simTrkIdx"].apply(lambda x: 1 if len(x)>0  else 0)
        dataframe.drop(columns='trk_simTrkIdx')
        return dataframe


qualityCutDictionary = {
   "InitialStep":          [-0.56, -0.08, 0.17],
   "LowPtQuadStep":        [-0.35, 0.13, 0.36],
   "HighPtTripletStep":    [0.41, 0.49, 0.57],
   "LowPtTripletStep":     [-0.29, 0.09, 0.36],
   "DetachedQuadStep":     [-0.63, -0.14, 0.49],
   "DetachedTripletStep":  [-0.32, 0.24, 0.81],
   "PixelPairStep":        [-0.38, -0.23, 0.04],
   "MixedTripletStep":     [-0.83, -0.63, -0.38],
   "PixelLessStep":        [-0.60, -0.40, 0.02],
   "TobTecStep":           [-0.71, -0.58, -0.46], 
   "JetCoreRegionalStep":  [-0.53, -0.33, 0.18],
}

trackdnn_CKF = {
   "InitialStep"         :   [-0.49, 0.08, 0.34],
   "LowPtQuadStep"       :   [-0.29, 0.17, 0.39],
   "HighPtTripletStep"   :   [0.5, 0.58, 0.65],
   "LowPtTripletStep"    :   [-0.30, 0.06, 0.32],
   "DetachedQuadStep"    :   [-0.61, -0.09, 0.51],
   "DetachedTripletStep" :   [-0.38, 0.31, 0.83],
   "PixelPairStep"       :   [-0.25, -0.07, 0.19],
   "MixedTripletStep"    :   [-0.86, -0.57, -0.12],
   "PixelLessStep"       :   [-0.81, -0.61, -0.17],
   "TobTecStep"          :   [-0.67, -0.54, -0.40],
   "JetCoreRegionalStep" :   [0.00, 0.03, 0.68]
}

iterDict={

   "InitialStep"         :   4,
   "LowPtQuadStep"       :   23,
   "HighPtTripletStep"   :   22,
   "LowPtTripletStep"    :   5,
   "DetachedQuadStep"    :   24,
   "DetachedTripletStep" :   7,
   "PixelPairStep"       :   6,
   "MixedTripletStep"    :   8,
   "PixelLessStep"       :   9,
   "TobTecStep"          :   10,
   "JetCoreRegionalStep" :   11,

}

effDictCKF={

   "LowPtQuadStep"       :   [0.9765811051607906, 0.9320072618959112, 0.898491741631912],
   "LowPtTripletStep"    :   [0.8928607007978844, 0.8196065823197515, 0.7515192406966319],
   "PixelLessStep"       :   [0.9181593652304486, 0.8889481500931312, 0.8418984007680252]

}


def get_isoAVG_point(column, column2, df_tot, iteration="InitialStep"):

        iN=iterDict[iteration]
        df_tot=df_tot[df_tot["trk_originalAlgo"]==iN]
        originalWPs=qualityCutDictionary[iteration]
        isoEWPs=[]
        isoFWPs=[]
        origEffs=[]
        newEffs=[]
        origFs=[]
        newFs=[]

        data_col=df_tot[column]*2-1
        data_col2=df_tot[column2]*2-1

        fpr, tpr, thresholds = roc_curve(df_tot["trk_isTrue"], data_col)
        fpr2, tpr2, thresholds2 = roc_curve(df_tot["trk_isTrue"], data_col2)

        idx = (np.abs(thresholds - originalWPs[0])).argmin()
        origEffs.append(tpr[idx])
        origFs.append(fpr[idx])
        idx2 = (np.abs(tpr2 - tpr[idx])).argmin()
        isoEWPs.append(thresholds2[idx2])
        idx2 = (np.abs(fpr2 - fpr[idx])).argmin()
        isoFWPs.append(thresholds2[idx2])

        idx = (np.abs(thresholds - originalWPs[1])).argmin()
        origEffs.append(tpr[idx])
        origFs.append(fpr[idx])
        idx2 = (np.abs(tpr2 - tpr[idx])).argmin()
        isoEWPs.append(thresholds2[idx2])
        idx2 = (np.abs(fpr2 - fpr[idx])).argmin()
        isoFWPs.append(thresholds2[idx2])        

        idx = (np.abs(thresholds - originalWPs[2])).argmin()
        origEffs.append(tpr[idx])
        origFs.append(fpr[idx])
        idx2 = (np.abs(tpr2 - tpr[idx])).argmin()
        isoEWPs.append(thresholds2[idx2])
        idx2 = (np.abs(fpr2 - fpr[idx])).argmin()
        isoFWPs.append(thresholds2[idx2])
        
        newWPs=(np.array(isoEWPs)+np.array(isoFWPs))/2.       
        avgidx=(np.abs(thresholds2 - newWPs[0])).argmin()  
        newEffs.append(tpr2[avgidx])
        newFs.append(fpr2[avgidx])  
        avgidx=(np.abs(thresholds2 - newWPs[1])).argmin()    
        newEffs.append(tpr2[avgidx])
        newFs.append(fpr2[avgidx])
        avgidx=(np.abs(thresholds2 - newWPs[2])).argmin()    
        newEffs.append(tpr2[avgidx])
        newFs.append(fpr2[avgidx])
 
        print (iteration)
        print (originalWPs, "original WPs")
        print (newWPs, "NEW WPs")
        print (newWPs.round(2), "NEWR WPs")
        print (isoEWPs, isoFWPs)
        print ("effs", origEffs, newEffs)
        print ("frs", origFs, newFs)

def get_isoFR_point(column, column2, df_tot, iteration="InitialStep"):

        iN=iterDict[iteration]
        df_tot=df_tot[df_tot["trk_originalAlgo"]==iN]
        originalWPs=qualityCutDictionary[iteration]  
        newWPs=[]
        origEffs=[]
        newEffs=[]
        origFs=[]
        newFs=[]

        data_col=df_tot[column]*2-1
        data_col2=df_tot[column2]*2-1

        fpr, tpr, thresholds = roc_curve(df_tot["trk_isTrue"], data_col)
        fpr2, tpr2, thresholds2 = roc_curve(df_tot["trk_isTrue"], data_col2)

        idx = (np.abs(thresholds - originalWPs[0])).argmin()
        origEffs.append(fpr[idx])
        idx2 = (np.abs(fpr2 - fpr[idx])).argmin()
        newWPs.append(thresholds2[idx2])
        newEffs.append(fpr2[idx2])
        origFs.append(tpr[idx])
        newFs.append(tpr2[idx2])

        idx = (np.abs(thresholds - originalWPs[1])).argmin()
        origEffs.append(fpr[idx])
        idx2 = (np.abs(fpr2 - fpr[idx])).argmin()
        newWPs.append(thresholds2[idx2])
        newEffs.append(fpr2[idx2])
        origFs.append(tpr[idx])
        newFs.append(tpr2[idx2])

        idx = (np.abs(thresholds - originalWPs[2])).argmin()
        origEffs.append(fpr[idx])
        idx2 = (np.abs(fpr2 - fpr[idx])).argmin()
        newWPs.append(thresholds2[idx2])
        newEffs.append(fpr2[idx2])
        origFs.append(tpr[idx])
        newFs.append(tpr2[idx2])

        print (iteration)
        print (originalWPs, "original WPs")
        print (newWPs, "new WPs")
        print ("frs", origEffs, newEffs, origFs, newFs)

def get_isoEff_point(column, column2, df_tot, iteration="InitialStep"):

        iN=iterDict[iteration]
        df_tot=df_tot[df_tot["trk_originalAlgo"]==iN]
        originalWPs=qualityCutDictionary[iteration]  
        newWPs=[] 
        origEffs=[]
        newEffs=[]   
        origFs=[]
        newFs=[]  

        data_col=df_tot[column]*2-1
        data_col2=df_tot[column2]*2-1
        
        fpr, tpr, thresholds = roc_curve(df_tot["trk_isTrue"], data_col)
        fpr2, tpr2, thresholds2 = roc_curve(df_tot["trk_isTrue"], data_col2)
       
        idx = (np.abs(thresholds - originalWPs[0])).argmin()
        origEffs.append(tpr[idx])
        idx2 = (np.abs(tpr2 - tpr[idx])).argmin() 
        newWPs.append(thresholds2[idx2]) 
        newEffs.append(tpr2[idx2])
        origFs.append(fpr[idx])
        newFs.append(fpr2[idx2])

        idx = (np.abs(thresholds - originalWPs[1])).argmin()
        origEffs.append(tpr[idx])
        idx2 = (np.abs(tpr2 - tpr[idx])).argmin()
        newWPs.append(thresholds2[idx2])
        newEffs.append(tpr2[idx2])
        origFs.append(fpr[idx])
        newFs.append(fpr2[idx2])
     
        idx = (np.abs(thresholds - originalWPs[2])).argmin()
        origEffs.append(tpr[idx])
        idx2 = (np.abs(tpr2 - tpr[idx])).argmin()
        newWPs.append(thresholds2[idx2])
        newEffs.append(tpr2[idx2])
        origFs.append(fpr[idx])
        newFs.append(fpr2[idx2])
        
        print (iteration)
        print (originalWPs, "original WPs")
        print (newWPs, "new WPs")
        print ("effs", origEffs, newEffs, origFs, newFs)


def get_isoEff_fromEff(column, column2, df_tot, iteration="InitialStep"):

        iN=iterDict[iteration]
        df_tot=df_tot[df_tot["trk_originalAlgo"]==iN]
        originalWPs=[]  
        newWPs=[]
        origEffs=effDictCKF[iteration]
        newEffs=[]

        data_col=df_tot[column]*2-1
        data_col2=df_tot[column2]*2-1

        fpr, tpr, thresholds = roc_curve(df_tot["trk_isTrue"], data_col)
        fpr2, tpr2, thresholds2 = roc_curve(df_tot["trk_isTrue"], data_col2)

        idx2 = (np.abs(tpr2 -origEffs[0])).argmin()
        #originalWPs.append(thresholds[idx])
        #idx2 = (np.abs(tpr2 - tpr[idx])).argmin()
        newWPs.append(thresholds2[idx2])
        newEffs.append(tpr2[idx2])

        idx2 = (np.abs(tpr2 - origEffs[1])).argmin()
        #originalWPs.append(thresholds[idx])
        #idx2 = (np.abs(tpr2 - tpr[idx])).argmin()
        newWPs.append(thresholds2[idx2])
        newEffs.append(tpr2[idx2])

        idx2 = (np.abs(tpr2 - origEffs[2])).argmin()
        #originalWPs.append(thresholds[idx])
        #idx2 = (np.abs(tpr2 - tpr[idx])).argmin()
        newWPs.append(thresholds2[idx2])
        newEffs.append(tpr2[idx2])

        print (iteration)
        print (originalWPs, "original WPs")
        print (newWPs, "new WPs")
        print ("effs", origEffs, newEffs)

path_to_save='/data2/legianni/TrainingFrameworkTrackDNN/MVA-newTraining125x-CKF-oldFiles-outofTrain/'

paths={
"DisSUSY1":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-113X/DisplacedSUSY_stopToBottom_M_1000_1000mm_TuneCP5_14TeV_pythia8/crab_DisSUSY11--fullCKF3-113/220824_210143/0000",
"DisSUSY2":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-113X/DisplacedSUSY_stopToBottom_M_1000_100mm_TuneCP5_14TeV_pythia8/crab_DisSUSY10--fullCKF3-113/220824_195412/0000",
"DisSUSY3":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-113X/DisplacedSUSY_stopToBottom_M_1800_1000mm_TuneCP5_14TeV_pythia8/crab_DisSUSY181--fullCKF3-113/220824_210000/0000",
"DisSUSY4":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-113X/DisplacedSUSY_stopToBottom_M_1800_100mm_TuneCP5_14TeV_pythia8/crab_DisSUSY180--fullCKF3-113/220824_210326/0000",
"QCD1":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-113X/QCD_Pt-15to7000_TuneCUETP8M1_Flat_14TeV-pythia8/crab_QCD--fullCKF3-113/220824_210727/0000",
"QCD2":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-113X/QCD_Pt-15to7000_TuneCUETP8M1_Flat_14TeV-pythia8/crab_QCD--fullCKF3-113/220824_210727/0001",
"TT":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-113X/TT_TuneCP5_14TeV-powheg-pythia8/crab_TT--fullCKF3-113/220824_210945/0000",
"ZToEE1":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-113X/ZToEE_TuneCUETP8M1_14TeV-pythia8/crab_ZEE--fullCKF3-113/220824_210544/0000",
"ZToEE2":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-113X/ZToEE_TuneCUETP8M1_14TeV-pythia8/crab_ZEE--fullCKF3-113/220824_210544/0001",
"RelValTTbar":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-newsamples-CMSSW125X/RelValTTbar_14TeV/crab_RelValTT--fullCKFn/220914_200030/0000",
"TrainTTbar":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-newsamples-CMSSW125X/TT_TuneCP5_13p6TeV-powheg-pythia8/crab_TT--fullCKFn/220905_235423/0000",
"RelValQCDHpT":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-newsamples-CMSSW125X/RelValQCD_Pt_1800_2400_14/crab_RelValQCDH--fullCKFn/221003_231642/0000"
}

paths={"RelValTTbar":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-newsamples-CMSSW125X/RelValTTbar_14TeV/crab_RelValTT--fullMKFITn/220914_200339/0000",
"RelValQCDHpT":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-newsamples-CMSSW125X/RelValQCD_Pt_1800_2400_14/crab_RelValQCDH--fullMKFITn/221003_231405/0000"}

for process in ["RelValTTbar", "RelValQCDHpT"]:#['DisSUSY2', 'DisSUSY3', 'DisSUSY4', 'QCD1', 'ZToEE1',]:#'TT', 'DisSUSY1', 'DisSUSY2', 'DisSUSY3', 'DisSUSY4', 'QCD1', 'ZToEE1',]:
        files = glob(''+paths[process]+'/tr*root')[0:20]
        n = 4
        n_bag = len(files)/n+1
        file_bags = np.array_split(files, n_bag)
        i=1
        df_tot_init=False
        for bag in file_bags[:-1]:
            print("running %s percent of the job" %str(float(i)/float(n_bag)*100))
            with ProcessPoolExecutor(max_workers=12) as executor:
                dfs = list(executor.map(load_to_pandas, bag, repeat(columns_to_load), repeat(columns_of_regular_inputs)))

            print(len(bag), "files in the bag")
            for df_ in dfs:
              print (len(df_), " ---- tracks per file")

            if (not df_tot_init):
              if len(dfs)>1: df_tot = dfs[0].append(dfs[1:])
              else: df_tot = dfs[0]
              df_tot_init=True
            else:
              df_tot = df_tot.append(dfs)

            print (len(df_tot), " ----- tracks in the full matrix")

            print("finished %s percent of the job" %str(float(i)/float(n_bag)*100))
            i+=1

        print(df_tot.shape)
        df_tot.dropna(inplace= True)
        print(df_tot.shape)

        column="new_MVA"

        #make model
        print("Found it")
        stored_model = tf.keras.models.load_model(WFILE)#"retrain-ALL-CKF125-OLDFiles_v2/modelTEST-04/model/")#("result_minxi/CKF_final/model")
        #predict
        prediction=stored_model.predict([df_tot[columns_of_regular_inputs].to_numpy(), df_tot["trk_originalAlgo"].to_numpy()])

        #add column under the name column
        print (df_tot.shape, prediction.shape)
        df_tot.loc[:, column] = prediction # or prediction[:,0]
        print (df_tot.shape)

        columnOLD="new_MVA2"

         #make model
        print("Found it")
        stored_model2 = tf.keras.models.load_model("result_minxi/run3_oneshot_mkfit6v2/model")
        #predict
        prediction2=stored_model2.predict([df_tot[columns_of_regular_inputs].to_numpy(), df_tot["trk_originalAlgo"].to_numpy()])

        #add column under the name column
        print (df_tot.shape, prediction2.shape)
        df_tot.loc[:, columnOLD] = prediction2 # or prediction[:,0]
        print (df_tot.shape)
       
        get_isoEff_point(columnOLD, column, df_tot, iteration="InitialStep")
        get_isoEff_point(columnOLD, column, df_tot, iteration="LowPtQuadStep")
        get_isoEff_point(columnOLD, column, df_tot, iteration="HighPtTripletStep")
        get_isoEff_point(columnOLD, column, df_tot, iteration="LowPtTripletStep")
        get_isoEff_point(columnOLD, column, df_tot, iteration="DetachedQuadStep")
        get_isoEff_point(columnOLD, column, df_tot, iteration="DetachedTripletStep")
        get_isoEff_point(columnOLD, column, df_tot, iteration="MixedTripletStep")
        get_isoEff_point(columnOLD, column, df_tot, iteration="PixelLessStep")
        get_isoEff_point(columnOLD, column, df_tot, iteration="TobTecStep")
        get_isoEff_point(columnOLD, column, df_tot, iteration="PixelPairStep")
        get_isoEff_point(columnOLD, column, df_tot, iteration="JetCoreRegionalStep")
        ##
        get_isoFR_point(columnOLD, column, df_tot, iteration="InitialStep")
        get_isoFR_point(columnOLD, column, df_tot, iteration="LowPtQuadStep")
        get_isoFR_point(columnOLD, column, df_tot, iteration="HighPtTripletStep")
        get_isoFR_point(columnOLD, column, df_tot, iteration="LowPtTripletStep")
        get_isoFR_point(columnOLD, column, df_tot, iteration="DetachedQuadStep")
        get_isoFR_point(columnOLD, column, df_tot, iteration="DetachedTripletStep")
        get_isoFR_point(columnOLD, column, df_tot, iteration="MixedTripletStep")
        get_isoFR_point(columnOLD, column, df_tot, iteration="PixelLessStep")
        get_isoFR_point(columnOLD, column, df_tot, iteration="TobTecStep")
        get_isoFR_point(columnOLD, column, df_tot, iteration="PixelPairStep")
        get_isoFR_point(columnOLD, column, df_tot, iteration="JetCoreRegionalStep")
        ###
        get_isoAVG_point(columnOLD, column, df_tot, iteration="InitialStep")
        get_isoAVG_point(columnOLD, column, df_tot, iteration="LowPtQuadStep")
        get_isoAVG_point(columnOLD, column, df_tot, iteration="HighPtTripletStep")
        get_isoAVG_point(columnOLD, column, df_tot, iteration="LowPtTripletStep")
        get_isoAVG_point(columnOLD, column, df_tot, iteration="DetachedQuadStep")
        get_isoAVG_point(columnOLD, column, df_tot, iteration="DetachedTripletStep")
        get_isoAVG_point(columnOLD, column, df_tot, iteration="MixedTripletStep")
        get_isoAVG_point(columnOLD, column, df_tot, iteration="PixelLessStep")
        get_isoAVG_point(columnOLD, column, df_tot, iteration="TobTecStep")
        get_isoAVG_point(columnOLD, column, df_tot, iteration="PixelPairStep")
        get_isoAVG_point(columnOLD, column, df_tot, iteration="JetCoreRegionalStep")
