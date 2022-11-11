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

columns_of_regular_inputs2 = [
    "trk_pt",
    "trk_inner_eta", "trk_inner_phi",  "trk_inner_pt",
    "trk_outer_eta", "trk_outer_phi",  "trk_outer_pt",
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
            print (dataframe.shape, "ddddd")
            dataframe[ "trk_inner_phi"] = np.arctan2(dataframe["trk_inner_py"],dataframe["trk_inner_px"])
            dataframe["trk_outer_phi"] = np.arctan2(dataframe["trk_outer_py"],dataframe["trk_outer_px"])
            dataframe["trk_inner_eta"] = np.arcsinh(dataframe["trk_inner_pz"]/dataframe["trk_inner_pt"])
            dataframe["trk_outer_eta"] = np.arcsinh(dataframe["trk_outer_pz"]/dataframe["trk_outer_pt"])
            print (dataframe.shape, "ddddd")
            dataframe = dataframe.loc[:, columns_of_regular_inputs2+["trk_originalAlgo"]+["trk_isTrue"]].astype('float32')

            return dataframe
        except:
            return 
          
def label(dataframe):
        dataframe.loc[:, "trk_isTrue"] = dataframe.loc[:, "trk_simTrkIdx"].apply(lambda x: 1 if len(x)>0  else 0)
        dataframe.drop(columns='trk_simTrkIdx')
        return dataframe

def mva_plots(column, df_tot, process, iteration=None, suffix=""):
        
        if (iteration!=None):
          df_tot=df_tot[df_tot["trk_originalAlgo"]==iteration]          
        
        data_col=df_tot[column]
        binning=np.linspace(-1,2,100)

        plt.hist(np.clip(data_col, binning[1], binning[-2]), bins=binning, log=False, histtype='step', color="blue", label="all")
        plt.hist(np.clip(data_col[df_tot["trk_isTrue"]==0], binning[1], binning[-2]), bins=binning, log=False, histtype='step', color="red", label="fake")
        plt.hist(np.clip(data_col[df_tot["trk_isTrue"]==1], binning[1], binning[-2]), bins=binning, log=False, histtype='step', color="green", label="true")
        plt.legend(loc='upper right')          

        fpr, tpr, thresholds = roc_curve(df_tot["trk_isTrue"], data_col)
        auc_score = auc(fpr, tpr)
        auc_score = max(auc_score, 1-auc_score)

        plt.title(column+" --- single var. discr. power (AUC) {:.3f}".format(auc_score))

        if not os.path.isdir(path_to_save+process+"/fig/"+column+"/"):
              os.makedirs(path_to_save+process+"/fig/"+column+"/")
              
        plt.savefig(path_to_save+process+"/fig/"+column+"/"+column+"__plot"+suffix+".png")
        plt.clf()

        plt.hist(np.clip(data_col, binning[1], binning[-2]), bins=binning, log=True, histtype='step', color="blue", label="all")
        plt.hist(np.clip(data_col[df_tot["trk_isTrue"]==0], binning[1], binning[-2]), bins=binning, log=True, histtype='step', color="red", label="fake")
        plt.hist(np.clip(data_col[df_tot["trk_isTrue"]==1], binning[1], binning[-2]), bins=binning, log=True, histtype='step', color="green", label="true")
        plt.legend(loc='upper right')          

        fpr, tpr, thresholds = roc_curve(df_tot["trk_isTrue"], data_col)
        auc_score = auc(fpr, tpr)
        auc_score = max(auc_score, 1-auc_score)

        plt.title(column+" --- single var. discr. power (AUC) {:.3f}".format(auc_score))

        plt.savefig(path_to_save+process+"/fig/"+column+"/"+column+"__plotLOG"+suffix+".png")
        plt.clf()

        plt.plot(tpr, fpr, label="baseline mva AUC = {:.3f}".format(auc_score), color="red", linewidth=2)
        plt.legend()
        plt.title("ROC curve")
        plt.ylabel("Fake rate")
        plt.xlabel("True efficiency")
        plt.xlim(0.0, 1.05)
        plt.ylim(0.0, 1.05)
        plt.grid(True,which="both")
        plt.savefig(path_to_save+process+"/fig/"+column+"/"+"ROC_baseline"+suffix+".png")
        plt.clf()
        
        plt.plot(tpr, fpr, label="baseline mva AUC = {:.3f}".format(auc_score), color="red", linewidth=2)
        plt.legend()
        plt.title("ROC curve")
        plt.ylabel("Fake rate")
        plt.xlabel("True efficiency")
        plt.semilogy()
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0001, 1.0)
        plt.grid(True,which="both")
        plt.savefig(path_to_save+process+"/fig/"+column+"/"+"ROC_baseline_logy"+suffix+".png")
        plt.clf()

path_to_save='/data2/legianni/TrainingFrameworkTrackDNN/MVA-test1epoch-innerPtEtaPhi/'

paths={
"DisSUSY1":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-113X/DisplacedSUSY_stopToBottom_M_1000_1000mm_TuneCP5_14TeV_pythia8/crab_DisSUSY11--fullCKF3-113/220824_210143/0000",
"DisSUSY2":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-113X/DisplacedSUSY_stopToBottom_M_1000_100mm_TuneCP5_14TeV_pythia8/crab_DisSUSY10--fullCKF3-113/220824_195412/0000",
"DisSUSY3":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-113X/DisplacedSUSY_stopToBottom_M_1800_1000mm_TuneCP5_14TeV_pythia8/crab_DisSUSY181--fullCKF3-113/220824_210000/0000",
"DisSUSY4":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-113X/DisplacedSUSY_stopToBottom_M_1800_100mm_TuneCP5_14TeV_pythia8/crab_DisSUSY180--fullCKF3-113/220824_210326/0000",
"QCD1":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-113X/QCD_Pt-15to7000_TuneCUETP8M1_Flat_14TeV-pythia8/crab_QCD--fullCKF3-113/220824_210727/0000",
"QCD2":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-113X/QCD_Pt-15to7000_TuneCUETP8M1_Flat_14TeV-pythia8/crab_QCD--fullCKF3-113/220824_210727/0001",
"TT":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-113X/TT_TuneCP5_14TeV-powheg-pythia8/crab_TT--fullCKF3-113/220824_210945/0000",
"ZToEE1":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-113X/ZToEE_TuneCUETP8M1_14TeV-pythia8/crab_ZEE--fullCKF3-113/220824_210544/0000",
"ZToEE2":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-113X/ZToEE_TuneCUETP8M1_14TeV-pythia8/crab_ZEE--fullCKF3-113/220824_210544/0001"
}

for process in ['TT']:#['DisSUSY1', 'DisSUSY2', 'DisSUSY3', 'DisSUSY4', 'QCD1', 'TT', 'ZToEE1',]:
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
            
            #if (len(df_tot)>5000000):
              #break
            
        #check stats
        print(df_tot.shape)
        df_tot.dropna(inplace= True) 
        print(df_tot.shape)
        
        column="new_MVA"
        
        #make model
        print("Found it")
        stored_model = tf.keras.models.load_model("retrain-testingInnerOuterEtaPhi/modelTEST-01")#("result_minxi/CKF_final/model")
        
        #predict
        prediction=stored_model.predict([df_tot[columns_of_regular_inputs2].to_numpy(), df_tot["trk_originalAlgo"].to_numpy()])
        
        #add column under the name column
        print (df_tot.shape, prediction.shape)
        df_tot.loc[:, column] = prediction # or prediction[:,0]
        print (df_tot.shape)
        
        mva_plots(column, df_tot, process)
        mva_plots(column, df_tot, process, iteration=4, suffix="InitialStep")
        mva_plots(column, df_tot, process, iteration=23, suffix="LowPtQuadStep")
        mva_plots(column, df_tot, process, iteration=22, suffix="HighPtTripletStep")
        mva_plots(column, df_tot, process, iteration=5, suffix="LowPtTripletStep")
        mva_plots(column, df_tot, process, iteration=24, suffix="DetachedQuadStep")
        mva_plots(column, df_tot, process, iteration=7, suffix="DetachedTripletStep")
        mva_plots(column, df_tot, process, iteration=6, suffix="PixelPairStep")
        mva_plots(column, df_tot, process, iteration=8, suffix="MixedTripletStep")
        mva_plots(column, df_tot, process, iteration=9, suffix="PixelLessStep")
        mva_plots(column, df_tot, process, iteration=10, suffix="TobTecStep")
