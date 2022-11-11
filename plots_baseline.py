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
#ffs
from sklearn.metrics import roc_curve, auc

'''
tool to convert a ROOT tkNtuple into a dataframe for preliminary plots 
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


bins_dictionary={
  
'trk_pt':[-1.,300.,100,],
'trk_eta':[-5.5,5.5,100,],
'trk_phi':[-4,4,100,],

'trk_ptErr':[-1.,300,100,],
'trk_etaErr':[-1,10,100,],
'trk_phiErr':[-1,10,100,],

'trk_px':[-200,200,100,],
'trk_py':[-200,200,100,],
'trk_pz':[-400,400,100,],

'trk_inner_px':[-200,200,100,],
'trk_inner_py':[-200,200,100,],
'trk_inner_pz':[-400,400,100,],
'trk_inner_pt':[-1.,300.,100],

'trk_outer_px':[-200,200,100,],
'trk_outer_py':[-200,200,100,],
'trk_outer_pz':[-400,400,100,],
'trk_outer_pt':[-1.,300.,100,],

'trk_dxyClosestPV':[-5,5,100],
'trk_dzClosestPV':[-10,10,100,],
'trk_dxy':[-5,5,100,],
'trk_dz':[-30,30,100,],

'trk_dxyErr':[-1,30,100,],
'trk_dzErr':[-1,30,100,],

'trk_nChi2':[-10,200.0,100],
'trk_ndof':[-10.0,100.0,111,],
'trk_nChi2_1Dmod':[-5,100,106],

'trk_nCluster':[-1,60,62],
'trk_nPixel':[-1,20,22,],
'trk_nStrip':[-1,30,32,],
'trk_nPixelLay':[-1,20,22,],
'trk_nStripLay':[-1,30,32,],
'trk_n3DLay':[-1,20,22,],
'trk_nInnerInactive':[-1,10,12,],
'trk_nOuterInactive':[-1,10,12,],
'trk_nInnerLost':[-1,20,22],
'trk_nOuterLost':[-1,20,22],
'trk_nValid':[-1,60,62],
'trk_nLostLay':[-1,10,12],
'trk_originalAlgo':[0,30,31],
'trk_mva':[-2,2,100,],
'trk_isTrue':[-1,2,8],

}


zoom_bins_dictionary={

'trk_pt':[-0.1,5.,100,],
'trk_ptErr':[-0.1,2,100,],
'trk_etaErr':[-0.1,1,100,],
'trk_phiErr':[-0.1,1,100,],

'trk_px':[-2,2,100,],
'trk_py':[-2,2,100,],
'trk_pz':[-10,10,100,],

'trk_inner_px':[-2,2,100,],
'trk_inner_py':[-2,2,100,],
'trk_inner_pz':[-10,10,100,],
'trk_inner_pt':[-0.1,5.,100],

'trk_outer_px':[-2,2,100,],
'trk_outer_py':[-2,2,100,],
'trk_outer_pz':[-10,10,100,],
'trk_outer_pt':[-0.1,5.,100,],

'trk_dxyClosestPV':[-0.2,0.2,100],
'trk_dzClosestPV':[-0.5,0.5,100,],
'trk_dxy':[-0.2,0.2,100,],

'trk_dxyErr':[-0.1,0.5,100,],
'trk_dzErr':[-0.1,.5,100,],
  
}

# just load the columns and labels
def load_to_pandas_plot(file_, columns):
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
            return dataframe
        except:
            return

def label(dataframe):
        dataframe.loc[:, "trk_isTrue"] = dataframe.loc[:, "trk_simTrkIdx"].apply(lambda x: 1 if len(x)>0  else 0)
        dataframe.drop(columns='trk_simTrkIdx') # this is not working - reassignment??
        return dataframe
      
def make_plots(column, df_tot, process, iteration=None, suffix=""):
        
        if (iteration!=None):
          df_tot=df_tot[df_tot["trk_originalAlgo"]==iteration]          
        
        data_col=df_tot[column]
        mybins=bins_dictionary[column]
        binning=np.linspace(mybins[0],mybins[1],mybins[2])

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

        if(column=="trk_mva"):
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
          
        if(column in zoom_bins_dictionary):

            mybins=zoom_bins_dictionary[column]
            binning=np.linspace(mybins[0],mybins[1],mybins[2])
            
            plt.hist(data_col, bins=binning, log=False, histtype='step', color="blue", label="all")
            plt.hist(data_col[df_tot["trk_isTrue"]==0], bins=binning, log=False, histtype='step', color="red", label="fake")
            plt.hist(data_col[df_tot["trk_isTrue"]==1], bins=binning, log=False, histtype='step', color="green", label="true")
            plt.legend(loc='upper right')          
            
            plt.title(column+" --- zoom")
            
            plt.savefig(path_to_save+process+"/fig/"+column+"/"+column+"__plot-ZOOM"+suffix+".png")
            plt.clf()
            
            plt.hist(data_col, bins=binning, log=True, histtype='step', color="blue", label="all")
            plt.hist(data_col[df_tot["trk_isTrue"]==0], bins=binning, log=True, histtype='step', color="red", label="fake")
            plt.hist(data_col[df_tot["trk_isTrue"]==1], bins=binning, log=True, histtype='step', color="green", label="true")
            plt.legend(loc='upper right')          
            
            plt.title(column+" --- zoom")
            
            plt.savefig(path_to_save+process+"/fig/"+column+"/"+column+"__plotLOG-ZOOM"+suffix+".png")
            plt.clf()

path_to_save='/data2/legianni/TrainingFrameworkTrackDNN/baseline-plots-125x-old/'

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
"TrainTTbar":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-CMSSW125X/TT_TuneCP5_14TeV-powheg-pythia8/crab_TT--fullCKFo/220905_224944/0000",
"TrainTTBS":"/data2/legianni/DNN-ntuples/CMSSW_12_5_0_pre5/src/validate2",
"TrainTTBS_v2ntuples":"/ceph/cms/store/user/legianni/tkNtuple-GeneralCKF-OldSamples-CMSSW125X-GTv2/TT_TuneCP5_14TeV-powheg-pythia8/crab_TT--fullCKFnn/220917_050726/0000",
"TrainTTBSmkFit_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-newsamples-CMSSW125X-GTv2/TT_TuneCP5_13p6TeV-powheg-pythia8/crab_TT--fullMKFITnnn/220921_213630/0000",
"RelValTTbarC":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-newsamples-CMSSW125X/RelValTTbar_14TeV/crab_RelValTT--fullCKFn/220914_200030/0000",
"TrainTTbar":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-newsamples-CMSSW125X/TT_TuneCP5_13p6TeV-powheg-pythia8/crab_TT--fullCKFn/220905_235423/0000",
"RelValTTbarM":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-newsamples-CMSSW125X/RelValTTbar_14TeV/crab_RelValTT--fullMKFITn/220914_200339/0000"
}

for process in ["RelValTTbarM", "RelValTTbarC"]:#for process in ['DisSUSY1', 'DisSUSY2', 'DisSUSY3', 'DisSUSY4', 'QCD1', 'TT', 'ZToEE1',]:
        files = glob(''+paths[process]+'/tr*root')[0:10]
        n = 4 
        n_bag = len(files)/n+1
        file_bags = np.array_split(files, n_bag)
        i=1
        df_tot_init=False
        for bag in file_bags:#[:-1]:
            print("running %s percent of the job" %str(float(i)/float(n_bag)*100))
            with ProcessPoolExecutor(max_workers=12) as executor:
                dfs = list(executor.map(load_to_pandas_plot, bag, repeat(columns_to_load)))
           
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
        
        bins=range(26)
        df_algo_t = df_tot.loc[df_tot["trk_isTrue"]>0.5,"trk_originalAlgo"]
        df_algo_f = df_tot.loc[df_tot["trk_isTrue"]<0.5,"trk_originalAlgo"]
        count_t, bngs_t = np.histogram(df_algo_t.values, bins=bins)
        count_f, bngs_f = np.histogram(df_algo_f.values, bins=bins)
        print ("fake tracks by iter number", count_f)
        print ("true tracks by iter number", count_t)

        for column in df_tot.columns:
          print (column)
          if  (column=="trk_simTrkIdx" or column=="trk_stopReason"): continue
          
          make_plots(column, df_tot, process)
          make_plots(column, df_tot, process, iteration=4, suffix="InitialStep")
          make_plots(column, df_tot, process, iteration=23, suffix="LowPtQuadStep")
          make_plots(column, df_tot, process, iteration=22, suffix="HighPtTripletStep")
          make_plots(column, df_tot, process, iteration=5, suffix="LowPtTripletStep")
          make_plots(column, df_tot, process, iteration=24, suffix="DetachedQuadStep")
          make_plots(column, df_tot, process, iteration=7, suffix="DetachedTripletStep")
          make_plots(column, df_tot, process, iteration=6, suffix="PixelPairStep")
          make_plots(column, df_tot, process, iteration=8, suffix="MixedTripletStep")
          make_plots(column, df_tot, process, iteration=9, suffix="PixelLessStep")
          make_plots(column, df_tot, process, iteration=10, suffix="TobTecStep")
          
          
