import pandas as pd
import numpy as np
import awkward as ak
import tensorflow as tf
import uproot, os

from glob import glob
from json import dumps
from itertools import repeat
from concurrent.futures import ProcessPoolExecutor

'''
tool to convert a ROOT tkNtuple into a dataframe with track features and then a serialized tfRecord
'''

# load these features (inclusive of DNN input and label)
# the only useful thing done with extra features is the label
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
            #dataframe[ "trk_inner_phi"] = np.arctan2(dataframe["trk_inner_py"],dataframe["trk_inner_px"])
            #dataframe["trk_outer_phi"] = np.arctan2(dataframe["trk_outer_py"],dataframe["trk_outer_px"])
            #dataframe["trk_inner_eta"] = np.arcsinh(dataframe["trk_inner_pz"]/dataframe["trk_inner_pt"])
            #dataframe["trk_outer_eta"] = np.arcsinh(dataframe["trk_outer_pz"]/dataframe["trk_outer_pt"])
            #print (dataframe.shape, "ddddd")         
            dataframe = dataframe.loc[:, columns_of_regular_inputs+["trk_originalAlgo"]+["trk_isTrue"]].astype('float32')
            #cleanup
            dataframe = dataframe[dataframe["trk_originalAlgo"]!=13]
            dataframe = dataframe[dataframe["trk_originalAlgo"]!=14]
            dataframe = dataframe[dataframe["trk_nStrip"]>=0]
            dataframe.dropna(inplace=True)
            print (dataframe.shape, "post cleanup")
            #dataframe = select_iterations(dataframe,[4])
            return dataframe
        except:
            return 
          
# not used yet
# but can be called inside load_to_pandas
# select a few iterations
def select_iterations(dataframe, iterations=[4,5,6]):          
        dataframe = dataframe[dataframe["trk_originalAlgo"].isin(iterations)]
        return dataframe
      
# oversample (to the dim of the larger class)
def oversampling(dataframe):
        dataframeT = dataframe[dataframe["trk_isTrue"]>0.5]
        dataframeF = dataframe[dataframe["trk_isTrue"]<0.5]
        if (len(dataframeT)>len(dataframeF)):
              dataframeF = dataframeF.sample(len(dataframeT), replace=True)
        else:        
              dataframeF = dataframeF.sample(len(dataframeT), replace=True)
        dataframe = pd.concat([dataframeT,dataframeF])
        return dataframe  
      
# undersample (to the dim of the smaller class)
def undersampling(dataframe):
        dataframeT = dataframe[dataframe["trk_isTrue"]>0.5]
        dataframeF = dataframe[dataframe["trk_isTrue"]<0.5]
        if (len(dataframeT)>len(dataframeF)):
              dataframeT = dataframeT.sample(len(dataframeF))
        else:        
              dataframeF = dataframeF.sample(len(dataframeT))
        dataframe = pd.concat([dataframeT,dataframeF])
        return dataframe  
      
### can be useful later on

def get_stats(df):
        min_, max_ = df.agg([min, max]).values
        bins=range(26)
        df_algo_t = df.loc[df["trk_isTrue"]>0.5,"trk_originalAlgo"]
        df_algo_f = df.loc[df["trk_isTrue"]<0.5,"trk_originalAlgo"]
        count_t, bngs = np.histogram(df_algo_t.values, bins=bins)
        count_f, bngs = np.histogram(df_algo_f.values, bins=bins)
        return (count_t, count_f, min_, max_)

def save_to_tfRecord(df, path, name):
        dataset = tf.data.Dataset.from_tensor_slices(df)
        file_writer = tf.data.experimental.TFRecordWriter(path+name) 
        dataset = dataset.map(tf.io.serialize_tensor)
        file_writer.write(dataset)

def label(dataframe):
        dataframe.loc[:, "trk_isTrue"] = dataframe.loc[:, "trk_simTrkIdx"].apply(lambda x: 1 if len(x)>0  else 0)
        dataframe.drop(columns='trk_simTrkIdx')
        return dataframe

def to_input(x):
    x = tf.io.parse_tensor(x, tf.float32)
    return ({'regular_input_layer':x[:29],'categorical_input_layer':x[-2]}, x[-1], 1.0) 

path_to_save='/ceph/cms/store/user/legianni/tfrecordNEW-mkfit-125_v2/'#'/data2/legianni/TrainingFrameworkTrackDNN/tfrecord2-benchmark/'

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

paths_all={
"DisSUSY-newCKF":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-newsamples-CMSSW125X/DisplacedSUSY_stopToBottom_M_1000_1000mm_TuneCP5_13p6TeV_pythia8/crab_SUSY--fullCKFn/220905_235011/0000",
"QCD-newCKF":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-newsamples-CMSSW125X/QCD_Pt-15to7000_TuneCP5_Flat_13p6TeV-pythia8/crab_QCD--fullCKFn/220905_235819/0000",
"SMS-newCKF":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-newsamples-CMSSW125X/SMS-T1qqqq-LLChipm_ctau-10cm_mGl-1800_mLSP-1400_TuneCP5_13p6TeV-madgraphMLM-pythia8/crab_Tqqqq--fullCKFn/220905_235624/0000",
"SoftQCD-newCKF":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-newsamples-CMSSW125X/SoftQCD_XiFilter_TuneCP5_13p6TeV_pythia8/crab_SoftQCD--fullCKFn/220906_000340/0000",
"TT-newCKF":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-newsamples-CMSSW125X/TT_TuneCP5_13p6TeV-powheg-pythia8/crab_TT--fullCKFn/220905_235423/0000",
"ZToEE-newCKF":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-newsamples-CMSSW125X/ZToEE_TuneCP5_13p6TeV-pythia8/crab_ZEE--fullCKFn/220906_000047/0000",
"DisSUSY1":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-CMSSW125X/DisplacedSUSY_stopToBottom_M_1000_1000mm_TuneCP5_14TeV_pythia8/crab_DisSUSY11--fullCKFo/220905_224320/0000",
"DisSUSY2":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-CMSSW125X/DisplacedSUSY_stopToBottom_M_1000_100mm_TuneCP5_14TeV_pythia8/crab_DisSUSY10--fullCKFo/220905_224006/0000",
"DisSUSY3":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-CMSSW125X/DisplacedSUSY_stopToBottom_M_1800_1000mm_TuneCP5_14TeV_pythia8/crab_DisSUSY181--fullCKFo/220905_224139/0000",
"DisSUSY4":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-CMSSW125X/DisplacedSUSY_stopToBottom_M_1800_100mm_TuneCP5_14TeV_pythia8/crab_DisSUSY180--fullCKFo/220905_224459/0000",
"QCD1":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-CMSSW125X/QCD_Pt-15to7000_TuneCUETP8M1_Flat_14TeV-pythia8/crab_QCD--fullCKFo/220905_224807/0000",
"QCD2":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-CMSSW125X/QCD_Pt-15to7000_TuneCUETP8M1_Flat_14TeV-pythia8/crab_QCD--fullCKFo/220905_224807/0001",
"TT":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-CMSSW125X/TT_TuneCP5_14TeV-powheg-pythia8/crab_TT--fullCKFo/220905_224944/0000",
"ZToEE1":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-CMSSW125X/ZToEE_TuneCUETP8M1_14TeV-pythia8/crab_ZEE--fullCKFo/220905_224631/0000",
"ZToEE2":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-oldsamples-CMSSW125X/ZToEE_TuneCUETP8M1_14TeV-pythia8/crab_ZEE--fullCKFo/220905_224631/0001",
"DisSUSY-newMK":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-newsamples-CMSSW125X/DisplacedSUSY_stopToBottom_M_1000_1000mm_TuneCP5_13p6TeV_pythia8/crab_SUSY--fullMKFITn/220905_234446/0000",
"QCD-newMK":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-newsamples-CMSSW125X/QCD_Pt-15to7000_TuneCP5_Flat_13p6TeV-pythia8/crab_QCD--fullMKFITn/220905_233538/0000",
"SMS-newMK":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-newsamples-CMSSW125X/SMS-T1qqqq-LLChipm_ctau-10cm_mGl-1800_mLSP-1400_TuneCP5_13p6TeV-madgraphMLM-pythia8/crab_Tqqqq--fullMKFITn/220905_233957/0000",
"SoftQCD-newMK":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-newsamples-CMSSW125X/SoftQCD_XiFilter_TuneCP5_13p6TeV_pythia8/crab_SoftQCD--fullMKFITn/220905_233148/0000",
"TT-newMK":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-newsamples-CMSSW125X/TT_TuneCP5_13p6TeV-powheg-pythia8/crab_TT--fullMKFITn/220905_234231/0000",
"ZToEE-newMK":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-newsamples-CMSSW125X/ZToEE_TuneCP5_13p6TeV-pythia8/crab_ZEE--fullMKFITn/220905_233322/0000",
"DisSUSY1-MK":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-oldsamples-CMSSW125X/DisplacedSUSY_stopToBottom_M_1000_1000mm_TuneCP5_14TeV_pythia8/crab_DisSUSY11--fullMKFITo/220905_231535/0000",
"DisSUSY2-MK":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-oldsamples-CMSSW125X/DisplacedSUSY_stopToBottom_M_1000_100mm_TuneCP5_14TeV_pythia8/crab_DisSUSY10--fullMKFITo/220905_231920/0000",
"DisSUSY3-MK":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-oldsamples-CMSSW125X/DisplacedSUSY_stopToBottom_M_1800_1000mm_TuneCP5_14TeV_pythia8/crab_DisSUSY181--fullMKFITo/220905_231717/0000",
"DisSUSY4-MK":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-oldsamples-CMSSW125X/DisplacedSUSY_stopToBottom_M_1800_100mm_TuneCP5_14TeV_pythia8/crab_DisSUSY180--fullMKFITo/220905_231256/0000",
"QCD1-MK":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-oldsamples-CMSSW125X/QCD_Pt-15to7000_TuneCUETP8M1_Flat_14TeV-pythia8/crab_QCD--fullMKFITo/220905_230459/0000",
"QCD2-MK":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-oldsamples-CMSSW125X/QCD_Pt-15to7000_TuneCUETP8M1_Flat_14TeV-pythia8/crab_QCD--fullMKFITo/220905_230459/0001",
"TT-MK":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-oldsamples-CMSSW125X/TT_TuneCP5_14TeV-powheg-pythia8/crab_TT--fullMKFITo/220905_230156/0000",
"ZToEE1-MK":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-oldsamples-CMSSW125X/ZToEE_TuneCUETP8M1_14TeV-pythia8/crab_ZEE--fullMKFITo/220905_230840/0000",
"ZToEE2-MK":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-oldsamples-CMSSW125X/ZToEE_TuneCUETP8M1_14TeV-pythia8/crab_ZEE--fullMKFITo/220905_230840/0001",
"DisSUSY1_v2":"/ceph/cms/store/user/legianni/tkNtuple-GeneralCKF-OldSamples-CMSSW125X-GTv2/DisplacedSUSY_stopToBottom_M_1000_1000mm_TuneCP5_14TeV_pythia8/crab_DisSUSY11--fullCKFnn/220917_045828/0000",
"DisSUSY2_v2":"/ceph/cms/store/user/legianni/tkNtuple-GeneralCKF-OldSamples-CMSSW125X-GTv2/DisplacedSUSY_stopToBottom_M_1000_100mm_TuneCP5_14TeV_pythia8/crab_DisSUSY10--fullCKFnn/220917_045305/0000",
"DisSUSY3_v2":"/ceph/cms/store/user/legianni/tkNtuple-GeneralCKF-OldSamples-CMSSW125X-GTv2/DisplacedSUSY_stopToBottom_M_1800_1000mm_TuneCP5_14TeV_pythia8/crab_DisSUSY181--fullCKFnn/220917_045633/0000",
"DisSUSY4_v2":"/ceph/cms/store/user/legianni/tkNtuple-GeneralCKF-OldSamples-CMSSW125X-GTv2/DisplacedSUSY_stopToBottom_M_1800_100mm_TuneCP5_14TeV_pythia8/crab_DisSUSY180--fullCKFnn/220917_050011/0000",
"QCD1_v2":"/ceph/cms/store/user/legianni/tkNtuple-GeneralCKF-OldSamples-CMSSW125X-GTv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_14TeV-pythia8/crab_QCD--fullCKFnn/220917_050549/0000",
"QCD2_v2":"/ceph/cms/store/user/legianni/tkNtuple-GeneralCKF-OldSamples-CMSSW125X-GTv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_14TeV-pythia8/crab_QCD--fullCKFnn/220917_050549/0001",
"TT_v2":"/ceph/cms/store/user/legianni/tkNtuple-GeneralCKF-OldSamples-CMSSW125X-GTv2/TT_TuneCP5_14TeV-powheg-pythia8/crab_TT--fullCKFnn/220917_050726/0000",
"ZToEE1_v2":"/ceph/cms/store/user/legianni/tkNtuple-GeneralCKF-OldSamples-CMSSW125X-GTv2/ZToEE_TuneCUETP8M1_14TeV-pythia8/crab_ZEE--fullCKFnn/220917_050226/0000",
"ZToEE2_v2":"/ceph/cms/store/user/legianni/tkNtuple-GeneralCKF-OldSamples-CMSSW125X-GTv2/ZToEE_TuneCUETP8M1_14TeV-pythia8/crab_ZEE--fullCKFnn/220917_050226/0001",
"DisSUSY1-MK_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-oldsamples-CMSSW125X-GTv2/DisplacedSUSY_stopToBottom_M_1000_1000mm_TuneCP5_14TeV_pythia8/crab_DisSUSY11--fullMKFITnn/220918_185510/0000",
"DisSUSY2-MK_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-oldsamples-CMSSW125X-GTv2/DisplacedSUSY_stopToBottom_M_1000_100mm_TuneCP5_14TeV_pythia8/crab_DisSUSY10--fullMKFITnn/220918_185902/0000",
"DisSUSY3-MK_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-oldsamples-CMSSW125X-GTv2/DisplacedSUSY_stopToBottom_M_1800_1000mm_TuneCP5_14TeV_pythia8/crab_DisSUSY181--fullMKFITnn/220918_185647/0000",
"DisSUSY4-MK_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-oldsamples-CMSSW125X-GTv2/DisplacedSUSY_stopToBottom_M_1800_100mm_TuneCP5_14TeV_pythia8/crab_DisSUSY180--fullMKFITnn/220918_185233/0000",
"QCD1-MK_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-oldsamples-CMSSW125X-GTv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_14TeV-pythia8/crab_QCD--fullMKFITnn/220918_184842/0000",
"QCD2-MK_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-oldsamples-CMSSW125X-GTv2/QCD_Pt-15to7000_TuneCUETP8M1_Flat_14TeV-pythia8/crab_QCD--fullMKFITnn/220918_184842/0001",
"TT-MK_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-oldsamples-CMSSW125X-GTv2/TT_TuneCP5_14TeV-powheg-pythia8/crab_TT--fullMKFITnn/220918_183819/0000",
"ZToEE1-MK_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-oldsamples-CMSSW125X-GTv2/ZToEE_TuneCUETP8M1_14TeV-pythia8/crab_ZEE--fullMKFITnn/220918_185039/0000",
"ZToEE2-MK_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-oldsamples-CMSSW125X-GTv2/ZToEE_TuneCUETP8M1_14TeV-pythia8/crab_ZEE--fullMKFITnn/220918_185039/0001",

"DisSUSY-newCKF_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-newsamples-CMSSW125X-GTv2/DisplacedSUSY_stopToBottom_M_1000_1000mm_TuneCP5_13p6TeV_pythia8/crab_SUSY--fullCKFnnn/220921_214116/0000",
"QCD-newCKF_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-newsamples-CMSSW125X-GTv2/QCD_Pt-15to7000_TuneCP5_Flat_13p6TeV-pythia8/crab_QCD--fullCKFnnn/220921_214620/0000",
"SMS-newCKF_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-newsamples-CMSSW125X-GTv2/SMS-T1qqqq-LLChipm_ctau-10cm_mGl-1800_mLSP-1400_TuneCP5_13p6TeV-madgraphMLM-pythia8/crab_Tqqqq--fullCKFnnn/220921_214443/0000",
"SoftQCD-newCKF_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-newsamples-CMSSW125X-GTv2/SoftQCD_XiFilter_TuneCP5_13p6TeV_pythia8/crab_SoftQCD--fullCKFnnn/220921_215108/0000",
"TT-newCKF_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-newsamples-CMSSW125X-GTv2/TT_TuneCP5_13p6TeV-powheg-pythia8/crab_TT--fullCKFnnn/220921_214258/0000",
"ZToEE-newCKF_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-CKF-newsamples-CMSSW125X-GTv2/ZToEE_TuneCP5_13p6TeV-pythia8/crab_ZEE--fullCKFnnn/220921_214928/0000",
"DisSUSY-newMK_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-newsamples-CMSSW125X-GTv2/DisplacedSUSY_stopToBottom_M_1000_1000mm_TuneCP5_13p6TeV_pythia8/crab_SUSY--fullMKFITni/221003_234304/0000",
"QCD-newMK_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-newsamples-CMSSW125X-GTv2/QCD_Pt-15to7000_TuneCP5_Flat_13p6TeV-pythia8/crab_QCD--fullMKFITni/221003_233720/0000",
"SMS-newMK_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-newsamples-CMSSW125X-GTv2/SMS-T1qqqq-LLChipm_ctau-10cm_mGl-1800_mLSP-1400_TuneCP5_13p6TeV-madgraphMLM-pythia8/crab_Tqqqq--fullMKFITni/221003_233857/0000",
"SoftQCD-newMK_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-newsamples-CMSSW125X-GTv2/SoftQCD_XiFilter_TuneCP5_13p6TeV_pythia8/crab_SoftQCD--fullMKFITni/221003_233144/0000",
"TT-newMK_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-newsamples-CMSSW125X-GTv2/TT_TuneCP5_13p6TeV-powheg-pythia8/crab_TT--fullMKFITni/221003_234040/0000",
"ZToEE-newMK_v2":"/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-newsamples-CMSSW125X-GTv2/ZToEE_TuneCP5_13p6TeV-pythia8/crab_ZEE--fullMKFITni/221003_233323/0000",
}

newCKF=['SoftQCD-newCKF', 'DisSUSY-newCKF', 'SMS-newCKF', 'TT-newCKF', 'QCD-newCKF', 'ZToEE-newCKF']
newMK=['SoftQCD-newMK',  'DisSUSY-newMK',  'SMS-newMK',  'TT-newMK',  'QCD-newMK',  'ZToEE-newMK']
oldCKF=['DisSUSY1', 'DisSUSY2','DisSUSY3','DisSUSY4', 'QCD1', 'QCD2','TT', 'ZToEE1', 'ZToEE2'] 
oldMK=['DisSUSY1-MK', 'DisSUSY2-MK', 'DisSUSY3-MK', 'DisSUSY4-MK','QCD1-MK','QCD2-MK','TT-MK','ZToEE1-MK','ZToEE2-MK']
oldCKF=['DisSUSY1_v2', 'DisSUSY2_v2','DisSUSY3_v2','DisSUSY4_v2', 'QCD1_v2', 'QCD2_v2','TT_v2', 'ZToEE1_v2', 'ZToEE2_v2']
oldMK=['DisSUSY1-MK_v2', 'DisSUSY2-MK_v2', 'DisSUSY3-MK_v2', 'DisSUSY4-MK_v2','QCD1-MK_v2','QCD2-MK_v2','TT-MK_v2','ZToEE1-MK_v2','ZToEE2-MK_v2']
newCKF=['SoftQCD-newCKF_v2', 'DisSUSY-newCKF_v2', 'SMS-newCKF_v2', 'TT-newCKF_v2', 'QCD-newCKF_v2', 'ZToEE-newCKF_v2']
newMK=['SoftQCD-newMK_v2']#,  'DisSUSY-newMK_v2',  'SMS-newMK_v2',  'TT-newMK_v2',  'QCD-newMK_v2',  'ZToEE-newMK_v2']
# ntuplize and save tf records by process
# full list is #['DisSUSY1', 'DisSUSY2', 'DisSUSY3', 'DisSUSY4', 'QCD1', 'QCD2', 'TT', 'ZToEE1', 'ZToEE2']:
for process in newMK: #oldMK:#['TT']:   
        files = glob(''+paths_all[process]+'/tr*root')#[0:20]
        n = 4 
        n_bag = len(files)/n+1
        file_bags = np.array_split(files, n_bag)
        stats = {}
        i=1
        for bag in file_bags[:-1]:
            print("running %s percent of the job" %str(float(i)/float(n_bag)*100))
            with ProcessPoolExecutor(max_workers=12) as executor:
                dfs = list(executor.map(load_to_pandas, bag, repeat(columns_to_load), repeat(columns_of_regular_inputs)))
            if len(dfs)>1:df = dfs[0].append(dfs[1:])
            else: df = dfs[0]
            stat = get_stats(df)
            if i == 1:
                stats["ntrk_t"] = stat[0]
                stats["ntrk_f"] = stat[1]  
                stats["min"] = stat[2]
                stats["max"] = stat[3]
            else:
                stats["ntrk_t"] += stat[0]
                stats["ntrk_f"] += stat[1]      
                stats["min"] = np.minimum(stat[2], stats["min"])
                stats["max"] = np.maximum(stat[3], stats["max"])
            if not os.path.isdir(path_to_save+process+"/train/"):
                os.makedirs(path_to_save+process+"/train/")
     
            save_to_tfRecord(df, path_to_save+process+"/train/", process+str(i)+".tfrecord")
            print("finished %s percent of the job" %str(float(i)/float(n_bag)*100))
            i+=1

        stats["ntrk_t"] = stats["ntrk_t"].tolist()
        stats["ntrk_f"] = stats["ntrk_f"].tolist()
        stats["min"] = stats["min"].tolist()
        stats["max"] = stats["max"].tolist()
        with open(path_to_save+process+".txt", "a") as summary:   
                summary.write(dumps(stats))
        
