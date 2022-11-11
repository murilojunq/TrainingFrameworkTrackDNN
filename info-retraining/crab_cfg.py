#from CRABClient.UserUtilities import config
#import CRABClient
import CRABClient
#from CRABClient.UserUtilities import config


from WMCore.Configuration import Configuration

def config():
    """
    Return a Configuration object containing all the sections that CRAB recognizes.
    """
    config = Configuration()  # pylint: disable=redefined-outer-name
    config.section_("General")
    config.section_("JobType")
    config.section_("Data")
    config.section_("Site")
    config.section_("User")
    config.section_("Debug")
    return config


datasets = {
            "QCD":"/QCD_Pt-15to7000_TuneCP5_Flat_13p6TeV-pythia8/Run3Winter22DR-PUForTRK_DIGI_122X_mcRun3_2021_realistic_v9-v2/GEN-SIM-DIGI-RAW",
            "SUSY":"/DisplacedSUSY_stopToBottom_M_1000_1000mm_TuneCP5_13p6TeV_pythia8/Run3Winter22DR-PUForTRK_DIGI_122X_mcRun3_2021_realistic_v9-v3/GEN-SIM-DIGI-RAW",
            "ZEE":"/ZToEE_TuneCP5_13p6TeV-pythia8/Run3Winter22DR-PUForTRK_DIGI_122X_mcRun3_2021_realistic_v9-v2/GEN-SIM-DIGI-RAW",
            "Tqqqq":"/SMS-T1qqqq-LLChipm_ctau-10cm_mGl-1800_mLSP-1400_TuneCP5_13p6TeV-madgraphMLM-pythia8/Run3Winter22DR-PUForTRK_DIGI_122X_mcRun3_2021_realistic_v9-v2/GEN-SIM-DIGI-RAW",
            "TT":"/TT_TuneCP5_13p6TeV-powheg-pythia8/Run3Winter22DR-PUForTRK_DIGI_122X_mcRun3_2021_realistic_v9-v2/GEN-SIM-DIGI-RAW",
            "SoftQCD":"/SoftQCD_XiFilter_TuneCP5_13p6TeV_pythia8/legianni-crab_softQCD--step2-ad824e777a2aa65d833e9704a3bad29f/USER"
           }


oldda={"TT":"/TT_TuneCP5_14TeV-powheg-pythia8/Run3Winter21DRMiniAOD-FlatPU20to70_for_DNN_112X_mcRun3_2021_realistic_v16_ext1-v2/GEN-SIM-DIGI-RAW",
      "ZEE":"/ZToEE_TuneCUETP8M1_14TeV-pythia8/Run3Winter21DRMiniAOD-FlatPU20to70_for_DNN_112X_mcRun3_2021_realistic_v16_ext1-v2/GEN-SIM-DIGI-RAW",
      "QCD":"/QCD_Pt-15to7000_TuneCUETP8M1_Flat_14TeV-pythia8/Run3Winter21DRMiniAOD-FlatPU20to70_for_DNN_112X_mcRun3_2021_realistic_v16_ext1-v2/GEN-SIM-DIGI-RAW",
       "DisSUSY180":"/DisplacedSUSY_stopToBottom_M_1800_100mm_TuneCP5_14TeV_pythia8/Run3Winter21DRMiniAOD-FlatPU20to70_112X_mcRun3_2021_realistic_v16-v3/GEN-SIM-DIGI-RAW",
       "DisSUSY11":"/DisplacedSUSY_stopToBottom_M_1000_1000mm_TuneCP5_14TeV_pythia8/Run3Winter21DRMiniAOD-FlatPU20to70_112X_mcRun3_2021_realistic_v16-v3/GEN-SIM-DIGI-RAW",
       "DisSUSY181":"/DisplacedSUSY_stopToBottom_M_1800_1000mm_TuneCP5_14TeV_pythia8/Run3Winter21DRMiniAOD-FlatPU20to70_112X_mcRun3_2021_realistic_v16-v3/GEN-SIM-DIGI-RAW",
       "DisSUSY10":"/DisplacedSUSY_stopToBottom_M_1000_100mm_TuneCP5_14TeV_pythia8/Run3Winter21DRMiniAOD-FlatPU20to70_112X_mcRun3_2021_realistic_v16-v3/GEN-SIM-DIGI-RAW",
      }

config = config()
config.General.workArea        = '../../crab_Winter22_trackingNtuple_v2'
config.General.transferOutputs = True
config.General.transferLogs    = False
config.Data.ignoreLocality  = True
config.JobType.sendExternalFolder = True
config.JobType.pluginName  = 'Analysis'
config.JobType.psetName    = 'trackNtuple_producer.py'
config.JobType.numCores    = 4
config.JobType.maxMemoryMB = 8000
config.Data.splitting   = 'EventAwareLumiBased'
config.Data.unitsPerJob = 40#20
config.Data.totalUnits  = -1#40
config.Data.outLFNDirBase = '/store/user/legianni/tkNtuple-General-MKFIT-newsamples-CMSSW125X-GTv2/'
config.JobType.outputFiles  = ['trackingNtuple.root']
config.JobType.allowUndistributedCMSSW=True
config.section_('Site')
config.Site.storageSite = 'T2_US_UCSD'
config.section_('Debug')
config.Debug.extraJDL = ['+CMS_ALLOW_OVERFLOW=False']
config.Site.whitelist = ['T2_US_*']

key="SUSY"
#key="TT"
#key="Tqqqq"
#key="QCD"
#key="ZEE"
#key="SoftQCD"

#key="RelValQCDH"

#key="TT"
#key="QCD"
#key="ZEE"
#key="DisSUSY180"
#key="DisSUSY11"
#key="DisSUSY181"
#key="DisSUSY10"

if (key=="SoftQCD"):
  config.Data.inputDBS = 'phys03'
#config.Data.inputDataset = oldda[key]
config.Data.inputDataset = datasets[key]
#config.Data.inputDataset = '/RelValQCD_Pt_1800_2400_14/CMSSW_12_5_0_pre5-PU_125X_mcRun3_2022_realistic_v3-v1/GEN-SIM-DIGI-RAW'
#'/RelValTTbar_14TeV/CMSSW_12_5_0_pre5-PU_125X_mcRun3_2022_realistic_v3-v1/GEN-SIM-DIGI-RAW'
config.General.requestName     =  key+"--fullMKFITni"

#/ceph/cms/store/user/legianni/tkNtuple-General-CKF-newsamples-CMSSW125X/RelValTTbar_14TeV:
#crab_RelValTT--fullCKFn

#/ceph/cms/store/user/legianni/tkNtuple-General-MKFIT-newsamples-CMSSW125X/RelValTTbar_14TeV:
#crab_RelValTT--fullMKFITn

#if __name__ == '__main__':

    #from CRABAPI.RawCommand import crabCommand
    #from CRABClient.ClientExceptions import ClientException
    #from httplib import HTTPException
    #for key in datasets.keys():

        #config.Data.inputDataset = datasets[key]
        #config.General.requestName     =  key
        #crabCommand('submit', config = config)

