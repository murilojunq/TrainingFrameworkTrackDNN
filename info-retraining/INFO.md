
## REPROCESSING

how to process 11_2_X samples under CMSSW_12_5_0_pre5

1. release

```
cmsrel CMSSW_12_5_0_pre5
cd CMSSW_12_5_0_pre5/src/
cmsenv
git cms-init
```
2. add DataFormats/SiPixelDetId (just for easier conflict solution)

```
git cms-addpkg DataFormats/SiPixelDetId
```



3. revert updates flag updates: you need to revert both 34662 and 34509 as mentioned in 
https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/2461/1/1.html

```
git revert 8cd0049f2fbb6b7b215890b482c266978864efad
git cms-checkdeps -a -A
git revert 62ee6607a1959b4218def6bb952914ae2c8de999
```


4. solve conflict by modifying DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h

solve this conflict 

<img width="552" alt="image" src="https://user-images.githubusercontent.com/7805577/201240787-36bb130c-1dbe-4865-b516-09f9c11f88ae.png">

by choosing a type and keeping the time shift. e.g.

<img width="269" alt="image" src="https://user-images.githubusercontent.com/7805577/201243171-4b2e3de7-3048-409a-8862-f54a33efa1f7.png">

and complete the second revert

```
git add DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h
git revert --continue
git cms-checkdeps -a -A
```

5. adjustments for compilation: Modify DataFormats/SiPixelDigi/interface/PixelDigi.h (NB: this is redundant)

```diff
@@ -18,16 +18,16 @@ public:
 
   explicit PixelDigi(PackedDigiType packed_value) : theData(packed_value) {}
 
-  PixelDigi(int row, int col, int adc) { init(row, col, adc); }
-
+  PixelDigi(int row, int col, int adc, int flag) { init(row, col, adc, flag); }
+  PixelDigi(int row, int col, int adc) { init(row, col, adc, 1); }
   PixelDigi(int chan, int adc) {
     std::pair<int, int> rc = channelToPixel(chan);
-    init(rc.first, rc.second, adc);
+    init(rc.first, rc.second, adc, 1);
   }
 
   PixelDigi() : theData(0) {}
 
-  void init(int row, int col, int adc) {
+  void init(int row, int col, int adc, int flag) {
 #ifdef FIXME_DEBUG
     // This check is for the maximal row or col number that can be packed
     // in a PixelDigi. The actual number of rows or columns in a detector
@@ -57,6 +57,9 @@ public:
            PixelChannelIdentifier::thePacking.column_mask;
   }
   //int time() const    {return (theData >> PixelChannelIdentifier::thePacking.time_shift) & PixelChannelIdentifier::thePacking.time_mask;}
+  int flag() const {
+    return 0;
+  }
   unsigned short adc() const {
     return (theData >> PixelChannelIdentifier::thePacking.adc_shift) & PixelChannelIdentifier::thePacking.adc_mask;
   }
```
6. compile
```
scram b -j12
```
7. setup for tk Ntuple production

```

```
8. prepare cmssw config and submit via crab (or other tool)

NB: keep the global tag "" to be consistent with the production beamspot, which was updated in 12_5_0_pre5
