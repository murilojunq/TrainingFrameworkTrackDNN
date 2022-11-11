
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


3. revert updates flag updates

need to revert both 34662 and 34509 as mentioned in 
https://hypernews.cern.ch/HyperNews/CMS/get/tracker-performance/2461/1/1.html

```
git revert 8cd0049f2fbb6b7b215890b482c266978864efad
git cms-checkdeps -a -A
git revert 62ee6607a1959b4218def6bb952914ae2c8de999
git cms-checkdeps -a -A
```

+ adjustment on the flag



MODIFY DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h

'
 +    const uint32_t row_shift;
 +    const uint32_t column_shift;
-     const uint32_t flag_shift;
++    //const uint32_t flag_shift;
++    const uint32_t time_shift;
 +    const uint32_t adc_shift;
++    //const int row_shift;
++    //const int column_shift;
++    //const int time_shift;
++    //const int adc_shift;    
+ 
 -    const int row_shift;
 -    const int column_shift;
 -    const int time_shift;
 -    const int adc_shift;
'

git add DataFormats/SiPixelDetId/interface/PixelChannelIdentifier.h
git revert --continue
MODIFY DataFormats/SiPixelDigi/interface/PixelDigi.h

'
   explicit PixelDigi(PackedDigiType packed_value) : theData(packed_value) {}
 
-  PixelDigi(int row, int col, int adc) { init(row, col, adc); }
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
@@ -57,6 +58,9 @@ public:
            PixelChannelIdentifier::thePacking.column_mask;
   }
   //int time() const    {return (theData >> PixelChannelIdentifier::thePacking.time_shift) & PixelChannelIdentifier::thePacking.time_mask;}
+  int flag() const {
+     return 0;
+  }
   unsigned short adc() const {
     return (theData >> PixelChannelIdentifier::thePacking.adc_shift) & PixelChannelIdentifier::thePacking.adc_mask;
   }
'

scram b -j12
