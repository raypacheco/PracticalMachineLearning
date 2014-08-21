##Practical Machine Learning: Human Activity Recognition

###Introduction
Devices such as the Fitbit, Jawbone Up, and Nike FuelBank now make it possible to collect a large amount of data about personal activity. Currently, however, these devices quantify only how much of a particular activity is done, but rarely how well the activity is done.

This analysis uses data gathered by Velloso, E., et. al. in their research: [Qualitative Activity Recognition of Weight Lifting Exercises](http://groupware.les.inf.puc-rio.br/har). In this study, six participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E). 

For the purposes of our project, we will explore using Random Forests to predict each class based on several of the variables in the dataset.

###Data Exploration and Cleaning

We need to load first load and subset the data. We will use the Caret package for this analysis.


```r
library(caret)

train <- read.csv("pml-training.csv", header=TRUE)
## Earlier exploration shows the variable new_window with "yes" contain only summary statistics.
train <- train[train$new_window == "no",]

summary(train)
```

```
##        X            user_name    raw_timestamp_part_1 raw_timestamp_part_2
##  Min.   :    1   adelmo  :3809   Min.   :1.32e+09     Min.   :   294      
##  1st Qu.: 4899   carlitos:3056   1st Qu.:1.32e+09     1st Qu.:248312      
##  Median : 9804   charles :3455   Median :1.32e+09     Median :488295      
##  Mean   : 9806   eurico  :3016   Mean   :1.32e+09     Mean   :490704      
##  3rd Qu.:14708   jeremy  :3325   3rd Qu.:1.32e+09     3rd Qu.:735084      
##  Max.   :19621   pedro   :2555   Max.   :1.32e+09     Max.   :998697      
##                                                                           
##           cvtd_timestamp  new_window    num_window    roll_belt    
##  28/11/2011 14:14: 1473   no :19216   Min.   :  1   Min.   :-28.9  
##  05/12/2011 11:24: 1470   yes:    0   1st Qu.:221   1st Qu.:  1.1  
##  30/11/2011 17:11: 1404               Median :423   Median :113.0  
##  05/12/2011 11:25: 1400               Mean   :430   Mean   : 64.3  
##  02/12/2011 13:34: 1349               3rd Qu.:644   3rd Qu.:123.0  
##  02/12/2011 14:57: 1348               Max.   :864   Max.   :162.0  
##  (Other)         :10772                                            
##    pitch_belt        yaw_belt      total_accel_belt kurtosis_roll_belt
##  Min.   :-55.80   Min.   :-180.0   Min.   : 0.0              :19216   
##  1st Qu.:  1.77   1st Qu.: -88.3   1st Qu.: 3.0     -0.016850:    0   
##  Median :  5.28   Median : -13.2   Median :17.0     -0.021024:    0   
##  Mean   :  0.30   Mean   : -11.3   Mean   :11.3     -0.025513:    0   
##  3rd Qu.: 14.90   3rd Qu.:  12.8   3rd Qu.:18.0     -0.033935:    0   
##  Max.   : 60.30   Max.   : 179.0   Max.   :29.0     -0.034743:    0   
##                                                     (Other)  :    0   
##  kurtosis_picth_belt kurtosis_yaw_belt skewness_roll_belt
##           :19216            :19216              :19216   
##  -0.021887:    0     #DIV/0!:    0     -0.003095:    0   
##  -0.060755:    0                       -0.010002:    0   
##  -0.099173:    0                       -0.014020:    0   
##  -0.108371:    0                       -0.015465:    0   
##  -0.109078:    0                       -0.024325:    0   
##  (Other)  :    0                       (Other)  :    0   
##  skewness_roll_belt.1 skewness_yaw_belt max_roll_belt   max_picth_belt 
##           :19216             :19216     Min.   : NA     Min.   : NA    
##  -0.005928:    0      #DIV/0!:    0     1st Qu.: NA     1st Qu.: NA    
##  -0.005960:    0                        Median : NA     Median : NA    
##  -0.008391:    0                        Mean   :NaN     Mean   :NaN    
##  -0.017954:    0                        3rd Qu.: NA     3rd Qu.: NA    
##  -0.038884:    0                        Max.   : NA     Max.   : NA    
##  (Other)  :    0                        NA's   :19216   NA's   :19216  
##   max_yaw_belt   min_roll_belt   min_pitch_belt   min_yaw_belt  
##         :19216   Min.   : NA     Min.   : NA            :19216  
##  -0.1   :    0   1st Qu.: NA     1st Qu.: NA     -0.1   :    0  
##  -0.2   :    0   Median : NA     Median : NA     -0.2   :    0  
##  -0.3   :    0   Mean   :NaN     Mean   :NaN     -0.3   :    0  
##  -0.4   :    0   3rd Qu.: NA     3rd Qu.: NA     -0.4   :    0  
##  -0.5   :    0   Max.   : NA     Max.   : NA     -0.5   :    0  
##  (Other):    0   NA's   :19216   NA's   :19216   (Other):    0  
##  amplitude_roll_belt amplitude_pitch_belt amplitude_yaw_belt
##  Min.   : NA         Min.   : NA                 :19216     
##  1st Qu.: NA         1st Qu.: NA          #DIV/0!:    0     
##  Median : NA         Median : NA          0.00   :    0     
##  Mean   :NaN         Mean   :NaN          0.0000 :    0     
##  3rd Qu.: NA         3rd Qu.: NA                            
##  Max.   : NA         Max.   : NA                            
##  NA's   :19216       NA's   :19216                          
##  var_total_accel_belt avg_roll_belt   stddev_roll_belt var_roll_belt  
##  Min.   : NA          Min.   : NA     Min.   : NA      Min.   : NA    
##  1st Qu.: NA          1st Qu.: NA     1st Qu.: NA      1st Qu.: NA    
##  Median : NA          Median : NA     Median : NA      Median : NA    
##  Mean   :NaN          Mean   :NaN     Mean   :NaN      Mean   :NaN    
##  3rd Qu.: NA          3rd Qu.: NA     3rd Qu.: NA      3rd Qu.: NA    
##  Max.   : NA          Max.   : NA     Max.   : NA      Max.   : NA    
##  NA's   :19216        NA's   :19216   NA's   :19216    NA's   :19216  
##  avg_pitch_belt  stddev_pitch_belt var_pitch_belt   avg_yaw_belt  
##  Min.   : NA     Min.   : NA       Min.   : NA     Min.   : NA    
##  1st Qu.: NA     1st Qu.: NA       1st Qu.: NA     1st Qu.: NA    
##  Median : NA     Median : NA       Median : NA     Median : NA    
##  Mean   :NaN     Mean   :NaN       Mean   :NaN     Mean   :NaN    
##  3rd Qu.: NA     3rd Qu.: NA       3rd Qu.: NA     3rd Qu.: NA    
##  Max.   : NA     Max.   : NA       Max.   : NA     Max.   : NA    
##  NA's   :19216   NA's   :19216     NA's   :19216   NA's   :19216  
##  stddev_yaw_belt  var_yaw_belt    gyros_belt_x      gyros_belt_y    
##  Min.   : NA     Min.   : NA     Min.   :-1.0400   Min.   :-0.6400  
##  1st Qu.: NA     1st Qu.: NA     1st Qu.:-0.0300   1st Qu.: 0.0000  
##  Median : NA     Median : NA     Median : 0.0300   Median : 0.0200  
##  Mean   :NaN     Mean   :NaN     Mean   :-0.0056   Mean   : 0.0395  
##  3rd Qu.: NA     3rd Qu.: NA     3rd Qu.: 0.1100   3rd Qu.: 0.1100  
##  Max.   : NA     Max.   : NA     Max.   : 2.2200   Max.   : 0.6400  
##  NA's   :19216   NA's   :19216                                      
##   gyros_belt_z     accel_belt_x     accel_belt_y    accel_belt_z   
##  Min.   :-1.460   Min.   :-120.0   Min.   :-69.0   Min.   :-275.0  
##  1st Qu.:-0.200   1st Qu.: -21.0   1st Qu.:  3.0   1st Qu.:-162.0  
##  Median :-0.100   Median : -15.0   Median : 34.0   Median :-152.0  
##  Mean   :-0.131   Mean   :  -5.6   Mean   : 30.1   Mean   : -72.5  
##  3rd Qu.:-0.020   3rd Qu.:  -5.0   3rd Qu.: 61.0   3rd Qu.:  27.0  
##  Max.   : 1.620   Max.   :  85.0   Max.   :164.0   Max.   : 105.0  
##                                                                    
##  magnet_belt_x   magnet_belt_y magnet_belt_z     roll_arm     
##  Min.   :-52.0   Min.   :354   Min.   :-623   Min.   :-180.0  
##  1st Qu.:  9.0   1st Qu.:581   1st Qu.:-375   1st Qu.: -31.6  
##  Median : 35.0   Median :601   Median :-320   Median :   0.0  
##  Mean   : 55.6   Mean   :594   Mean   :-346   Mean   :  17.9  
##  3rd Qu.: 59.0   3rd Qu.:610   3rd Qu.:-306   3rd Qu.:  77.3  
##  Max.   :485.0   Max.   :673   Max.   : 293   Max.   : 180.0  
##                                                               
##    pitch_arm         yaw_arm        total_accel_arm var_accel_arm  
##  Min.   :-88.80   Min.   :-180.00   Min.   : 1.0    Min.   : NA    
##  1st Qu.:-25.90   1st Qu.: -43.10   1st Qu.:17.0    1st Qu.: NA    
##  Median :  0.00   Median :   0.00   Median :27.0    Median : NA    
##  Mean   : -4.64   Mean   :  -0.63   Mean   :25.5    Mean   :NaN    
##  3rd Qu.: 11.20   3rd Qu.:  45.80   3rd Qu.:33.0    3rd Qu.: NA    
##  Max.   : 88.50   Max.   : 180.00   Max.   :66.0    Max.   : NA    
##                                                     NA's   :19216  
##   avg_roll_arm   stddev_roll_arm  var_roll_arm   avg_pitch_arm  
##  Min.   : NA     Min.   : NA     Min.   : NA     Min.   : NA    
##  1st Qu.: NA     1st Qu.: NA     1st Qu.: NA     1st Qu.: NA    
##  Median : NA     Median : NA     Median : NA     Median : NA    
##  Mean   :NaN     Mean   :NaN     Mean   :NaN     Mean   :NaN    
##  3rd Qu.: NA     3rd Qu.: NA     3rd Qu.: NA     3rd Qu.: NA    
##  Max.   : NA     Max.   : NA     Max.   : NA     Max.   : NA    
##  NA's   :19216   NA's   :19216   NA's   :19216   NA's   :19216  
##  stddev_pitch_arm var_pitch_arm    avg_yaw_arm    stddev_yaw_arm 
##  Min.   : NA      Min.   : NA     Min.   : NA     Min.   : NA    
##  1st Qu.: NA      1st Qu.: NA     1st Qu.: NA     1st Qu.: NA    
##  Median : NA      Median : NA     Median : NA     Median : NA    
##  Mean   :NaN      Mean   :NaN     Mean   :NaN     Mean   :NaN    
##  3rd Qu.: NA      3rd Qu.: NA     3rd Qu.: NA     3rd Qu.: NA    
##  Max.   : NA      Max.   : NA     Max.   : NA     Max.   : NA    
##  NA's   :19216    NA's   :19216   NA's   :19216   NA's   :19216  
##   var_yaw_arm     gyros_arm_x      gyros_arm_y      gyros_arm_z    
##  Min.   : NA     Min.   :-6.370   Min.   :-3.400   Min.   :-2.330  
##  1st Qu.: NA     1st Qu.:-1.330   1st Qu.:-0.800   1st Qu.:-0.070  
##  Median : NA     Median : 0.080   Median :-0.240   Median : 0.230  
##  Mean   :NaN     Mean   : 0.043   Mean   :-0.257   Mean   : 0.269  
##  3rd Qu.: NA     3rd Qu.: 1.560   3rd Qu.: 0.140   3rd Qu.: 0.720  
##  Max.   : NA     Max.   : 4.870   Max.   : 2.840   Max.   : 3.020  
##  NA's   :19216                                                     
##   accel_arm_x      accel_arm_y      accel_arm_z      magnet_arm_x 
##  Min.   :-404.0   Min.   :-318.0   Min.   :-636.0   Min.   :-584  
##  1st Qu.:-242.0   1st Qu.: -54.0   1st Qu.:-144.0   1st Qu.:-301  
##  Median : -44.0   Median :  14.0   Median : -47.0   Median : 289  
##  Mean   : -60.3   Mean   :  32.6   Mean   : -71.4   Mean   : 192  
##  3rd Qu.:  84.0   3rd Qu.: 139.0   3rd Qu.:  23.0   3rd Qu.: 638  
##  Max.   : 437.0   Max.   : 308.0   Max.   : 292.0   Max.   : 782  
##                                                                   
##   magnet_arm_y   magnet_arm_z  kurtosis_roll_arm kurtosis_picth_arm
##  Min.   :-386   Min.   :-597           :19216            :19216    
##  1st Qu.: -10   1st Qu.: 130   -0.02438:    0    -0.00484:    0    
##  Median : 201   Median : 444   -0.04190:    0    -0.01311:    0    
##  Mean   : 156   Mean   : 306   -0.05051:    0    -0.02967:    0    
##  3rd Qu.: 323   3rd Qu.: 545   -0.05695:    0    -0.07394:    0    
##  Max.   : 583   Max.   : 694   -0.08050:    0    -0.10385:    0    
##                                (Other) :    0    (Other) :    0    
##  kurtosis_yaw_arm skewness_roll_arm skewness_pitch_arm skewness_yaw_arm
##          :19216           :19216            :19216             :19216  
##  -0.01548:    0   -0.00051:    0    -0.00184:    0     -0.00311:    0  
##  -0.01749:    0   -0.00696:    0    -0.01185:    0     -0.00562:    0  
##  -0.02101:    0   -0.01884:    0    -0.01247:    0     -0.00800:    0  
##  -0.04059:    0   -0.03359:    0    -0.02063:    0     -0.01697:    0  
##  -0.04626:    0   -0.03484:    0    -0.02652:    0     -0.03455:    0  
##  (Other) :    0   (Other) :    0    (Other) :    0     (Other) :    0  
##   max_roll_arm   max_picth_arm    max_yaw_arm     min_roll_arm  
##  Min.   : NA     Min.   : NA     Min.   : NA     Min.   : NA    
##  1st Qu.: NA     1st Qu.: NA     1st Qu.: NA     1st Qu.: NA    
##  Median : NA     Median : NA     Median : NA     Median : NA    
##  Mean   :NaN     Mean   :NaN     Mean   :NaN     Mean   :NaN    
##  3rd Qu.: NA     3rd Qu.: NA     3rd Qu.: NA     3rd Qu.: NA    
##  Max.   : NA     Max.   : NA     Max.   : NA     Max.   : NA    
##  NA's   :19216   NA's   :19216   NA's   :19216   NA's   :19216  
##  min_pitch_arm    min_yaw_arm    amplitude_roll_arm amplitude_pitch_arm
##  Min.   : NA     Min.   : NA     Min.   : NA        Min.   : NA        
##  1st Qu.: NA     1st Qu.: NA     1st Qu.: NA        1st Qu.: NA        
##  Median : NA     Median : NA     Median : NA        Median : NA        
##  Mean   :NaN     Mean   :NaN     Mean   :NaN        Mean   :NaN        
##  3rd Qu.: NA     3rd Qu.: NA     3rd Qu.: NA        3rd Qu.: NA        
##  Max.   : NA     Max.   : NA     Max.   : NA        Max.   : NA        
##  NA's   :19216   NA's   :19216   NA's   :19216      NA's   :19216      
##  amplitude_yaw_arm roll_dumbbell    pitch_dumbbell    yaw_dumbbell    
##  Min.   : NA       Min.   :-153.7   Min.   :-149.6   Min.   :-150.87  
##  1st Qu.: NA       1st Qu.: -18.5   1st Qu.: -40.8   1st Qu.: -77.66  
##  Median : NA       Median :  48.2   Median : -20.9   Median :  -3.15  
##  Mean   :NaN       Mean   :  23.9   Mean   : -10.8   Mean   :   1.73  
##  3rd Qu.: NA       3rd Qu.:  67.6   3rd Qu.:  17.6   3rd Qu.:  79.79  
##  Max.   : NA       Max.   : 153.6   Max.   : 149.4   Max.   : 154.95  
##  NA's   :19216                                                        
##  kurtosis_roll_dumbbell kurtosis_picth_dumbbell kurtosis_yaw_dumbbell
##         :19216                 :19216                  :19216        
##  -0.0035:    0          -0.0163:    0           #DIV/0!:    0        
##  -0.0073:    0          -0.0233:    0                                
##  -0.0115:    0          -0.0280:    0                                
##  -0.0262:    0          -0.0308:    0                                
##  -0.0292:    0          -0.0322:    0                                
##  (Other):    0          (Other):    0                                
##  skewness_roll_dumbbell skewness_pitch_dumbbell skewness_yaw_dumbbell
##         :19216                 :19216                  :19216        
##  -0.0082:    0          -0.0053:    0           #DIV/0!:    0        
##  -0.0096:    0          -0.0084:    0                                
##  -0.0172:    0          -0.0166:    0                                
##  -0.0224:    0          -0.0452:    0                                
##  -0.0234:    0          -0.0458:    0                                
##  (Other):    0          (Other):    0                                
##  max_roll_dumbbell max_picth_dumbbell max_yaw_dumbbell min_roll_dumbbell
##  Min.   : NA       Min.   : NA               :19216    Min.   : NA      
##  1st Qu.: NA       1st Qu.: NA        -0.1   :    0    1st Qu.: NA      
##  Median : NA       Median : NA        -0.2   :    0    Median : NA      
##  Mean   :NaN       Mean   :NaN        -0.3   :    0    Mean   :NaN      
##  3rd Qu.: NA       3rd Qu.: NA        -0.4   :    0    3rd Qu.: NA      
##  Max.   : NA       Max.   : NA        -0.5   :    0    Max.   : NA      
##  NA's   :19216     NA's   :19216      (Other):    0    NA's   :19216    
##  min_pitch_dumbbell min_yaw_dumbbell amplitude_roll_dumbbell
##  Min.   : NA               :19216    Min.   : NA            
##  1st Qu.: NA        -0.1   :    0    1st Qu.: NA            
##  Median : NA        -0.2   :    0    Median : NA            
##  Mean   :NaN        -0.3   :    0    Mean   :NaN            
##  3rd Qu.: NA        -0.4   :    0    3rd Qu.: NA            
##  Max.   : NA        -0.5   :    0    Max.   : NA            
##  NA's   :19216      (Other):    0    NA's   :19216          
##  amplitude_pitch_dumbbell amplitude_yaw_dumbbell total_accel_dumbbell
##  Min.   : NA                     :19216          Min.   : 0.0        
##  1st Qu.: NA              #DIV/0!:    0          1st Qu.: 4.0        
##  Median : NA              0.00   :    0          Median :10.0        
##  Mean   :NaN                                     Mean   :13.7        
##  3rd Qu.: NA                                     3rd Qu.:19.0        
##  Max.   : NA                                     Max.   :58.0        
##  NA's   :19216                                                       
##  var_accel_dumbbell avg_roll_dumbbell stddev_roll_dumbbell
##  Min.   : NA        Min.   : NA       Min.   : NA         
##  1st Qu.: NA        1st Qu.: NA       1st Qu.: NA         
##  Median : NA        Median : NA       Median : NA         
##  Mean   :NaN        Mean   :NaN       Mean   :NaN         
##  3rd Qu.: NA        3rd Qu.: NA       3rd Qu.: NA         
##  Max.   : NA        Max.   : NA       Max.   : NA         
##  NA's   :19216      NA's   :19216     NA's   :19216       
##  var_roll_dumbbell avg_pitch_dumbbell stddev_pitch_dumbbell
##  Min.   : NA       Min.   : NA        Min.   : NA          
##  1st Qu.: NA       1st Qu.: NA        1st Qu.: NA          
##  Median : NA       Median : NA        Median : NA          
##  Mean   :NaN       Mean   :NaN        Mean   :NaN          
##  3rd Qu.: NA       3rd Qu.: NA        3rd Qu.: NA          
##  Max.   : NA       Max.   : NA        Max.   : NA          
##  NA's   :19216     NA's   :19216      NA's   :19216        
##  var_pitch_dumbbell avg_yaw_dumbbell stddev_yaw_dumbbell var_yaw_dumbbell
##  Min.   : NA        Min.   : NA      Min.   : NA         Min.   : NA     
##  1st Qu.: NA        1st Qu.: NA      1st Qu.: NA         1st Qu.: NA     
##  Median : NA        Median : NA      Median : NA         Median : NA     
##  Mean   :NaN        Mean   :NaN      Mean   :NaN         Mean   :NaN     
##  3rd Qu.: NA        3rd Qu.: NA      3rd Qu.: NA         3rd Qu.: NA     
##  Max.   : NA        Max.   : NA      Max.   : NA         Max.   : NA     
##  NA's   :19216      NA's   :19216    NA's   :19216       NA's   :19216   
##  gyros_dumbbell_x  gyros_dumbbell_y gyros_dumbbell_z accel_dumbbell_x
##  Min.   :-204.00   Min.   :-2.10    Min.   : -2.4    Min.   :-419.0  
##  1st Qu.:  -0.03   1st Qu.:-0.14    1st Qu.: -0.3    1st Qu.: -50.0  
##  Median :   0.13   Median : 0.03    Median : -0.1    Median :  -8.0  
##  Mean   :   0.16   Mean   : 0.05    Mean   : -0.1    Mean   : -28.5  
##  3rd Qu.:   0.35   3rd Qu.: 0.21    3rd Qu.:  0.0    3rd Qu.:  11.0  
##  Max.   :   2.22   Max.   :52.00    Max.   :317.0    Max.   : 235.0  
##                                                                      
##  accel_dumbbell_y accel_dumbbell_z magnet_dumbbell_x magnet_dumbbell_y
##  Min.   :-189.0   Min.   :-334.0   Min.   :-643      Min.   :-3600    
##  1st Qu.:  -8.0   1st Qu.:-141.0   1st Qu.:-535      1st Qu.:  231    
##  Median :  41.0   Median :  -1.0   Median :-479      Median :  311    
##  Mean   :  52.6   Mean   : -38.2   Mean   :-328      Mean   :  221    
##  3rd Qu.: 111.0   3rd Qu.:  38.0   3rd Qu.:-304      3rd Qu.:  390    
##  Max.   : 315.0   Max.   : 318.0   Max.   : 584      Max.   :  633    
##                                                                       
##  magnet_dumbbell_z  roll_forearm     pitch_forearm     yaw_forearm    
##  Min.   :-262.0    Min.   :-180.00   Min.   :-72.50   Min.   :-180.0  
##  1st Qu.: -45.0    1st Qu.:  -0.67   1st Qu.:  0.00   1st Qu.: -68.5  
##  Median :  13.0    Median :  21.90   Median :  9.21   Median :   0.0  
##  Mean   :  46.2    Mean   :  33.88   Mean   : 10.67   Mean   :  19.3  
##  3rd Qu.:  95.0    3rd Qu.: 140.00   3rd Qu.: 28.30   3rd Qu.: 110.0  
##  Max.   : 452.0    Max.   : 180.00   Max.   : 89.80   Max.   : 180.0  
##                                                                       
##  kurtosis_roll_forearm kurtosis_picth_forearm kurtosis_yaw_forearm
##         :19216                :19216                 :19216       
##  -0.0227:    0         -0.0073:    0          #DIV/0!:    0       
##  -0.0359:    0         -0.0442:    0                              
##  -0.0567:    0         -0.0489:    0                              
##  -0.0781:    0         -0.0523:    0                              
##  -0.1363:    0         -0.0891:    0                              
##  (Other):    0         (Other):    0                              
##  skewness_roll_forearm skewness_pitch_forearm skewness_yaw_forearm
##         :19216                :19216                 :19216       
##  -0.0004:    0         -0.0113:    0          #DIV/0!:    0       
##  -0.0013:    0         -0.0131:    0                              
##  -0.0063:    0         -0.0405:    0                              
##  -0.0088:    0         -0.0478:    0                              
##  -0.0090:    0         -0.0482:    0                              
##  (Other):    0         (Other):    0                              
##  max_roll_forearm max_picth_forearm max_yaw_forearm min_roll_forearm
##  Min.   : NA      Min.   : NA              :19216   Min.   : NA     
##  1st Qu.: NA      1st Qu.: NA       -0.1   :    0   1st Qu.: NA     
##  Median : NA      Median : NA       -0.2   :    0   Median : NA     
##  Mean   :NaN      Mean   :NaN       -0.3   :    0   Mean   :NaN     
##  3rd Qu.: NA      3rd Qu.: NA       -0.4   :    0   3rd Qu.: NA     
##  Max.   : NA      Max.   : NA       -0.5   :    0   Max.   : NA     
##  NA's   :19216    NA's   :19216     (Other):    0   NA's   :19216   
##  min_pitch_forearm min_yaw_forearm amplitude_roll_forearm
##  Min.   : NA              :19216   Min.   : NA           
##  1st Qu.: NA       -0.1   :    0   1st Qu.: NA           
##  Median : NA       -0.2   :    0   Median : NA           
##  Mean   :NaN       -0.3   :    0   Mean   :NaN           
##  3rd Qu.: NA       -0.4   :    0   3rd Qu.: NA           
##  Max.   : NA       -0.5   :    0   Max.   : NA           
##  NA's   :19216     (Other):    0   NA's   :19216         
##  amplitude_pitch_forearm amplitude_yaw_forearm total_accel_forearm
##  Min.   : NA                    :19216         Min.   :  0.0      
##  1st Qu.: NA             #DIV/0!:    0         1st Qu.: 29.0      
##  Median : NA             0.00   :    0         Median : 36.0      
##  Mean   :NaN                                   Mean   : 34.7      
##  3rd Qu.: NA                                   3rd Qu.: 41.0      
##  Max.   : NA                                   Max.   :108.0      
##  NA's   :19216                                                    
##  var_accel_forearm avg_roll_forearm stddev_roll_forearm var_roll_forearm
##  Min.   : NA       Min.   : NA      Min.   : NA         Min.   : NA     
##  1st Qu.: NA       1st Qu.: NA      1st Qu.: NA         1st Qu.: NA     
##  Median : NA       Median : NA      Median : NA         Median : NA     
##  Mean   :NaN       Mean   :NaN      Mean   :NaN         Mean   :NaN     
##  3rd Qu.: NA       3rd Qu.: NA      3rd Qu.: NA         3rd Qu.: NA     
##  Max.   : NA       Max.   : NA      Max.   : NA         Max.   : NA     
##  NA's   :19216     NA's   :19216    NA's   :19216       NA's   :19216   
##  avg_pitch_forearm stddev_pitch_forearm var_pitch_forearm avg_yaw_forearm
##  Min.   : NA       Min.   : NA          Min.   : NA       Min.   : NA    
##  1st Qu.: NA       1st Qu.: NA          1st Qu.: NA       1st Qu.: NA    
##  Median : NA       Median : NA          Median : NA       Median : NA    
##  Mean   :NaN       Mean   :NaN          Mean   :NaN       Mean   :NaN    
##  3rd Qu.: NA       3rd Qu.: NA          3rd Qu.: NA       3rd Qu.: NA    
##  Max.   : NA       Max.   : NA          Max.   : NA       Max.   : NA    
##  NA's   :19216     NA's   :19216        NA's   :19216     NA's   :19216  
##  stddev_yaw_forearm var_yaw_forearm gyros_forearm_x   gyros_forearm_y 
##  Min.   : NA        Min.   : NA     Min.   :-22.000   Min.   : -7.02  
##  1st Qu.: NA        1st Qu.: NA     1st Qu.: -0.220   1st Qu.: -1.48  
##  Median : NA        Median : NA     Median :  0.050   Median :  0.03  
##  Mean   :NaN        Mean   :NaN     Mean   :  0.158   Mean   :  0.08  
##  3rd Qu.: NA        3rd Qu.: NA     3rd Qu.:  0.560   3rd Qu.:  1.62  
##  Max.   : NA        Max.   : NA     Max.   :  3.970   Max.   :311.00  
##  NA's   :19216      NA's   :19216                                     
##  gyros_forearm_z  accel_forearm_x  accel_forearm_y accel_forearm_z 
##  Min.   : -8.09   Min.   :-498.0   Min.   :-632    Min.   :-446.0  
##  1st Qu.: -0.18   1st Qu.:-178.0   1st Qu.:  57    1st Qu.:-182.0  
##  Median :  0.08   Median : -57.0   Median : 201    Median : -39.0  
##  Mean   :  0.15   Mean   : -61.3   Mean   : 164    Mean   : -55.2  
##  3rd Qu.:  0.49   3rd Qu.:  77.0   3rd Qu.: 312    3rd Qu.:  26.0  
##  Max.   :231.00   Max.   : 477.0   Max.   : 923    Max.   : 291.0  
##                                                                    
##  magnet_forearm_x magnet_forearm_y magnet_forearm_z classe  
##  Min.   :-1280    Min.   :-896     Min.   :-973     A:5471  
##  1st Qu.: -616    1st Qu.:   4     1st Qu.: 189     B:3718  
##  Median : -377    Median : 592     Median : 511     C:3352  
##  Mean   : -312    Mean   : 381     Mean   : 393     D:3147  
##  3rd Qu.:  -73    3rd Qu.: 737     3rd Qu.: 653     E:3528  
##  Max.   :  672    Max.   :1480     Max.   :1090             
## 
```

We see several columns that are fully NA. These appear to be the summary statistics variables. We can remove all these by finding the near zero variance predictor columns. This will also remove any other columns with few unique values that would also be poor predictors.


```r
nzv <- nearZeroVar(train, saveMetrics = TRUE)

keep <- row.names(nzv[nzv[,"zeroVar"] == FALSE,])

train <- train[,keep]

names(train)
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "num_window"          
##  [7] "roll_belt"            "pitch_belt"           "yaw_belt"            
## [10] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
## [13] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [16] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [19] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [22] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [25] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [28] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [31] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [34] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [37] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [40] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [43] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [46] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [49] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [52] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [55] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [58] "magnet_forearm_z"     "classe"
```

We are now down to 59 variables from the original 160. The first 6 columns only contain subject id and timestamps. Since these should be tied directly to each subject, we can remove these, as they would no make for good predictors. This will put us down to 53 variables.


```r
train <- train[,-c(1:6)]
```

###Fitting the Model

Now that the data has been processed we can proceed to splitting the data into two sets.


```r
set.seed(1985)

inTrain <- createDataPartition(train$classe, p=0.7, list=FALSE)

training <- train[inTrain,]

testing <- train[-inTrain,]
```

We can now fit our Random Forest model to the training set using cross-validation with 10 folds.


```r
ctrl <- trainControl(method="cv", number=10, allowParallel = TRUE)

set.seed(1234)
modelFit <- train(classe ~ ., data = training, method = "rf", trControl = ctrl)

modelFit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 2
## 
##         OOB estimate of  error rate: 0.7%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3825    4    0    0    1    0.001305
## B   15 2584    4    0    0    0.007299
## C    0   16 2329    2    0    0.007669
## D    0    0   43 2158    2    0.020427
## E    0    0    2    5 2463    0.002834
```

###Results

The confusion matrix shows very low classification error, only a 0.7% out of bag error rate. We can now apply our model to the test set and get our out of sample error.


```r
prediction <- predict(modelFit, newdata = testing)

(sum(prediction != testing$classe)/nrow(testing))*100 ## out of sample error calculation
```

```
## [1] 0.6767
```

```r
confusionMatrix(prediction, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1641    7    0    0    0
##          B    0 1103    2    0    0
##          C    0    5 1001   20    0
##          D    0    0    2  924    3
##          E    0    0    0    0 1055
## 
## Overall Statistics
##                                         
##                Accuracy : 0.993         
##                  95% CI : (0.991, 0.995)
##     No Information Rate : 0.285         
##     P-Value [Acc > NIR] : <2e-16        
##                                         
##                   Kappa : 0.991         
##  Mcnemar's Test P-Value : NA            
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             1.000    0.989    0.996    0.979    0.997
## Specificity             0.998    1.000    0.995    0.999    1.000
## Pos Pred Value          0.996    0.998    0.976    0.995    1.000
## Neg Pred Value          1.000    0.997    0.999    0.996    0.999
## Prevalence              0.285    0.193    0.174    0.164    0.184
## Detection Rate          0.285    0.191    0.174    0.160    0.183
## Detection Prevalence    0.286    0.192    0.178    0.161    0.183
## Balanced Accuracy       0.999    0.994    0.995    0.989    0.999
```

We can see that the out of sample error rate is only 0.68%, which is good enough for our purposes. Looking at the confusion matrix we see that the model had the most problems labeling class D (lowering the dumbbell halfway). It put a few instances as class C (raising the dumbbell halfway). This makes sense as the motions are similar, with the biggest differences being the direction (up or down) of the motion.

In addition to our test dataset, a 20 row dataset was also provided for this project. When the model was applied to this small dataset and submitted for grading, it predicted the classe with 100% accuracy. The code is not provided, but the dataset can be found in the GitHub repo.

###Conclusion

For the purposes of this report we used a very simplistic approach with the Random Forest learning method, using only a training and test set with 10 fold cross-validation. To improve our accuracy we may attempt to use a more robust model validation technique. Combining predictors may also be something to explore in future studies.
