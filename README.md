# AI-X_DeepLearning_2022Fall_FinalProject
## Title
**Exploring Survival in the Hospital**(입원 중인 환자의 생존률)
<br>블로그 링크: https://github.com/cdh1110/AI-X_DeepLearning_2022
## Members
<br> 차도현, 물리학과, chamena1110@gmail.com (데이터셋 탐색, feature 분석, 데이터셋 가공, 코드 작성 및 실행, 블로그 작성, 유튜브 촬영) <br> Khaliun, 경제금융학부, haliunurgamaltuya@gmail.com (feature 분석 도움)
## YouTube Link

[![YouTube](https://img.youtube.com/vi/s9ta5gagG-I/0.jpg)](https://youtu.be/s9ta5gagG-I)
<br> https://youtu.be/s9ta5gagG-I

## Index
####    I. Proposal
####    II. Datasets
####    III. Methodology
####    IV. Evaluation & Analysis
####    V. Related Works
####    VI. Conclusion: Discussion

## I. Proposal (Option A)

- Motivation: <br> 본 수업에서 제공된 타이타닉 예제, Exploring Survivals on Titanic 글은 ML의 기초를 다질 입문 과정으로 적합해 보입니다. 따라서 공부와 연습을 위해 그와 유사한 타입의 주제를 선정하고 싶었습니다. Kaggle과 각종 공공데이터 웹사이트를 이용해 Binary Classification이면서, Feature Engineering, EDA를 연습하기 좋은 데이터를 검색하여 여러 후보들을 추리고 조원과 상의한 결과, 현 주제인 Patient Survival Prediction을 하기로 최종 결정했습니다. 제공된 feature의 종류가 85개에다, 결측치도 간간히 포함되어 있어 보다 팀 프로젝트에 최적인 양으로 보이고, 실재 기반 데이터이며 예측 모델을 성공적으로 구축 시, '사망'을 예방할 수 있다는 실용적인 의미도 있는 것이 그 이유입니다.   

- What do you want to see at the end?: <br> 내적으로는, 입원 중인 환자의 병원 내 사망률에 대한 주 요인을 파악하고, 이에 머신러닝 기반의 예측 모델을 구축하는 것이 목표입니다. <br> 외적으로는 다음의 학습 목표를 통해 EDA와 ML의 기초를 다지는 것입니다. <ol> -약 80여개의 다양한 feature중 중요 feature을 선별 <br> -일부 존재하는 결측치를 합리적으로 처리 <br> -머신러닝에 사용되는 알고리즘을 이해하고 실전 데이터에 적용 


## II. Datasets
 
 
 
 
- 데이터는 다음의 kaggle 웹사이트로부터 얻었습니다.
https://www.kaggle.com/datasets/mitishaagarwal/patient 

ICU(중환자실) 입원 환자의 여러 특징과 해당 환자 사망 여부를 나타낸 데이터입니다. 환자가 생존 했을 경우 0, 사망 했을 경우 1입니다. 
*타이타닉 예제와 같은 binary classification입니다. 약 80개나 되는 다양한 feature가 존재하고 중간중간 결측치(Null, NaN) 또한 가지고 있어 EDA에 대부분의 시간이 할애될 것입니다. 따라서 이 두 가지 특징 때문에, 본 프로젝트는 타이타닉 예제와 매우 유사한 방식으로 진행될 것이 예상됩니다.*
<br>
<br>사용할 언어는 R과 python입니다. II장 데이터 분석에 R이, 그 외에는 대부분 python이 사용되었습니다. 
<br>
### II-1. Overview

먼저 데이터 분석에 필요한 패키지들을 불러옵니다.
```R
library('dplyr')    #data manipulation
library('tidyr')    #data manipulation
library('naniar')    #NA manipulation
```
```Python
import numpy as np   #data manipulation
import pandas as pd   #data manipulation
import matplotlib.pyplot as plt    #plot manipulation
import seaborn as sns    #visualiztion
import os
```

이제 데이터를 로드합니다.
```R
data <- read.csv('./project_data.csv', stringsAsFactors = F, na.strings = c("", " ","  ", NA))
str(data)
```
```
'data.frame':   91713 obs. of  85 variables:
 $ encounter_id                 : int  66154 114252 119783 79267 92056 33181 82208 120995 80471 42871 ...
 $ patient_id                   : int  25312 59342 50777 46918 34377 74489 49526 50129 10577 90749 ...
 $ hospital_id                  : int  118 81 118 118 33 83 83 33 118 118 ...
 $ age                          : int  68 77 25 81 19 67 59 70 45 50 ...
 $ bmi                          : num  22.7 27.4 31.9 22.6 NA ...
 $ elective_surgery             : int  0 0 0 1 0 0 0 0 0 0 ...
 $ ethnicity                    : chr  "Caucasian" "Caucasian" "Caucasian" "Caucasian" ...
 $ gender                       : chr  "M" "F" "F" "F" ...
 $ height                       : num  180 160 173 165 188 ...
 $ icu_admit_source             : chr  "Floor" "Floor" "Accident & Emergency" "Operating Room / Recovery" ...
 $ icu_id                       : int  92 90 93 92 91 95 95 91 114 114 ...
 $ icu_stay_type                : chr  "admit" "admit" "admit" "admit" ...
 $ icu_type                     : chr  "CTICU" "Med-Surg ICU" "Med-Surg ICU" "CTICU" ...
 $ pre_icu_los_days             : num  0.541667 0.927778 0.000694 0.000694 0.073611 ...
 $ weight                       : num  73.9 70.2 95.3 61.7 NA ...
 $ apache_2_diagnosis           : int  113 108 122 203 119 301 108 113 116 112 ...
 $ apache_3j_diagnosis          : num  502 203 703 1206 601 ...
 $ apache_post_operative        : int  0 0 0 1 0 0 0 0 0 0 ...
 $ arf_apache                   : int  0 0 0 0 0 0 0 0 0 0 ...
 $ gcs_eyes_apache              : int  3 1 3 4 NA 4 4 4 4 4 ...
 $ gcs_motor_apache             : int  6 3 6 6 NA 6 6 6 6 6 ...
 $ gcs_unable_apache            : int  0 0 0 0 NA 0 0 0 0 0 ...
 $ gcs_verbal_apache            : int  4 1 5 5 NA 5 5 5 5 5 ...
 $ heart_rate_apache            : int  118 120 102 114 60 113 133 120 82 94 ...
 $ intubated_apache             : int  0 0 0 1 0 0 1 0 0 0 ...
 $ map_apache                   : int  40 46 68 60 103 130 138 60 66 58 ...
 $ resprate_apache              : num  36 33 37 4 16 35 53 28 14 46 ...
 $ temp_apache                  : num  39.3 35.1 36.7 34.8 36.7 36.6 35 36.6 36.9 36.3 ...
 $ ventilated_apache            : int  0 1 0 1 0 0 1 1 1 0 ...
 $ d1_diasbp_max                : int  68 95 88 48 99 100 76 84 65 83 ...
 $ d1_diasbp_min                : int  37 31 48 42 57 61 68 46 59 48 ...
 $ d1_diasbp_noninvasive_max    : int  68 95 88 48 99 100 76 84 65 83 ...
 $ d1_diasbp_noninvasive_min    : int  37 31 48 42 57 61 68 46 59 48 ...
 $ d1_heartrate_max             : int  119 118 96 116 89 113 112 118 82 96 ...
 $ d1_heartrate_min             : int  72 72 68 92 60 83 70 86 82 57 ...
 $ d1_mbp_max                   : int  89 120 102 84 104 127 117 114 93 101 ...
 $ d1_mbp_min                   : int  46 38 68 84 90 80 97 60 71 59 ...
 $ d1_mbp_noninvasive_max       : int  89 120 102 84 104 127 117 114 93 101 ...
 $ d1_mbp_noninvasive_min       : int  46 38 68 84 90 80 97 60 71 59 ...
 $ d1_resprate_max              : int  34 32 21 23 18 32 38 28 24 44 ...
 $ d1_resprate_min              : int  10 12 8 7 16 10 16 12 19 14 ...
 $ d1_spo2_max                  : int  100 100 98 100 100 97 100 100 97 100 ...
 $ d1_spo2_min                  : int  74 70 91 95 96 91 87 92 97 96 ...
 $ d1_sysbp_max                 : int  131 159 148 158 147 173 151 147 104 135 ...
 $ d1_sysbp_min                 : int  73 67 105 84 120 107 133 71 98 78 ...
 $ d1_sysbp_noninvasive_max     : int  131 159 148 158 147 173 151 147 104 135 ...
 $ d1_sysbp_noninvasive_min     : num  73 67 105 84 120 107 133 71 98 78 ...
 $ d1_temp_max                  : num  39.9 36.3 37 38 37.2 36.8 37.2 38.5 36.9 37.1 ...
 $ d1_temp_min                  : num  37.2 35.1 36.7 34.8 36.7 36.6 35 36.6 36.9 36.4 ...
 $ h1_diasbp_max                : int  68 61 88 62 99 89 107 74 65 83 ...
 $ h1_diasbp_min                : int  63 48 58 44 68 89 79 55 59 61 ...
 $ h1_diasbp_noninvasive_max    : int  68 61 88 NA 99 89 NA 74 65 83 ...
 $ h1_diasbp_noninvasive_min    : int  63 48 58 NA 68 89 NA 55 59 61 ...
 $ h1_heartrate_max             : int  119 114 96 100 89 83 79 118 82 96 ...
 $ h1_heartrate_min             : int  108 100 78 96 76 83 72 114 82 60 ...
 $ h1_mbp_max                   : int  86 85 91 92 104 111 117 88 93 101 ...
 $ h1_mbp_min                   : int  85 57 83 71 92 111 117 60 71 77 ...
 $ h1_mbp_noninvasive_max       : int  86 85 91 NA 104 111 117 88 93 101 ...
 $ h1_mbp_noninvasive_min       : int  85 57 83 NA 92 111 117 60 71 77 ...
 $ h1_resprate_max              : int  26 31 20 12 NA 12 18 28 24 29 ...
 $ h1_resprate_min              : int  18 28 16 11 NA 12 18 26 19 17 ...
 $ h1_spo2_max                  : int  100 95 98 100 100 97 100 96 97 100 ...
 $ h1_spo2_min                  : int  74 70 91 99 100 97 100 92 97 96 ...
 $ h1_sysbp_max                 : int  131 95 148 136 130 143 191 119 104 135 ...
 $ h1_sysbp_min                 : int  115 71 124 106 120 143 163 106 98 103 ...
 $ h1_sysbp_noninvasive_max     : int  131 95 148 NA 130 143 NA 119 104 135 ...
 $ h1_sysbp_noninvasive_min     : int  115 71 124 NA 120 143 NA 106 98 103 ...
 $ d1_glucose_max               : int  168 145 NA 185 NA 156 197 129 365 134 ...
 $ d1_glucose_min               : int  109 128 NA 88 NA 125 129 129 288 134 ...
 $ d1_potassium_max             : num  4 4.2 NA 5 NA 3.9 5 5.8 5.2 4.1 ...
 $ d1_potassium_min             : num  3.4 3.8 NA 3.5 NA 3.7 4.2 2.4 5.2 3.3 ...
 $ apache_4a_hospital_death_prob: num  0.1 0.47 0 0.04 NA 0.05 0.1 0.11 NA 0.02 ...
 $ apache_4a_icu_death_prob     : num  0.05 0.29 0 0.03 NA 0.02 0.05 0.06 NA 0.01 ...
 $ aids                         : int  0 0 0 0 0 0 0 0 0 0 ...
 $ cirrhosis                    : int  0 0 0 0 0 0 0 0 0 0 ...
 $ diabetes_mellitus            : int  1 1 0 0 0 1 1 0 0 0 ...
 $ hepatic_failure              : int  0 0 0 0 0 0 0 0 0 0 ...
 $ immunosuppression            : int  0 0 0 0 0 0 0 1 0 0 ...
 $ leukemia                     : int  0 0 0 0 0 0 0 0 0 0 ...
 $ lymphoma                     : int  0 0 0 0 0 0 0 0 0 0 ...
 $ solid_tumor_with_metastasis  : int  0 0 0 0 0 0 0 0 0 0 ...
 $ apache_3j_bodysystem         : chr  "Sepsis" "Respiratory" "Metabolic" "Cardiovascular" ...
 $ apache_2_bodysystem          : chr  "Cardiovascular" "Respiratory" "Metabolic" "Cardiovascular" ...
 $ X                            : logi  NA NA NA NA NA NA ...
 $ hospital_death               : int  0 0 0 0 0 0 0 0 1 0 ...
        
```    
총 **91713명**의 환자에 대한 **85개**의 변수를 확인할 수 있습니다.<br>각 변수에 대한 설명을 간략히 아래 표로 정리했습니다. (자세한 설명은 섹션 III에 이어서 합니다.)
        
| 변수 이름 | 설명 |
| ------------- | ------------- |
| encounter_id   | 입원과 관련된 ID  |
| patient_id    | 환자 ID  |                    
|hospital_id| 병원 ID  |
|age| 나이  |                        
|bmi| 체질량지수  |                           
|elective_surgery| 선택적 수술 동의(1), 거부(0) |
|ethnicity| 인종  |
|gender| 성별  |
|height| 키  |
|weight| 체중 | 
|icu_~| 집중치료실(ICU) 관련 데이터  |
|apache_2_...| 의학적 점수인 APACHE II와 관련된 데이터 |  
|apache_3_...| 의학적 점수인 APACHE III와 관련된 데이터 | 
|apache_4_...| 의학적 점수인 APACHE IV와 관련된 데이터 |  
|apache_post_operative| 수술 받음(1), 받지 않음(0) |
|gcs_~| 글래스고 혼수척도 관련 데이터 |          
|d1_heartrate_max/min| 최고/최저 심박수(24h) |
|...| ...  |
|h1_heartrate_max/min| 최고/최저 심박수(1h) |
|...| ...  |       
|aids/cirrhosis/...| 에이즈/경화증/... 관련 병력(1), 이상없음(0) |        
| soliol_tumor_with_metastasis | 전이 종양 진단(1) 해당없음 (0) |
|...| ... |
|**hospital_death**| **사망(1), 생존(0)**  |   

### II-2. Check Missing Values        
데이터의 결측치를 확인합니다.
   <code>is.na()</code>를 이용해서 결측치를 간단히 확인하는 것도 좋지만,
```R
df_null <- data.frame(colSums(is.na(data)))        
head(df_null)
```
```
                 colSums.is.na.data..
encounter_id                        0
patient_id                          0
hospital_id                         0
age                              4228
bmi                              3429
elective_surgery                    0
```        
<code>naniar</code> 패키지를 사용해 결측치를 좀 더 체계적으로 확인합니다.
```R
print(naniar::miss_var_summary(data),n=85)
```        
```        
# A tibble: 85 × 3
   variable                      n_miss pct_miss
   <chr>                          <int>    <dbl>
 1 X                              91713 100     
 2 d1_potassium_max                9585  10.5   
 3 d1_potassium_min                9585  10.5   
 4 h1_mbp_noninvasive_max          9084   9.90  
 5 h1_mbp_noninvasive_min          9084   9.90  
 6 apache_4a_hospital_death_prob   7947   8.67  
 7 apache_4a_icu_death_prob        7947   8.67  
 8 h1_diasbp_noninvasive_max       7350   8.01  
 9 h1_diasbp_noninvasive_min       7350   8.01  
10 h1_sysbp_noninvasive_max        7341   8.00  
11 h1_sysbp_noninvasive_min        7341   8.00  
12 d1_glucose_max                  5807   6.33  
13 d1_glucose_min                  5807   6.33  
14 h1_mbp_max                      4639   5.06  
15 h1_mbp_min                      4639   5.06  
16 h1_resprate_max                 4357   4.75  
17 h1_resprate_min                 4357   4.75  
18 age                             4228   4.61  
19 h1_spo2_max                     4185   4.56  
20 h1_spo2_min                     4185   4.56  
21 temp_apache                     4108   4.48  
22 h1_diasbp_max                   3619   3.95  
23 h1_diasbp_min                   3619   3.95  
24 h1_sysbp_max                    3611   3.94  
25 h1_sysbp_min                    3611   3.94  
26 bmi                             3429   3.74  
27 h1_heartrate_max                2790   3.04  
28 h1_heartrate_min                2790   3.04  
29 weight                          2720   2.97  
30 d1_temp_max                     2324   2.53  
31 d1_temp_min                     2324   2.53  
32 gcs_eyes_apache                 1901   2.07  
33 gcs_motor_apache                1901   2.07  
34 gcs_verbal_apache               1901   2.07  
35 apache_2_diagnosis              1662   1.81  
36 apache_3j_bodysystem            1662   1.81  
37 apache_2_bodysystem             1662   1.81  
38 d1_mbp_noninvasive_max          1479   1.61  
39 d1_mbp_noninvasive_min          1479   1.61  
40 ethnicity                       1395   1.52  
41 height                          1334   1.45  
42 resprate_apache                 1234   1.35  
43 apache_3j_diagnosis             1101   1.20  
44 d1_diasbp_noninvasive_max       1040   1.13  
45 d1_diasbp_noninvasive_min       1040   1.13  
46 gcs_unable_apache               1037   1.13  
47 d1_sysbp_noninvasive_max        1027   1.12  
48 d1_sysbp_noninvasive_min        1027   1.12  
49 map_apache                       994   1.08  
50 heart_rate_apache                878   0.957 
51 arf_apache                       715   0.780 
52 intubated_apache                 715   0.780 
53 ventilated_apache                715   0.780 
54 aids                             715   0.780 
55 cirrhosis                        715   0.780 
56 diabetes_mellitus                715   0.780 
57 hepatic_failure                  715   0.780 
58 immunosuppression                715   0.780 
59 leukemia                         715   0.780 
60 lymphoma                         715   0.780 
61 solid_tumor_with_metastasis      715   0.780 
62 d1_resprate_max                  385   0.420 
63 d1_resprate_min                  385   0.420 
64 d1_spo2_max                      333   0.363 
65 d1_spo2_min                      333   0.363 
66 d1_mbp_max                       220   0.240 
67 d1_mbp_min                       220   0.240 
68 d1_diasbp_max                    165   0.180 
69 d1_diasbp_min                    165   0.180 
70 d1_sysbp_max                     159   0.173 
71 d1_sysbp_min                     159   0.173 
72 d1_heartrate_max                 145   0.158 
73 d1_heartrate_min                 145   0.158 
74 icu_admit_source                 112   0.122 
75 gender                            25   0.0273
76 encounter_id                       0   0     
77 patient_id                         0   0     
78 hospital_id                        0   0     
79 elective_surgery                   0   0     
80 icu_id                             0   0     
81 icu_stay_type                      0   0     
82 icu_type                           0   0     
83 pre_icu_los_days                   0   0     
84 apache_post_operative              0   0     
85 hospital_death                     0   0 
``` 
결측치 개수에 따라 내림차순으로 정리한 결과를 얻었습니다. 각 3개의 열은 변수의 이름, 결측치, 결측 퍼센티지를 나타냅니다. 

### II-3. Analysis of Missing Values   

위 tibble 프레임의 첫 줄을 살펴보면, "X" 열의 결측값이 91713개로, 100% 비율의 결측을 가지고 있습니다. <br>열 이름에서 유추할 수 있듯 *아무런 의미가 없는* 열이기 때문에 "X" 열은 **삭제**합니다.
```R  
data <- subset(data, select=-c(X))
```
다시 간략히 확인해보면,
```R  
naniar::miss_var_summary(data)
```
```
# A tibble: 84 × 3
   variable                      n_miss pct_miss
   <chr>                          <int>    <dbl>
 1 d1_potassium_max                9585    10.5 
 2 d1_potassium_min                9585    10.5 
 3 h1_mbp_noninvasive_max          9084     9.90
 4 h1_mbp_noninvasive_min          9084     9.90
 5 apache_4a_hospital_death_prob   7947     8.67
 6 apache_4a_icu_death_prob        7947     8.67
 7 h1_diasbp_noninvasive_max       7350     8.01
 8 h1_diasbp_noninvasive_min       7350     8.01
 9 h1_sysbp_noninvasive_max        7341     8.00
10 h1_sysbp_noninvasive_min        7341     8.00
# … with 74 more rows  
```  
'X'행이 삭제되고 84개열이 정상적으로 남겨졌음을 확인할 수 있습니다.
<br>
<br>이번에는 결측치가 한 개도 존재하지 않는 행의 개수를 확인합니다.
```R  
nona <- sum(complete.cases(data)) #결측치가 없는 행의 개수 
yesna <- sum(!complete.cases(data)) #결측치가 있는 행의 개수
rows_with_na <- list(Non_NA = nona, With_NA = yesna, Sum = nona + yesna) #리스트에 저장
unlist(rows_with_na) #출력
```
```
 Non_NA With_NA     Sum 
  56935   34778   91713 
```
총 91713 행 중 56935 행은 결측치가 없는 완전한 데이터이며, 나머지 34778 행은 1개 이상의 결측치가 존재합니다. <br>즉, 단순히 결측치 행을 제거(Deletion)하면 전체 데이터의 약 38% 를 잃게 됩니다.<br>이 손실을 조금이라도 줄이고자, 각 변수들을 one by one 분석하면서 열의 삭제 여부를 결정하겠습니다.








  
## III. Methodology 


### III - 1. Deleting 'ID's
데이터에 포함된 feature 중 큰 의미가 없는 '고유식별자(ID)' 열들을 제거합니다.
```Python
id_list=[] 
for name in data.columns:
    if '_id' in name:               #열 이름에 '_id'가 포함되어있다면
        id_list.append(name)        #해당 열 이름 추출
print(id_list)        
```
```Python
['encounter_id', 'patient_id', 'hospital_id', 'icu_id']
```
다음으로 해당 열들의 유니크한 값의 개수를 확인합니다.
```Python
for _id in id_list:
    nun = data[_id].nunique()                    #nunique로 id 열들의 유니크 값 개수 확인
    print(_id, '의 유니크 값 개수: ',nun)          #출력
```
```
encounter_id 의 유니크 값 개수:  91713
patient_id 의 유니크 값 개수:  91713
hospital_id 의 유니크 값 개수:  147
icu_id 의 유니크 값 개수:  241
```
'encounter_id'나 'patient_id'는 고유값의 개수가 행의 개수(91713)과 동일합니다. 환자 별로 할당된 고유식별자임을 쉽게 알 수 있지만,
<br>'hospital_id', 'icu_id'는 그렇지 않습니다. 혹시 모르니 <code>value_counts</code>를 사용해 고유값 별로 몇개의 데이터가 있는지 확인합니다.
```Python
data['hospital_id'].value_counts()          #hospital_id 고유값 분포 확인
```
```
118    4333
19     3925
188    3095
161    2792
70     2754
       ... 
23        7
4         7
93        6
95        6
130       2
Name: hospital_id, Length: 147, dtype: int64
```
```Python
data['icu_id'].value_counts()          #icu_id 고유값 분포 확인
```
```
646    1325
653    1307
876    1284
413    1239
236    1140
       ... 
494       3
365       2
302       2
603       2
241       1
Name: icu_id, Length: 241, dtype: int64
```
위 분포와 해당 id의 description을 미루어 볼때, 두 id 모두 병원, 입원실과 관련된 고유 id임을 확인할 수 있습니다.
<br> 따라서 <code>id_list</code>에 해당되는 네 id feature는 학습 모델에 포함시키기 적절하지 않으므로 데이터셋에서 제외합니다.

```Python
data.drop(labels=id_list, axis=1, inplace=True)
print(data)
```
```
        age        bmi  ...  apache_2_bodysystem hospital_death
0      68.0  22.730000  ...       Cardiovascular              0
1      77.0  27.420000  ...          Respiratory              0
2      25.0  31.950000  ...            Metabolic              0
3      81.0  22.640000  ...       Cardiovascular              0
4      19.0        NaN  ...               Trauma              0
...     ...        ...  ...                  ...            ...
91708  75.0  23.060250  ...       Cardiovascular              0
91709  56.0  47.179671  ...       Cardiovascular              0
91710  48.0  27.236914  ...            Metabolic              0
91711   NaN  23.297481  ...          Respiratory              0
91712  82.0  22.031250  ...     Gastrointestinal              0

[91713 rows x 80 columns]
```

추가로 인종 'ethnicity' 열도 제거합니다. 과학적으로 유의미한 feature가 될 가능성이 없진 않지만, 아래의 그래프와 같이 인종은 사망률에 큰 영향을 주지 않아 보입니다.
```Python
data['ethnicity'].nunique()    #인종 종류 수 확인
>>> 6
eth_group = data.groupby(['hospital_death','ethnicity'])    #인종과 사망여부 그룹화
eth = dict(data['ethnicity'].value_counts())
sorted_eth = sorted(eth.items())
death = dict(eth_groups.size()[6:])
sorted_death = sorted(death.items())    #인종과 사망여부 순서 정렬
x_eth = []
y_death_ratio =[]
for i in range(6):
    y_death_ratio.append(sorted_death[i][1]/sorted_eth[i][1])    #사망률
    x_eth.append(sorted_eth[i][0])    #인종
plt.figure(figsize=(15,5))    #그래프 그리기
plt.ylim([0,1])
plt.plot(x_eth,y_death_ratio,'-x')
plt.title('Death Ratio by ethnicity', fontsize=15)
plt.savefig('eth.png')
```
![eth](./img/eth.png)

따라서 ethnicity 항목은 제거합니다.
```Python
data.drop('ethnicity', axis=1, inplace=True)
```


### III - 2.  EDA

각 변수 간 상관관계를 살펴보기 위해 히트맵을 그립니다.
```Python
plt.figure(figsize=(20,20))
plt.title("Correlation Heatmap", y = 1.05, size = 15)
sns.heatmap(data.corr(),cmap='RdBu')
```
![htmap](./img/htmap.png)

가장 눈에 띄는 점은 가운데 보이는 파란색 구역들입니다.
<br> 이들의 공통점은 이름의 '_noninvasive_'라는 단어가 들어간 변수라는 것입니다.
<br> 따라서 이들의 상관관계를 좀 더 자세히 확인합니다.
```Python
for name in data.columns:    #'_noninvasive'가 포함된 변수 찾기
    if '_noninvasive' in name:
        non_invasive_list.append(name)
print(non_invasive_list)
```
```
['d1_diasbp_noninvasive_max', 'd1_diasbp_noninvasive_min', 'd1_mbp_noninvasive_max', 'd1_mbp_noninvasive_min', 'd1_sysbp_noninvasive_max', 'd1_sysbp_noninvasive_min', 'h1_diasbp_noninvasive_max', 'h1_diasbp_noninvasive_min', 'h1_mbp_noninvasive_max', 'h1_mbp_noninvasive_min', 'h1_sysbp_noninvasive_max', 'h1_sysbp_noninvasive_min']
```
이제 상관관계를 파악하기 위한 리스트를 따로 만들어 준 뒤, 상관관계 계수가 포함된 히트맵을 출력합니다.
```Python
df_noninv=data[['d1_diasbp_max','d1_diasbp_min','d1_mbp_max', 'd1_mbp_min', 'd1_sysbp_max', 'd1_sysbp_min', 'h1_diasbp_max', 'h1_diasbp_min', 'h1_mbp_max', 'h1_mbp_min', 'h1_sysbp_max', 'h1_sysbp_min','d1_diasbp_noninvasive_max', 'd1_diasbp_noninvasive_min', 'd1_mbp_noninvasive_max', 'd1_mbp_noninvasive_min', 'd1_sysbp_noninvasive_max', 'd1_sysbp_noninvasive_min', 'h1_diasbp_noninvasive_max', 'h1_diasbp_noninvasive_min', 'h1_mbp_noninvasive_max', 'h1_mbp_noninvasive_min', 'h1_sysbp_noninvasive_max', 'h1_sysbp_noninvasive_min']]
plt.figure(figsize=(20,20))
sns.heatmap(df_noninv.corr(),annot=True, fmt=".1f", cmap='Blues')
plt.savefig('noninv.png')
```
![noninv](./img/noninv.png)

위 히트맵의 상관계수 '1.0'의 위치를 따져봤을 때, "XX_noninvasive_YY"는 변수 "XX_YY"와 완전히 같은 데이터임을 확인할 수 있습니다.
<br> 차이는 noninvasive 쪽에 추가 결측치가 있다는 것인데, 예시로 'd1_diasbp_max'와 'd1_diasbp_noninvasive_max'의 결측치를 시각화하면

![ex](./img/ex.png)

결론은, 'XX_noninvasive' 종류의 데이터는, 결측치만 추가로 존재할 뿐, 'XX'와 100% 일치하는 데이터라는 것입니다.
<br> 즉, 해당 종류의 데이터는 전부 삭제해도 무방할듯 합니다.

```Python
data.drop(labels=non_invasive_list, axis=1, inplace=True)
```
현재까지 정리한 데이터 셋의 결측치 현황을 점검해봅니다.
```R
sum(complete.cases(data))    #결측치가 없는 행의 개수
>>> 60909
```
초기에는 결측치가 없는 행의 개수가 56935개 였으니, 위의 필터링 과정을 통해 모델에 쓰일 '완전한' 행의 개수를 조금 늘리게 된 성과를 얻었다 할 수 있겠습니다.
<br>------------------------------------------------------------------------------------  
이제 for 루프를 사용해서 데이터 마다 그래프를 그려보겠습니다.
```Python
for i in data.columns:
    if data[i].value_counts().shape[0]>20:
        plt.figure(figsize=(12,8))
        sns.distplot(data[i][data['hospital_death']==0],color='g', label='Survive ',hist_kws={'edgecolor':'black'})
        _=sns.distplot(data[i][data['hospital_death']==1],color='r',label='Death',hist_kws={'edgecolor':'black'})
        plt.legend()
        plt.title(f'{dict_description[i]}', fontsize=8)
        plt.savefig(str(i)+'.pdf')
    else:
        plt.figure(figsize=(14, 6))
        sns.countplot(x=i, hue="hospital_death", data=data, palette='coolwarm')
        plt.legend(loc='upper right')
        plt.yscale('log')
        plt.xticks(rotation=45)
        plt.title(f'{dict_description[i]}', fontsize=8)
        plt.savefig(str(i)+'.pdf')
```
예를 들어 정수형 데이터인 'age'(나이)는 아래와 같은 밀도 plot을 그렸습니다. (+실수형 데이터)
<br>녹색선은 생존한 사람의 나이별 분포, 적색선은 사망한 사람의 나이별 분포입니다. 상대적으로 고령에서 적색선이 우세한 것을 대략적으로 확인할 수 있습니다.
![age](./img/age.png)

'gender'(성별)과 같은 카테고리형(또는 불리언형) 데이터는 아래와 같은 count plot을 그렸습니다.
![gender](./img/gender.PNG)

나머지 데이터에 대한 플롯은 ![plt_merged.pdf](./plt_merged.pdf)에 담겨있습니다.

### III - 3.  Making final dataframe

모델에 사용할 데이터프레임을 만들기 위해 마지막으로 몇 가지 작업을 합니다.
```Python
#제거할 feature들 목록 생성
drop_list = ['icu_type', 'pre_icu_los_days', 'weight', 'height', 'gcs_unable_apache', 'd1_spo2_max', 'h1_spo2_max', 'apache_4a_icu_death_prob', 'apache_2_bodysystem', 'apache_3j_bodysystem']
data.drop(labels=drop_list, axis=1, inplace=True)    #해당 feature들 제거
```
위 feature들은 위에서 그렸던 플롯과 변수 특성을 고려하여 삭제를 결정했습니다.
<br>사실상 고유 식별자(ID)의 역할을 하는 변수거나, 키, 체중과 같이 BMI 변수에 이미 포함되어 있는 중복 변수거나, 의미를 이해할 수 없는(ex. 음수가 존재할 수 없는데 존재한다던가) 변수들입니다.
<br>모델의 정확성 향상에 기여하는 dimensionality reduction을 위해 제거하였습니다.
<br>  
<br>이제 결측값을 포함하는 행을 제거합니다.
```Python
df = data.dropna(axis=0)    #결측치가 존재하는 행 제거
df.reset_index(drop=True, inplace=True)    #인덱스 초기화
df.info()
```
```Python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 60909 entries, 0 to 60908
Data columns (total 57 columns):
 #   Column                         Non-Null Count  Dtype  
---  ------                         --------------  -----  
 0   age                            60909 non-null  float64
 1   bmi                            60909 non-null  float64
 2   elective_surgery               60909 non-null  int64  
 3   gender                         60909 non-null  object 
 4   icu_admit_source               60909 non-null  object 
 5   icu_stay_type                  60909 non-null  object 
 6   apache_2_diagnosis             60909 non-null  float64
 7   apache_3j_diagnosis            60909 non-null  float64
...
 51  hepatic_failure                60909 non-null  float64
 52  immunosuppression              60909 non-null  float64
 53  leukemia                       60909 non-null  float64
 54  lymphoma                       60909 non-null  float64
 55  solid_tumor_with_metastasis    60909 non-null  float64
 56  hospital_death                 60909 non-null  int64  
dtypes: float64(51), int64(3), object(3)
memory usage: 26.5+ MB
```
최종적으로 57개 feature와 60909개의 결측치 없는 행이 남았음을 확인할 수 있습니다.

```Python
df.hospital_death.value_counts().plot(kind='pie',autopct="%.2f",title ='Mortality in hospital(%)')
plt.savefig('mortality.png')
```
모델의 사용할 최종 데이터에서 hospital_death가 1인 비율이 어느 정도인지 파악합니다.
![mortality](./img/mortality.png)
<br>원본 데이터의 비율과 크게 다르지 않음을 확인할 수 있습니다.


### III - 4.  Explaining algorithms

본 프로젝트에선 다음의 알고리즘을 선택했습니다.

#### 의사결정나무(Decision Tree)
![dt](./img/dt.png)
데이터에 존재하는 임의의 규칙과 기준을, 학습을 통해 그림과 같은 Tree 형태의 분류 규칙을 만들어내는 모델을 의사결정나무라고 합니다.
이 모델의 장점은 일반적으로 feature들을 정규화하거나 스케일링하는 작업이 필요없다는 것 입니다.
<br>또한, 수치형과 범주형 데이터를 모두 사용할 수 있어, One Hot Encoding과 같은 별도의 범주형 데이터 처리가 불필요하는 것도 장점이겠습니다.

## IV. Models & Evaluation

#### 레이블 인코딩(Label encoding)
먼저, 모델에 적용하기 전 레이블 인코딩(Label encoding) 과정을 거쳐야합니다.
<br>사이킷런은 범주형 데이터의 문자열 값(dType = object)을 입력 받지 못하기 때문에, 이를 모두 숫자로 바꿔줘야 합니다. 
<br>레이블 인코딩(Label encoding)은 문자열인 유니크한 값들을 단순히 순서대로 정렬한 뒤, 각각에 0 부터 1씩 증가하는 값을 부여해 숫자 데이터로 변환하는 과정입니다.
<br>*다만, 이러한 숫자 부여 방식은 트리 모델류에서만 유용하며, 숫자 크기 자체가 영향을 주는 선형 모델(로지스틱 회귀, 신경망 등)에는 사용할 수 없습니다. (이 경우, one hot encoding이란 방식을 사용해야합니다.)*
```Python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 60909 entries, 0 to 60908
Data columns (total 57 columns):
 #   Column                         Non-Null Count  Dtype  
---  ------                         --------------  -----  
 0   age                            60909 non-null  float64
 1   bmi                            60909 non-null  float64
 2   elective_surgery               60909 non-null  int64  
 3   gender                         60909 non-null  object 
 4   icu_admit_source               60909 non-null  object 
 5   icu_stay_type                  60909 non-null  object 
 6   apache_2_diagnosis             60909 non-null  float64
 ...
 dtypes: float64(51), int64(3), object(3)
memory usage: 26.5+ MB
```
df.info()로 살펴본 바로는, 현재 범주형 데이터 중 문자열(object)을 가지는 feature는 3 가지 존재합니다.
<br> 'gender', 'icu_admit_source', 'icu_stay_type' 이 세 가지입니다.
```Python
from sklearn.preprocessing import LabelEncoder     #모듈 import
# 'gender' 인코딩
LE = LabelEncoder()
LE.fit(df['gender'])
labels = LE.transform(df['gender'])
print(labels)
>>>[1 0 0 ... 1 0 0]
# 'icu_admit_source' 인코딩
LE.fit(df['icu_admit_source'])
labels2 = LE.transform(df['icu_admit_source'])
print(labels2)
>>>[1 1 2 ... 1 1 2]
# 'icu_stay_type' 인코딩
LE.fit(df['icu_stay_type'])
labels3 = LE.transform(df['icu_stay_type'])
print(labels3)
>>>[0 0 0 ... 0 0 0]
#기존값 대체
df['gender'] = labels
df['icu_admit_source'] = labels2
df['icu_stay_type'] = labels3
```
#### 의사결정나무(Decision Tree)
먼저, 데이터를 train, test 그룹으로, 7대3 비율로 나눕니다.
```Python
from sklearn.model_selection import train_test_split
y = df.hospital_death    #타겟=Hospital Death
x = df.drop('hospital_death',axis=1)     #타겟 제외 피쳐들
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y , random_state=1)    #7:3비율로 트레이닝, 테스팅 세트 분할
x_train.shape, x_test.shape, y_train.shape, y_test.shape    #확인
>>>((42636, 56), (18273, 56), (42636,), (18273,))
```
이제 모델에 데이터를 학습시킨 뒤, 예측합니다!
```Python
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)    #트레이닝 세트 학습 시키기
```
```Python
from sklearn.metrics import accuracy_score
predict_train = tree.predict(x_train)     #트레이닝 세트 예측
predict_test = tree.predict(x_test)    #테스트 세트 예측
accuracy_train = accuracy_score(y_train, predict_train)
accuracy_test = accuracy_score(y_test, predict_test)
accuracy_train    #예측 모델의 트레이닝 세트 예측 점수(즉, 1이어야 정상)
>>>1.0
accuracy_test    #예측 모델의 테스트 세트 예측 점수
>>>0.8856235976577465
```
즉, **의사결정나무** 알고리즘으로 학습했을 때, 예측 점수 **88.56%의 정확도**를 보였습니다. 
<br>  
<br> 마지막으로 Confusion Matrix를 구한 뒤, 정밀도(Precision)와 재현률(Recall), F1점수를 계산하겠습니다.
```Python
#Confusion Matrix 시각화
from sklearn.metrics import confusion_matrix
CM = confusion_matrix(y_test, predict_test)
CM_df = pd.DataFrame(CM, columns=['Alive', 'Dead'], index=['Alive', 'Dead'])    #예측 모델의 Confision Matrix 데이터프레임
sns.heatmap(CM_df,annot=True, fmt="d", cmap='PuBu')    #시각화
plt.ylabel('True Value', fontsize=15)
plt.xlabel('Prediction', fontsize=15)
plt.title('Confusion Matrix(Decision Tree)', fontsize=18)    #축 레이블과 제목 추가
plt.savefig('dtcm.png')
```
![dtcm](./img/dtcm.png)
```Python
#Precision, Recall, F1 score 계산
from sklearn.metrics import precision_score, recall_score, f1_score
p = precision_score(y_test, predict_test)
r = recall_score(y_test, predict_test)
f1 = f1_score(y_test, predict_test)
print(f'#정밀도 : {p:.2f}'+f'  #재현률 : {r:.2f}'+f'  #f1스코어 : {f1:.2f}')
>>> #정밀도 : 0.33  #재현률 : 0.35  #f1스코어 : 0.34
```
## V. Related Work 
- Tools, libraries, blogs, or any documentation that you have used to do this project.
- https://www.kaggle.com/datasets/mitishaagarwal/patient (Dataset)
- https://www.kaggle.com/code/mrisdal/exploring-survival-on-the-titanic/report (전체적인 방향성 참고)
- https://www.kaggle.com/code/danielwarda/patient-survival-prediction-eda-part-1 (데이터 plot 코드 참고)
- https://ysyblog.tistory.com/71 (Label Encoding)
- https://ysyblog.tistory.com/68 (Decision Tree 코딩)
- https://regenerativetoday.com/simple-explanation-on-how-decision-tree-algorithm-makes-decisions/ (Decision Tree 설명)
- https://chrisalbon.com/code/python/data_visualization/seaborn_color_palettes/ (sns plot)
- http://www.gisdeveloper.co.kr/?p=9932 (Confusion Matrix 코드 참고)

## VI. Conclusion & Discussion
#### Abstract
<br>'입원 중 환자의 생존률'의 초기 **84개의 features** 를 분석하고, 차원을 축소시키는 데 많은 시간을 할애했습니다.
<br>다음의 조건 중 하나라도 해당한다면 그 열을 삭제하였습니다.
- 고유 식별자(ID)라 학습 모델에 영향을 끼치지 않음
- 다른 feature와 거의 일치하거나, 이미 포함되어 있는 중복된 feature임
- 음수가 포함될 수 없는 데이터인데 음수가 포함됨
- 의미를 이해할 수 없거나, 정상적인 데이터가 아니라고 판단됨

이 과정을 통해 최종적으로 28개의 열을 삭제하여 **56개의 feature로 축소** 시켰으며, <br>결과적으로 결측치가 존재하는 행도 34778개에서 30804개가 되어 **총 3974개의 행을 살렸습니다.**
<br>그 후, 데이터셋을 7:3의 비율로 학습, 시험 세트를 나눈 뒤, Decision Tree 알고리즘으로 학습시킨 결과, 시험 세트에 대해
<br> **모델 정확도 88.56%** 를 얻었습니다. 또한, **precision 점수** 는 **33%** , **recall 점수** 는 **35%** , 그리고  **f1 점수** 는 **34%** 를 얻었습니다.

#### Discussion
<br>Python과 R을 ML에 적용하는 방법을 알게 되었습니다. Decision Tree 알고리즘의 특성과 원리를 이해할 수 있었습니다. 이번 프로젝트에는 트리 기반 모델만을 사용했지만, 다음 기회에는 one hot encoding 기법을 사용해 선형 모델(로지스틱, SVM, 신경망) 알고리즘에도 본 데이터셋을 적용하여 더욱 의미있는 프로젝트가 되었음 합니다.

<br> 차도현, 물리학과, chamena1110@gmail.com (데이터셋 탐색, feature 분석, 데이터셋 가공, 코드 작성 및 실행, 블로그 작성, 유튜브 촬영)
<br> Khaliun, 경제금융학부, haliunurgamaltuya@gmail.com (feature 분석 도움)
