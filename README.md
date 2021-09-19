# semi-supervised learning
implementation of the semi-supervised learning algorithms
## self_training.py
implementation of the self-training algorithm

[[pdf]](https://aclanthology.org/P95-1026.pdf)D. Yarowsky, "Unsupervised word sense disambiguation rivaling supervised methods," 

In 33rd annual meeting of the association for computational linguistics, pp. 189-196, 1995.
## co_training.py
implementation of the co-training algorithm

[[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5560662&casa_token=M2Z5V2aSgEgAAAAA:NNfl9aTxrBg6xeXSgk0SOLOJpAiS7NE6ymNy2hKDJQz_SYopAWge1vzWYCTE7WybC77iSrnu3Q&tag=1)
J. Du, C. X. Ling and Z. H. Zhou, "When does cotraining work in real data," 

IEEE Transactions on Knowledge and Data Engineering, vol. 23, no. 5, pp. 788-799, 2010.
## tri_training.py
implementation of the tri-training algorithm

[[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=1512038&casa_token=gmWWpqv6uUwAAAAA:dcDXh2Td2Vwba4zHeS_T4szVqfuCtVTypnQgZTSwR_fUUxnC5PMsngL2aUiOPBR66auhym0oTQ)
Z.-H. Zhou and M. Li., "Tri-training: Exploiting unlabeled data using three classifiers," 

IEEE Transactions on knowledge and Data Engineering, vol. 17, no. 11, pp. 1529-1541, 2005.
## multi_train.py
implementation of the multi-train algorithm

[[pdf]](https://www.sciencedirect.com/science/article/pii/S0925231217306094?casa_token=o6S-EB7_wy8AAAAA:HOlDOxKorsjDIynGhvUkd_DsugEYp3fVPqjYA5tyy4VxQBc5UQ1yOPWwu-uxijJ2jYFdI0FstkQ)
S. Gu and Y. Jin, "Multi-train: A semi-supervised heterogeneous ensemble classifier," 

Neurocomputing, vol. 249, pp. 202-211, 2017.
## two_phase_auto_fit.py
improve the tri-training algorithm, using second model to learning confidences of base classifiers,

and auto select classification threshold when predict unlabeled data.
## Run the demo
use the defalut setting
```
python two_phase_auto_fit.py
```
use optional parameter
```
python two_phase_auto_fit.py --experiment_num 5
```
use nargs = "+" parameter
```
python two_phase_auto_fit.py --label_rates 0.1 0.2
```
use store_true parameter
```
python two_phase_auto_fit.py --use_auto_select_threshold
```
## optional parameters
<table class=MsoTableGrid border=1 cellspacing=0 cellpadding=0 width=642
 style='width:481.7pt;border-collapse:collapse;border:none;mso-border-alt:solid windowtext .5pt;
 mso-yfti-tbllook:1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt'>
 <tr style='mso-yfti-irow:0;mso-yfti-firstrow:yes'>
  <td width=642 colspan=3 valign=top style='width:481.7pt;border:solid windowtext 1.0pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><b style='mso-bidi-font-weight:normal'><span lang=EN-US
  style='font-size:14.0pt;mso-bidi-font-size:11.0pt'>common parameters<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>parameter</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>default</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>description </span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:2'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>train_size: float</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>0.75</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>the size of the training dataset</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:3'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>label_rates: float</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>[0.1]</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>nargs = '+'</span></p>
  <p class=MsoNormal><span lang=EN-US>the size of the labeled data = train_size
  * label_rate</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:4'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>experiment_num: int</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>10</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>number of experiments</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:5'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>data_dir: str</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>None</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>if data_dir == None: read data from
  sklearn.datasets package</span></p>
  <p class=MsoNormal><span lang=EN-US>else: read csv data from data directory</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:6'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>log_dir: str</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>'\log\<i>method_<o:p></o:p></i></span></p>
  <p class=MsoNormal><i><span lang=EN-US>name_time</span></i><span lang=EN-US>'</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>log directory</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:7'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>avg_types: str</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>['weighted']</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>nargs = '+'</span></p>
  <p class=MsoNormal><span lang=EN-US>choices = ['micro', 'macro', 'weighted']</span></p>
  <p class=MsoNormal><span lang=EN-US>micro =&gt; micro average</span></p>
  <p class=MsoNormal><span lang=EN-US>macro =&gt; macro average</span></p>
  <p class=MsoNormal><span lang=EN-US>weighted =&gt; weighted macro average</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:8'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>save_pred: bool</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>False</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>action = 'store_true'</span></p>
  <p class=MsoNormal><span lang=EN-US>whether to save prediction and number of
  unlabeled</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:9'>
  <td width=642 colspan=3 valign=top style='width:481.7pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><b style='mso-bidi-font-weight:normal'><span lang=EN-US
  style='font-size:14.0pt;mso-bidi-font-size:11.0pt'>self_training<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:10'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>parameter</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>default</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>description </span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:11'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>groups: str</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>['NB']</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>nargs = '+'</span></p>
  <p class=MsoNormal><span lang=EN-US>choices = ['NB', 'SVM', 'RF', 'AdaBoost',
  'KNN', 'DT']</span></p>
  <p class=MsoNormal><span lang=EN-US>the type of base classifier </span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:12'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>confidence_thresholds: float</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>[0.1]</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>nargs = '+' </span></p>
  <p class=MsoNormal><span lang=EN-US>the confidence threshold which decide
  whether to add unlabeled data</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:13'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>use_unlabeled_pool: bool</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>False</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>action = 'store_true'</span></p>
  <p class=MsoNormal><span lang=EN-US>whether to use the unlabeled pool</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:14'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>pool_size: int</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>75</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>the size of the unlabeled pool</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:15'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>k: int</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>30</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>number of iterations of self-training
  algorithm</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:16'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>data_pre_type: str</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>'category_and_numeric'</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>choices = ['all_category', 'category_and_numeric']</span></p>
  <p class=MsoNormal><span lang=EN-US>category =&gt; 10 equal-width bins</span></p>
  <p class=MsoNormal><span lang=EN-US>numeric =&gt; log</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:17'>
  <td width=642 colspan=3 valign=top style='width:481.7pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><b style='mso-bidi-font-weight:normal'><span lang=EN-US
  style='font-size:14.0pt;mso-bidi-font-size:11.0pt'>co_training<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:18'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>parameter</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>default</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>description </span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:19'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>split_types: str</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>['entropy_split']</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>nargs = '+'</span></p>
  <p class=MsoNormal><span lang=EN-US>choices = ['random_split', 'entropy_split',
  'entropy_hill', 'random_hill']</span></p>
  <p class=MsoNormal><span lang=EN-US>the method which split the single
  view(feature subset) into two views</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:20'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>split_small: bool</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>False</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>action = 'store_true'</span></p>
  <p class=MsoNormal><span lang=EN-US>whether split view only based on labeled
  data or whole dataset</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:21'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>groups: str</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>['NB']</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>nargs = '+'</span></p>
  <p class=MsoNormal><span lang=EN-US>choices = ['NB', 'SVM', 'RF', 'AdaBoost',
  'KNN', 'DT']</span></p>
  <p class=MsoNormal><span lang=EN-US>the two base classifiers use the same
  type of classifier</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:22'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>use_unlabeled_pool: bool</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>False</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>action = 'store_true'</span></p>
  <p class=MsoNormal><span lang=EN-US>whether to use the unlabeled pool</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:23'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>pool_size: int</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>75</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>the size of the unlabeled pool</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:24'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>k: int</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>30</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>number of iterations of self-training
  algorithm</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:25'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>data_pre_type: str</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>'all_category'</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>choices = ['all_category']</span></p>
  <p class=MsoNormal><span lang=EN-US>category =&gt; 10 equal-width bins</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:26'>
  <td width=642 colspan=3 valign=top style='width:481.7pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><b style='mso-bidi-font-weight:normal'><span lang=EN-US
  style='font-size:14.0pt;mso-bidi-font-size:11.0pt'>tri_training<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:27'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>parameter</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>default</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>description </span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:28'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>groups: str</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>['NB']</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>nargs = '+'</span></p>
  <p class=MsoNormal><span lang=EN-US>choices = ['NB', 'SVM', 'RF', 'AdaBoost',
  'KNN', 'DT']</span></p>
  <p class=MsoNormal><span lang=EN-US>the three base classifiers use the same
  type of classifier</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:29'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>data_pre_type: str</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>'category_and_numeric'</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>choices = ['all_category', 'category_and_numeric']</span></p>
  <p class=MsoNormal><span lang=EN-US>category =&gt; 10 equal-width bins</span></p>
  <p class=MsoNormal><span lang=EN-US>numeric =&gt; log</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:30'>
  <td width=642 colspan=3 valign=top style='width:481.7pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><b style='mso-bidi-font-weight:normal'><span lang=EN-US
  style='font-size:14.0pt;mso-bidi-font-size:11.0pt'>multi_train<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:31'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>parameter</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>default</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>description </span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:32'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>groups: str</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>['NB_AdaBoost_DT']</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>nargs = '+'</span></p>
  <p class=MsoNormal><span lang=EN-US>the type of base classifiers can be
  different</span></p>
  <p class=MsoNormal><span lang=EN-US>use _ to separate the classifier type</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:33'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>confidence_thresholds: float</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>[0.1]</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>nargs = '+' </span></p>
  <p class=MsoNormal><span lang=EN-US>the confidence threshold which decide
  whether to add unlabeled data</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:34'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>feature_pre: str</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>None</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>choices = [None, 'PCA']</span></p>
  <p class=MsoNormal><span lang=EN-US>use principal component analysis for
  transforming the features</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:35'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>data_pre_type: str</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>'category_and_numeric'</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>choices = ['all_category', 'category_and_numeric']</span></p>
  <p class=MsoNormal><span lang=EN-US>category =&gt; 10 equal-width bins</span></p>
  <p class=MsoNormal><span lang=EN-US>numeric =&gt; log</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:36'>
  <td width=642 colspan=3 valign=top style='width:481.7pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><b style='mso-bidi-font-weight:normal'><span lang=EN-US
  style='font-size:14.0pt;mso-bidi-font-size:11.0pt'>two_phase_auto_fit<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:37'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>parameter</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>default</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>description </span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:38'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>groups: str</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>['NB_AdaBoost_DT']</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>nargs = '+'</span></p>
  <p class=MsoNormal><span lang=EN-US>the type of base classifiers can be
  different</span></p>
  <p class=MsoNormal><span lang=EN-US>use _ to separate the classifier type</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:39'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>use_auto_select_threshold: bool</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>False</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>action = 'store_true'</span></p>
  <p class=MsoNormal><span lang=EN-US>whether to use auto select threshold when
  predict unlabeled data</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:40'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>confidence_thresholds: float</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>[0.1]</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>nargs = '+' </span></p>
  <p class=MsoNormal><span lang=EN-US>the confidence threshold which decide
  whether to add unlabeled data</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:41'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>feature_pre: str</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>None</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>choices = [None, 'PCA']</span></p>
  <p class=MsoNormal><span lang=EN-US>use principal component analysis for
  transforming the features</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:42;mso-yfti-lastrow:yes'>
  <td width=195 valign=top style='width:146.4pt;border:solid windowtext 1.0pt;
  border-top:none;mso-border-top-alt:solid windowtext .5pt;mso-border-alt:solid windowtext .5pt;
  padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>data_pre_type: str</span></p>
  </td>
  <td width=172 valign=top style='width:128.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>'category_and_numeric'</span></p>
  </td>
  <td width=276 valign=top style='width:206.65pt;border-top:none;border-left:
  none;border-bottom:solid windowtext 1.0pt;border-right:solid windowtext 1.0pt;
  mso-border-top-alt:solid windowtext .5pt;mso-border-left-alt:solid windowtext .5pt;
  mso-border-alt:solid windowtext .5pt;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>choices = ['all_category', 'category_and_numeric']</span></p>
  <p class=MsoNormal><span lang=EN-US>category =&gt; 10 equal-width bins</span></p>
  <p class=MsoNormal><span lang=EN-US>numeric =&gt; log</span></p>
  </td>
 </tr>
</table>
