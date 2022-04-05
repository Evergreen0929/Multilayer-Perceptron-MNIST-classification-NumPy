# Multilayer-Perceptron-MNIST-classification-NumPy
Class project： Realizing image classification of MNIST by MLP on the basis of NumPy.


## Preparations

### Requirements 
>python = 3.6.13  
>NumPy = 1.19.2  

**for visualization:** matplotlib = 3.3.4, PIL = 8.2.0, sklearn = 0.24.2  

### Dataset 
The MNIST can be downloaded from [official webset](http://yann.lecun.com/exdb/mnist/) or from [https://pan.baidu.com/s/1s8HoOEW_cBaVtq8_rQRHyA](https://pan.baidu.com/s/1s8HoOEW_cBaVtq8_rQRHyA), (pwd: rxor). If you use the baidu netdesk url, directly unzip the file at root dir is acceptable. Else, put the four files (train-images-idx3-ubyte, train-labels-idx1-ubyte, t10k-images-idx3-ubyte, t10k-labels-idx1-ubyte) in `./mnist/`. 

### Pre-trained weights
I have pretrained models with different sizes, the number of neurons in the hidden layer range from 100 to 900. You can choose the pre-trained weights you need to make inference or train from scratch.  
The pretrained-models can be downloaded from [https://pan.baidu.com/s/1SYb54A2h098INL1qKwefAw](https://pan.baidu.com/s/1SYb54A2h098INL1qKwefAw), (pwd: 40yz). The downloaded weights should be placed in `./save_model/`.

<table class=MsoTable15Plain1 border=1 cellspacing=0 cellpadding=0
 style='border-collapse:collapse;border:none;mso-border-alt:solid #BFBFBF .5pt;
 mso-border-themecolor:background1;mso-border-themeshade:191;mso-yfti-tbllook:
 1184;mso-padding-alt:0cm 5.4pt 0cm 5.4pt'>
 <tr style='mso-yfti-irow:-1;mso-yfti-firstrow:yes;mso-yfti-lastfirstrow:yes'>
  <td width=138 valign=top style='width:103.7pt;border:solid #BFBFBF 1.0pt;
  mso-border-themecolor:background1;mso-border-themeshade:191;mso-border-alt:
  solid #BFBFBF .5pt;mso-border-themecolor:background1;mso-border-themeshade:
  191;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal style='mso-yfti-cnfc:5'><b><span lang=EN-US>Hidden Nodes<o:p></o:p></span></b></p>
  </td>
  <td width=138 valign=top style='width:103.7pt;border:solid #BFBFBF 1.0pt;
  mso-border-themecolor:background1;mso-border-themeshade:191;border-left:none;
  mso-border-left-alt:solid #BFBFBF .5pt;mso-border-left-themecolor:background1;
  mso-border-left-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal style='mso-yfti-cnfc:1'><b><span lang=EN-US>Acc<o:p></o:p></span></b></p>
  </td>
  <td width=138 valign=top style='width:103.7pt;border:solid #BFBFBF 1.0pt;
  mso-border-themecolor:background1;mso-border-themeshade:191;border-left:none;
  mso-border-left-alt:solid #BFBFBF .5pt;mso-border-left-themecolor:background1;
  mso-border-left-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal style='mso-yfti-cnfc:1'><b><span lang=EN-US>Hidden Nodes<o:p></o:p></span></b></p>
  </td>
  <td width=138 valign=top style='width:103.7pt;border:solid #BFBFBF 1.0pt;
  mso-border-themecolor:background1;mso-border-themeshade:191;border-left:none;
  mso-border-left-alt:solid #BFBFBF .5pt;mso-border-left-themecolor:background1;
  mso-border-left-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal style='mso-yfti-cnfc:1'><b><span lang=EN-US>Acc<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:0'>
  <td width=138 valign=top style='width:103.7pt;border:solid #BFBFBF 1.0pt;
  mso-border-themecolor:background1;mso-border-themeshade:191;border-top:none;
  mso-border-top-alt:solid #BFBFBF .5pt;mso-border-top-themecolor:background1;
  mso-border-top-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;background:#F2F2F2;mso-background-themecolor:
  background1;mso-background-themeshade:242;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal style='mso-yfti-cnfc:68'><span lang=EN-US
  style='mso-bidi-font-weight:bold'>100<o:p></o:p></span></p>
  </td>
  <td width=138 valign=top style='width:103.7pt;border-top:none;border-left:
  none;border-bottom:solid #BFBFBF 1.0pt;mso-border-bottom-themecolor:background1;
  mso-border-bottom-themeshade:191;border-right:solid #BFBFBF 1.0pt;mso-border-right-themecolor:
  background1;mso-border-right-themeshade:191;mso-border-top-alt:solid #BFBFBF .5pt;
  mso-border-top-themecolor:background1;mso-border-top-themeshade:191;
  mso-border-left-alt:solid #BFBFBF .5pt;mso-border-left-themecolor:background1;
  mso-border-left-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;background:#F2F2F2;mso-background-themecolor:
  background1;mso-background-themeshade:242;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal style='mso-yfti-cnfc:64'><span lang=EN-US>0.9759</span></p>
  </td>
  <td width=138 valign=top style='width:103.7pt;border-top:none;border-left:
  none;border-bottom:solid #BFBFBF 1.0pt;mso-border-bottom-themecolor:background1;
  mso-border-bottom-themeshade:191;border-right:solid #BFBFBF 1.0pt;mso-border-right-themecolor:
  background1;mso-border-right-themeshade:191;mso-border-top-alt:solid #BFBFBF .5pt;
  mso-border-top-themecolor:background1;mso-border-top-themeshade:191;
  mso-border-left-alt:solid #BFBFBF .5pt;mso-border-left-themecolor:background1;
  mso-border-left-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;background:#F2F2F2;mso-background-themecolor:
  background1;mso-background-themeshade:242;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal style='mso-yfti-cnfc:64'><span lang=EN-US>600</span></p>
  </td>
  <td width=138 valign=top style='width:103.7pt;border-top:none;border-left:
  none;border-bottom:solid #BFBFBF 1.0pt;mso-border-bottom-themecolor:background1;
  mso-border-bottom-themeshade:191;border-right:solid #BFBFBF 1.0pt;mso-border-right-themecolor:
  background1;mso-border-right-themeshade:191;mso-border-top-alt:solid #BFBFBF .5pt;
  mso-border-top-themecolor:background1;mso-border-top-themeshade:191;
  mso-border-left-alt:solid #BFBFBF .5pt;mso-border-left-themecolor:background1;
  mso-border-left-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;background:#F2F2F2;mso-background-themecolor:
  background1;mso-background-themeshade:242;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal style='mso-yfti-cnfc:64'><span lang=EN-US>0.9840</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:1'>
  <td width=138 valign=top style='width:103.7pt;border:solid #BFBFBF 1.0pt;
  mso-border-themecolor:background1;mso-border-themeshade:191;border-top:none;
  mso-border-top-alt:solid #BFBFBF .5pt;mso-border-top-themecolor:background1;
  mso-border-top-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal style='mso-yfti-cnfc:4'><span lang=EN-US style='mso-bidi-font-weight:
  bold'>200<o:p></o:p></span></p>
  </td>
  <td width=138 valign=top style='width:103.7pt;border-top:none;border-left:
  none;border-bottom:solid #BFBFBF 1.0pt;mso-border-bottom-themecolor:background1;
  mso-border-bottom-themeshade:191;border-right:solid #BFBFBF 1.0pt;mso-border-right-themecolor:
  background1;mso-border-right-themeshade:191;mso-border-top-alt:solid #BFBFBF .5pt;
  mso-border-top-themecolor:background1;mso-border-top-themeshade:191;
  mso-border-left-alt:solid #BFBFBF .5pt;mso-border-left-themecolor:background1;
  mso-border-left-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>0.9809</span></p>
  </td>
  <td width=138 valign=top style='width:103.7pt;border-top:none;border-left:
  none;border-bottom:solid #BFBFBF 1.0pt;mso-border-bottom-themecolor:background1;
  mso-border-bottom-themeshade:191;border-right:solid #BFBFBF 1.0pt;mso-border-right-themecolor:
  background1;mso-border-right-themeshade:191;mso-border-top-alt:solid #BFBFBF .5pt;
  mso-border-top-themecolor:background1;mso-border-top-themeshade:191;
  mso-border-left-alt:solid #BFBFBF .5pt;mso-border-left-themecolor:background1;
  mso-border-left-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>700</span></p>
  </td>
  <td width=138 valign=top style='width:103.7pt;border-top:none;border-left:
  none;border-bottom:solid #BFBFBF 1.0pt;mso-border-bottom-themecolor:background1;
  mso-border-bottom-themeshade:191;border-right:solid #BFBFBF 1.0pt;mso-border-right-themecolor:
  background1;mso-border-right-themeshade:191;mso-border-top-alt:solid #BFBFBF .5pt;
  mso-border-top-themecolor:background1;mso-border-top-themeshade:191;
  mso-border-left-alt:solid #BFBFBF .5pt;mso-border-left-themecolor:background1;
  mso-border-left-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>0.9843</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:2'>
  <td width=138 valign=top style='width:103.7pt;border:solid #BFBFBF 1.0pt;
  mso-border-themecolor:background1;mso-border-themeshade:191;border-top:none;
  mso-border-top-alt:solid #BFBFBF .5pt;mso-border-top-themecolor:background1;
  mso-border-top-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;background:#F2F2F2;mso-background-themecolor:
  background1;mso-background-themeshade:242;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal style='mso-yfti-cnfc:68'><span lang=EN-US
  style='mso-bidi-font-weight:bold'>300<o:p></o:p></span></p>
  </td>
  <td width=138 valign=top style='width:103.7pt;border-top:none;border-left:
  none;border-bottom:solid #BFBFBF 1.0pt;mso-border-bottom-themecolor:background1;
  mso-border-bottom-themeshade:191;border-right:solid #BFBFBF 1.0pt;mso-border-right-themecolor:
  background1;mso-border-right-themeshade:191;mso-border-top-alt:solid #BFBFBF .5pt;
  mso-border-top-themecolor:background1;mso-border-top-themeshade:191;
  mso-border-left-alt:solid #BFBFBF .5pt;mso-border-left-themecolor:background1;
  mso-border-left-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;background:#F2F2F2;mso-background-themecolor:
  background1;mso-background-themeshade:242;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal style='mso-yfti-cnfc:64'><span lang=EN-US>0.9828</span></p>
  </td>
  <td width=138 valign=top style='width:103.7pt;border-top:none;border-left:
  none;border-bottom:solid #BFBFBF 1.0pt;mso-border-bottom-themecolor:background1;
  mso-border-bottom-themeshade:191;border-right:solid #BFBFBF 1.0pt;mso-border-right-themecolor:
  background1;mso-border-right-themeshade:191;mso-border-top-alt:solid #BFBFBF .5pt;
  mso-border-top-themecolor:background1;mso-border-top-themeshade:191;
  mso-border-left-alt:solid #BFBFBF .5pt;mso-border-left-themecolor:background1;
  mso-border-left-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;background:#F2F2F2;mso-background-themecolor:
  background1;mso-background-themeshade:242;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal style='mso-yfti-cnfc:64'><span lang=EN-US>800</span></p>
  </td>
  <td width=138 valign=top style='width:103.7pt;border-top:none;border-left:
  none;border-bottom:solid #BFBFBF 1.0pt;mso-border-bottom-themecolor:background1;
  mso-border-bottom-themeshade:191;border-right:solid #BFBFBF 1.0pt;mso-border-right-themecolor:
  background1;mso-border-right-themeshade:191;mso-border-top-alt:solid #BFBFBF .5pt;
  mso-border-top-themecolor:background1;mso-border-top-themeshade:191;
  mso-border-left-alt:solid #BFBFBF .5pt;mso-border-left-themecolor:background1;
  mso-border-left-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;background:#F2F2F2;mso-background-themecolor:
  background1;mso-background-themeshade:242;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal style='mso-yfti-cnfc:64'><span lang=EN-US>0.9842</span></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:3'>
  <td width=138 valign=top style='width:103.7pt;border:solid #BFBFBF 1.0pt;
  mso-border-themecolor:background1;mso-border-themeshade:191;border-top:none;
  mso-border-top-alt:solid #BFBFBF .5pt;mso-border-top-themecolor:background1;
  mso-border-top-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal style='mso-yfti-cnfc:4'><span lang=EN-US style='mso-bidi-font-weight:
  bold'>400<o:p></o:p></span></p>
  </td>
  <td width=138 valign=top style='width:103.7pt;border-top:none;border-left:
  none;border-bottom:solid #BFBFBF 1.0pt;mso-border-bottom-themecolor:background1;
  mso-border-bottom-themeshade:191;border-right:solid #BFBFBF 1.0pt;mso-border-right-themecolor:
  background1;mso-border-right-themeshade:191;mso-border-top-alt:solid #BFBFBF .5pt;
  mso-border-top-themecolor:background1;mso-border-top-themeshade:191;
  mso-border-left-alt:solid #BFBFBF .5pt;mso-border-left-themecolor:background1;
  mso-border-left-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><span lang=EN-US>0.9827</span></p>
  </td>
  <td width=138 valign=top style='width:103.7pt;border-top:none;border-left:
  none;border-bottom:solid #BFBFBF 1.0pt;mso-border-bottom-themecolor:background1;
  mso-border-bottom-themeshade:191;border-right:solid #BFBFBF 1.0pt;mso-border-right-themecolor:
  background1;mso-border-right-themeshade:191;mso-border-top-alt:solid #BFBFBF .5pt;
  mso-border-top-themecolor:background1;mso-border-top-themeshade:191;
  mso-border-left-alt:solid #BFBFBF .5pt;mso-border-left-themecolor:background1;
  mso-border-left-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><b style='mso-bidi-font-weight:normal'><span lang=EN-US>900<o:p></o:p></span></b></p>
  </td>
  <td width=138 valign=top style='width:103.7pt;border-top:none;border-left:
  none;border-bottom:solid #BFBFBF 1.0pt;mso-border-bottom-themecolor:background1;
  mso-border-bottom-themeshade:191;border-right:solid #BFBFBF 1.0pt;mso-border-right-themecolor:
  background1;mso-border-right-themeshade:191;mso-border-top-alt:solid #BFBFBF .5pt;
  mso-border-top-themecolor:background1;mso-border-top-themeshade:191;
  mso-border-left-alt:solid #BFBFBF .5pt;mso-border-left-themecolor:background1;
  mso-border-left-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal><b style='mso-bidi-font-weight:normal'><span lang=EN-US>0.9858<o:p></o:p></span></b></p>
  </td>
 </tr>
 <tr style='mso-yfti-irow:4;mso-yfti-lastrow:yes'>
  <td width=138 valign=top style='width:103.7pt;border:solid #BFBFBF 1.0pt;
  mso-border-themecolor:background1;mso-border-themeshade:191;border-top:none;
  mso-border-top-alt:solid #BFBFBF .5pt;mso-border-top-themecolor:background1;
  mso-border-top-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;background:#F2F2F2;mso-background-themecolor:
  background1;mso-background-themeshade:242;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal style='mso-yfti-cnfc:68'><span lang=EN-US
  style='mso-bidi-font-weight:bold'>500<o:p></o:p></span></p>
  </td>
  <td width=138 valign=top style='width:103.7pt;border-top:none;border-left:
  none;border-bottom:solid #BFBFBF 1.0pt;mso-border-bottom-themecolor:background1;
  mso-border-bottom-themeshade:191;border-right:solid #BFBFBF 1.0pt;mso-border-right-themecolor:
  background1;mso-border-right-themeshade:191;mso-border-top-alt:solid #BFBFBF .5pt;
  mso-border-top-themecolor:background1;mso-border-top-themeshade:191;
  mso-border-left-alt:solid #BFBFBF .5pt;mso-border-left-themecolor:background1;
  mso-border-left-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;background:#F2F2F2;mso-background-themecolor:
  background1;mso-background-themeshade:242;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal style='mso-yfti-cnfc:64'><span lang=EN-US>0.9829</span></p>
  </td>
  <td width=138 valign=top style='width:103.7pt;border-top:none;border-left:
  none;border-bottom:solid #BFBFBF 1.0pt;mso-border-bottom-themecolor:background1;
  mso-border-bottom-themeshade:191;border-right:solid #BFBFBF 1.0pt;mso-border-right-themecolor:
  background1;mso-border-right-themeshade:191;mso-border-top-alt:solid #BFBFBF .5pt;
  mso-border-top-themecolor:background1;mso-border-top-themeshade:191;
  mso-border-left-alt:solid #BFBFBF .5pt;mso-border-left-themecolor:background1;
  mso-border-left-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;background:#F2F2F2;mso-background-themecolor:
  background1;mso-background-themeshade:242;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal style='mso-yfti-cnfc:64'><span lang=EN-US>1000</span></p>
  </td>
  <td width=138 valign=top style='width:103.7pt;border-top:none;border-left:
  none;border-bottom:solid #BFBFBF 1.0pt;mso-border-bottom-themecolor:background1;
  mso-border-bottom-themeshade:191;border-right:solid #BFBFBF 1.0pt;mso-border-right-themecolor:
  background1;mso-border-right-themeshade:191;mso-border-top-alt:solid #BFBFBF .5pt;
  mso-border-top-themecolor:background1;mso-border-top-themeshade:191;
  mso-border-left-alt:solid #BFBFBF .5pt;mso-border-left-themecolor:background1;
  mso-border-left-themeshade:191;mso-border-alt:solid #BFBFBF .5pt;mso-border-themecolor:
  background1;mso-border-themeshade:191;background:#F2F2F2;mso-background-themecolor:
  background1;mso-background-themeshade:242;padding:0cm 5.4pt 0cm 5.4pt'>
  <p class=MsoNormal style='mso-yfti-cnfc:64'><span lang=EN-US>0.9854</span></p>
  </td>
 </tr>
</table>


## Get Started

### Train a MLP
```
python main.py train --hidden_nodes 900 --lr 0.01 --lambda_w 0.01 --vis_train True --vis_feature True
```
The visualization of training process and hidden features can be found in `./results/model_h-nodes${hidden_nodes}_lr${lr}_w${lambda_w}` after training.  

The option `--vis_train` is a choice for visualizing the training process. The option `--vis_feature` is a choice for visualizing the hidden-layer neurons, which needs to utilize the PIL and sklearn package. I found the sklearn has some version problem when running on VScode, so if you want to watch some visualization results, please use PyCharm or other IDE, directly running in Win or Linux cmd is also acceptable. Or you may choose to shut down this option.

### Inference
```
python main.py inference --hidden_nodes 900 --lr 0.01 --lambda_w 0.01
```

### Grid Search
```
python main.py search
```
The searching process including all of the hyper-parameter combinations from:  
>`--hidden_nodes`: {100, 200, 300, 400, 500, 600, 700, 800, 900}  
>`--lr`: {0.01, 0.003, 0.001, 0.0003, 0.0001}  
>`--lambda_w`: {0.1, 0.03, 0.01, 0.003, 0}  

The searching process may take a long period, a full result list can be found in `./logs_full_search.txt`.


## Contact

19307140032@fudan.edu.cn

