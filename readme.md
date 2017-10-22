# Style Transfer & Fast Style Transfer

This is a tensorflow implementation of two classical style transfer algorithms:
* [A Neural Algorithm of Artistic Style](https://arxiv.org/pdf/1508.06576v2.pdf)
* [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/)

## Neural Style Transfer
First run the following code to download the VGG Net,
```
bash set_up_style_transfer.sh
```

To apply style transfering, run the following code,
```
python run_main.py --content <content file> --style <style file> --output <output file>
```

## Fast Style Transfer
First run the following code to download the VGG Net and training dataset,
```
bash set_up_style_transfer.sh
```
Then run the following code to train your model,
```
python style.py --style <style file> --checkpoint-dir <checkpoint path>

```
Finally, use the following code to create your own transfered pictures,
```
python evaluate.py --checkpoint <ckpt file> --in-path <content file> --out-path <output file>
```

## Results
Some results are placed below:

<div align='center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_raw.JPG" height="200px">
</div>

<div align = 'center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_1_sty.jpg" width='180px' height = '200px'></a>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_1.jpg" height = '200px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_2.jpg"  height = '200px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_2_sty.jpg"  height = '200px'></a>
<br>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_3_sty.jpg" width='180px' height = '200px'></a>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_3.jpg" height = '200px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_4.jpg"  height = '200px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_4_sty.jpg"  height = '200px'></a>
<br>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_5_sty.jpg" width='180px' height = '200px'></a>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_5.jpg" height = '200px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_6.jpg"  height = '200px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_6_sty.jpg"  height = '200px'></a>
