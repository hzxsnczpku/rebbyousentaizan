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

### Chicken Leg
<div align='center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_raw.JPG" height="200px">
</div>

<div align = 'center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_1_sty.jpg" width='160px' height = '200px'></a>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_1.jpg" height = '200px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_2.jpg"  height = '200px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_2_sty.jpg"  width='160px' height = '200px'></a>
<br>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_3_sty.jpg" width='160px' height = '200px'></a>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_3.jpg" height = '200px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_4.jpg"  height = '200px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_4_sty.jpeg"  width='160px' height = '200px'></a>
<br>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_5_sty.jpg" width='160px' height = '200px'></a>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_5.jpg" height = '200px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_6.jpg"  height = '200px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/jt_6_sty.jpg" width='160px' height = '200px'></a>
<br>
</div>

### Hebe
<div align = 'center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/hebe.jpg" width='225px' height = '300px'></a>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/hebe_sty.jpeg" width='225px' height = '300px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/hebe_after.jpg" width='225px' height = '300px'>
<br>
</div>

### Ni Mengjiao Horse Rider
<div align = 'center'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/horse.jpeg"width='225px' height = '300px'></a>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/horse_sty.jpg" width='225px' height = '300px'>
<img src = "https://raw.githubusercontent.com/hzxsnczpku/rebbyousentaizan/master/examples/horse_after.jpg"  width='225px' height = '300px'>
<br>
</div>
