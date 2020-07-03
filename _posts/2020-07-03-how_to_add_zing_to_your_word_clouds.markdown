---
layout: post
title:      "How to Add Zing to Your Word Clouds"
date:       2020-07-03 20:47:36 +0000
permalink:  how_to_add_zing_to_your_word_clouds
---


Word Clouds are a really easy way to summarize text and make the most popular words jump to your attention. The trouble is, Word Clouds have become really popular. How do you make your Word Cloud jump out from the cloud of Word Clouds? 

Here are some examples that I found helpful.

I’m using an image of an emoji that I found at the link below, and text about emojis from Wikipedia to form the cloud<br>

[emoji](https://www.publicdomainpictures.net/pictures/320000/velka/smiling-emoji-1581514868roK.png) <br>


Here are the libraries that I used:

```
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
```

**Basic**<br>
Let's start off with a basic WordCloud


```
# Generate a wordcloud
wordcloud = WordCloud(max_font_size=80, max_words=1000, background_color="black", colormap='Paired').generate(text2)

plt.figure(figsize=(10,6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```

![basic](https://imgur.com/XsaG3hF.png)


I made a couple of changes to the defaults, background_color, max_font_size, and colormap - any Matplotlib palette will work.

<br>
<br>
**Shape + Font**<br>
You can shape a word cloud by using a mask. You upload the image and convert it to an array. Wordcloud will draw words in positions where the image is not white. The value of 255 is white and the value of 1 is black. You may need to tweak an image if it has a lot of 0 values.

```
# some adjustments to the image, not always necessary 
emoj = Image.open("emoji.png")
mask2 = Image.new("RGB", emoj.size, (255,255,255))
mask2.paste(emoj,emoj)
mask2 = np.array(mask2)


emo = WordCloud(font_path = font_path, background_color="white", max_words=1500, mask=mask2, stopwords=stopwords,
               min_font_size=10, colormap='Dark2')

# Generate a wordcloud
emo.generate(text2)

# show
plt.figure(figsize=[7,7])
plt.imshow(emo, interpolation='bilinear')
plt.axis("off")
plt.show()
```


![shape & font](https://imgur.com/4nweWAE.png)

In this example I'm also using font_path to choose a custom font. You set this to where the font you want to use is located on your computer:
for example:
```
font_path = "/Users/username/AppData/Local/Microsoft/Windows/Fonts/Comfortaa-Bold.ttf"
```

WordCloud also has a built in list of stopwords that you can ignore, adjust, or use like this:

```
stopwords = set(STOPWORDS)
```
<br>
<br>

**Smiley**<br>
Here's a similar example, where I adjusted the numpy array so that the eyes and mouth of the emoji are left blank


```
mask2 = np.array(Image.open("emoji.png"))

mask3 = np.where(mask2==0, 255, mask2) 


emo = WordCloud(background_color="white", max_words=1000, mask=mask3, stopwords=stopwords, colormap='Dark2')

# Generate a wordcloud
emo.generate(text2)


# show
plt.figure(figsize=[5,5])
plt.imshow(emo, interpolation='bilinear')
plt.axis("off")
plt.show()
```





![smiley](https://imgur.com/vgmkPwo.png)

<br>
<br>

**Color**<br>
You can also use the ImageColorGenerator function to set the colormap to match the image. (The dark patches are the eyes and mouth peaking through. You may need to reduce max_font_size to improve the look of image details )


```
# note - need to set max font size, or font extends over the colors
mask = np.array(Image.open("emo_3.png"))
emo_clr = WordCloud(background_color="white", mode="RGBA", max_words=1000, max_font_size=80, mask=mask).generate(text2)

# create coloring from image
image_colors = ImageColorGenerator(mask)
plt.figure(figsize=[5,5])
plt.imshow(emo_clr.recolor(color_func=image_colors), interpolation="bilinear")
plt.axis("off")

plt.show()
```



![image color](https://imgur.com/qlIiBYw.png)

<br>
<br>

**Reverse**<br>
Finally, you can also experiment with a reverse image. This plots the WordCloud around a shadow, rather than inside a shape. The code is the same, I tweaked the image in MS paint before opening it.

![shadow](https://imgur.com/h3kJ3Ll.png)


Once you have a shadow, you can open it and drop the original image on top, like this.


![combined](https://imgur.com/YgeiSZB.png)

<br>
<br>

Here's a great tutorial with some more detail:
[DataCamp WordClouds](https://www.datacamp.com/community/tutorials/wordcloud-python)

Happy Clouding!
