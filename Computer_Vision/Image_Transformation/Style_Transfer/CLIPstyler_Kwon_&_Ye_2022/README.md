# CLIPstyler: Image Style Transfer with a Single Text Condition
[[Paper]](https://arxiv.org/abs/2112.00374) [[Official Implementation]](https://github.com/cyclomon/CLIPstyler)


![All outputs](outputs/output_all.png)


# Usage

Default content images and text conditions located in `data/content_images/` and `data/text_conditions.txt`:
```
$ python src/CLIPstyler.py
```

Custom folder with content images and custom text conditions file:
```
$ python src/CLIPstyler.py --content path/to/folder --text path/to/file
```

Single content image and single text condition:
```
$ python src/CLIPstyler.py --content path/to/image --text 'custom text condition'
```

Set of images specified by a folder path and a single text condition string:
```
$ python src/CLIPstyler.py --content path/to/folder --text 'custom text condition'
```

Single content image on a set of text conditions specified by a text file:
```
$ python src/CLIPstyler.py --content path/to/image --text path/to/file
```

The stylized images will be saved to `outputs`
