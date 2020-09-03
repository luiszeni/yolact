
# **Y**ou **O**nly **L**ook **A**t **C**oefficien**T**s
```
    ██╗   ██╗ ██████╗ ██╗      █████╗  ██████╗████████╗ +   \
    ╚██╗ ██╔╝██╔═══██╗██║     ██╔══██╗██╔════╝╚══██╔══╝  \ O \
     ╚████╔╝ ██║   ██║██║     ███████║██║        ██║      \ N \
      ╚██╔╝  ██║   ██║██║     ██╔══██║██║        ██║       \ N \
       ██║   ╚██████╔╝███████╗██║  ██║╚██████╗   ██║        \ X \  
       ╚═╝    ╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝   ╚═╝         \
```

In this repo, I organized the code to convert the YOLOACT.pth model to onnx. The repo just organizes the code in a better way. The original onnx conversion comes from this repo: https://github.com/Ma-Dan/yolact/tree/onnx and this thread: https://github.com/dbolya/yolact/issues/74  (thanks for the people involved :])

Here I will focus on the conversion of the model of the following paper:
 - [YOLACT: Real-time Instance Segmentation](https://arxiv.org/abs/1904.02689)

I did no experiments with the YOLACT++, and I am omitting it in this readme.

Please refer to the original repo  (https://github.com/dbolya/yolact) for more information regarding training and evaluating in the COCO dataset.


# Installation

I am assuming that you will use (and know how to use) docker to run this code. If you do not use docker yet, please do a favor to yourself and start using it (a good starting point: https://towardsdatascience.com/how-docker-can-help-you-become-a-more-effective-data-scientist-7fc048ef91d5)! I am also assuming that you are using gnu-linux in the host machine. 

 - Clone this repository and enter in it:
   ```Shell
   git clone https://github.com/luiszeni/yolact_onnx.git
   cd yolact_onnx
   ```

 - Download the pretrained weights and put in the right folder:
 ```Shell
 wget http://inf.ufrgs.br/~lfazeni/yolact/yolact_resnet50_54_800000.pth
 mkdir weights && mv yolact_resnet50_54_800000.pth weights
 ``` 

 - Download some meme images to test (I hope that the links are not broke in future):
 ```Shell
 wget https://pbs.twimg.com/media/EJTaqTHVUAAboJg.jpg -O meme01.jpg
 wget https://i.pinimg.com/236x/8c/cb/59/8ccb5905351695a77e4d2a723d098761.jpg -O meme02.jpg
 wget https://img.devrant.com/devrant/rant/r_1397964_58Lc4.jpg -O meme03.jpg
 ``` 

 - Build the docker machine
 ```Shell
 docker build -f docker/Dockerfile -t yolact .
 ``` 

 - Create a container using the image.  I prefer to mount an external volume with the code in a folder in the host machine. It makes it easier to edit the code using a GUI-text-editor or ide. This command will drop you in the container shell.

 ```Shell
 docker run --gpus all -v  $(pwd):/root/yolact_onnx --shm-size 12G -ti \
 --name yolact yolact
 ```
  
- If you exit the container at any moment of the future, you can enter in the container again with this command.
 ```Shell
 docker start -ai yolact 
 ```
  
- **Observation:** It is possible to display OS windows from the container using X11 forwarding from the container to the host server X. It is also possible to forward it through an SSH connection. As configuring it is a little bit complex and specific to each environment, I will omit it. There are lots of tutorials on the internet, teaching X11 Foward in Docker. Anyway, we do not need the forwarding to view the outputs as we can ask the scripts to save the result as a file image.

- **Observation2:** All next commands are assuming that you are inside the docker container.



## Converting the model.pth to model.onnx

TODO bro

 ```Shell
 python3 convert_to_onnx.py 
 ``` 



## Testing in single images using the model.pth
```Shell
# Display qualitative results on the specified image.
python3 eval.py --trained_model=weights/yolact_resnet50_54_800000.pth --score_threshold=0.15 --top_k=15 --image=meme01.jpg 

# Process an image and save it to another file.
python3 eval.py --trained_model=weights/yolact_resnet50_54_800000.pth --score_threshold=0.15 --top_k=15 --image=meme01.jpg:meme01_out.jpg
```

## Testing in single images using the model.onnx
```Shell

# Display qualitative results on the specified image.
python3 eval.py --trained_model=weights/yolact_resnet50_54_800000.onnx --score_threshold=0.15 --top_k=15 --image=meme01.jpg 

# Process an image and save it to another file.
python3 eval.py --trained_model=weights/yolact_resnet50_54_800000.onnx --score_threshold=0.15 --top_k=15 --image=meme01.jpg:meme01_out.jpg
```



# Citation
If you use YOLACT or this code base in your work, please cite
```
@inproceedings{yolact-iccv2019,
  author    = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  title     = {YOLACT: {Real-time} Instance Segmentation},
  booktitle = {ICCV},
  year      = {2019},
}
```

For YOLACT++, please cite
```
@misc{yolact-plus-arxiv2019,
  title         = {YOLACT++: Better Real-time Instance Segmentation},
  author        = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  year          = {2019},
  eprint        = {1912.06218},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV}
}
```
