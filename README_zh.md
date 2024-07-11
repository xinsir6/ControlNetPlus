# ControlNetPlus
## ControlNet++: 适用于图像生成和编辑的一体化ControlNet！
![images_display](./images/masonry.webp)

***我们设计了一种新架构，可在条件文本到图像生成中支持10多种控制类型，并能生成与midjourney视觉上相媲美的高分辨率图像***。该网络基于原始ControlNet架构，我们提出了两个新模块：1. 扩展原始ControlNet以使用相同的网络参数支持不同的图像条件。2. 在不增加计算负担的情况下支持多个条件输入，这对于希望详细编辑图像的设计师尤其重要。不同条件使用相同的条件编码器，无需增加额外的计算或参数。我们在SDXL上进行了彻底的实验，在控制能力和美学评分方面均表现出色。我们向开源社区发布了方法和模型，让每个人都能享受它。

**如果你觉得这有用，请给我一个星标，非常感谢！！**

**超过500星，发布带有平铺和修复功能的ProMax版本！！**
**超过1000星，发布适用于SD3的ControlNet++模型！！**

## 模型的优势
- 使用类似novelai的桶训练，可以生成任何宽高比的高分辨率图像
- 使用大量高质量数据（超过10000000张图像），数据集涵盖了多种情况
- 使用类似DALLE.3的重新描述提示，利用CogVLM生成详细描述，具有良好的提示跟随能力
- 训练期间使用了许多有用的技巧，包括但不限于数据增强、多种损失、多分辨率
- 与原始ControlNet相比，使用几乎相同的参数，网络参数或计算没有明显增加
- 支持10多种控制条件，在任何单一条件下的性能与独立训练相比没有明显下降
- 支持多条件生成，条件融合在训练期间学习，无需设置超参数或设计提示
- 与其它开源SDXL模型兼容，如BluePencilXL、CounterfeitXL，与其它Lora模型兼容

## 我们发布的其他热门模型
https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0  
https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0  
https://huggingface.co/xinsir/controlnet-tile-sdxl-1.0  
https://huggingface.co/xinsir/controlnet-canny-sdxl-1.0


## 新闻
- [07/06/2024] 发布`ControlNet++`及预训练模型。
- [07/06/2024] 发布推理代码（单条件 & 多条件）。

## 待办事项:
- [ ] 为gradio发布ControlNet++
- [ ] 为Comfyui发布ControlNet++
- [ ] 发布训练代码和训练指导。
- [ ] 发布arxiv论文。

## 视觉示例
### Openpose
这是最重要的ControlNet模型之一，我们在训练此模型时使用了许多技巧，与 https://huggingface.co/xinsir/controlnet-openpose-sdxl-1.0 的性能相当，是姿态控制的最新技术。
为了使Openpose模型达到最佳性能，你应该替换controlnet_aux包中的draw_pose函数（Comfyui有其自己的controlnet_aux包），详细信息请参阅**推理脚本**。
![pose0](./images/000000_pose_concat.webp)
![pose1](./images/000001_pose_concat.webp)
![pose2](./images/000002_pose_concat.webp)
![pose3](./images/000003_pose_concat.webp)
![pose4](./images/000004_pose_concat.webp)
### Depth
![depth0](./images/000005_depth_concat.webp)
![depth1](./images/000006_depth_concat.webp)
![depth2](./images/000007_depth_concat.webp)
![depth3](./images/000008_depth_concat.webp)
![depth4](./images/000009_depth_concat.webp)
### Canny
这是最重要的ControlNet模型之一，canny模型与lineart、anime lineart、mlsd混合训练。在处理任何细线时具有稳健的性能，该模型是降低畸形率的关键，建议使用细线重新绘制手/脚。
![canny0](./images/000010_canny_concat.webp)
![canny1](./images/000011_canny_concat.webp)
![canny2](./images/000012_canny_concat.webp)
![canny3](./images/000013_canny_concat.webp)
![canny4](./images/000014_canny_concat.webp)
### Lineart
![lineart0](./images/000015_lineart_concat.webp)
![lineart1](./images/000016_lineart_concat.webp)
![lineart2](./images/000017_lineart_concat.webp)
![lineart3](./images/000018_lineart_concat.webp)
![lineart4](./images/000019_lineart_concat.webp)
### AnimeLineart
![animelineart0](./images/000020_anime_lineart_concat.webp)
![animelineart1](./images/000021_anime_lineart_concat.webp)
![animelineart2](./images/000022_anime_lineart_concat.webp)
![animelineart3](./images/000023_anime_lineart_concat.webp)
![animelineart4](./images/000024_anime_lineart_concat.webp)
### Mlsd
![mlsd0](./images/000025_mlsd_concat.webp)
![mlsd1](./images/000026_mlsd_concat.webp)
![mlsd2](./images/000027_mlsd_concat.webp)
![mlsd3](./images/000028_mlsd_concat.webp)
![mlsd4](./images/000029_mlsd_concat.webp)
### Scribble
这是最重要的ControlNet模型之一，涂鸦模型可以支持任何线宽和任何线型。其性能与 https://huggingface.co/xinsir/controlnet-scribble-sdxl-1.0 相当，让每个人都能成为灵魂画师。
![scribble0](./images/000030_scribble_concat.webp)
![scribble1](./images/000031_scribble_concat.webp)
![scribble2](./images/000032_scribble_concat.webp)
![scribble3](./images/000033_scribble_concat.webp)
![scribble4](./images/000034_scribble_concat.webp)
### Hed
![hed0](./images/000035_hed_concat.webp)
![hed1](./images/000036_hed_concat.webp)
![hed2](./images/000037_hed_concat.webp)
![hed3](./images/000038_hed_concat.webp)
![hed4](./images/000039_hed_concat.webp)
### Pidi(Softedge)
![pidi0](./images/000040_softedge_concat.webp)
![pidi1](./images/000041_softedge_concat.webp)
![pidi2](./images/000042_softedge_concat.webp)
![pidi3](./images/000043_softedge_concat.webp)
![pidi4](./images/000044_softedge_concat.webp)
### Teed(512 detect, higher resolution, thiner line)
![ted0](./images/000045_ted_concat.webp)
![ted1](./images/000046_ted_concat.webp)
![ted2](./images/000047_ted_concat.webp)
![ted3](./images/000048_ted_concat.webp)
![ted4](./images/000049_ted_concat.webp)
### Segment
![segment0](./images/000050_seg_concat.webp)
![segment1](./images/000051_seg_concat.webp)
![segment2](./images/000052_seg_concat.webp)
![segment3](./images/000053_seg_concat.webp)
![segment4](./images/000054_seg_concat.webp)
### Normal
![normal0](./images/000055_normal_concat.webp)
![normal1](./images/000056_normal_concat.webp)
![normal2](./images/000057_normal_concat.webp)
![normal3](./images/000058_normal_concat.webp)
![normal4](./images/000059_normal_concat.webp)

## 多控制视觉示例
### Openpose + Canny
注意：使用姿态骨架来控制人体姿势，使用细线绘制手部/脚部细节以避免畸形。
![pose_canny0](./images/000007_openpose_canny_concat.webp)
![pose_canny1](./images/000008_openpose_canny_concat.webp)
![pose_canny2](./images/000009_openpose_canny_concat.webp)
![pose_canny3](./images/000010_openpose_canny_concat.webp)
![pose_canny4](./images/000011_openpose_canny_concat.webp)
![pose_canny5](./images/000012_openpose_canny_concat.webp)

### Openpose + Depth
注意：深度图包含细节信息，建议将深度用于背景，将姿态骨架用于前景。
![pose_depth0](./images/000013_openpose_depth_concat.webp)
![pose_depth1](./images/000014_openpose_depth_concat.webp)
![pose_depth2](./images/000015_openpose_depth_concat.webp)
![pose_depth3](./images/000016_openpose_depth_concat.webp)
![pose_depth4](./images/000017_openpose_depth_concat.webp)
![pose_depth5](./images/000018_openpose_depth_concat.webp)

### Openpose + Scribble
  注意：涂鸦是一种强大的线条模型，如果你想画一些不需要严格轮廓的东西，你可以使用它。Openpose + 涂鸦让你在生成初始图像时有更多自由，然后你可以使用细线来编辑细节。
![pose_scribble0](./images/000001_openpose_scribble_concat.webp)
![pose_scribble1](./images/000002_openpose_scribble_concat.webp)
![pose_scribble2](./images/000003_openpose_scribble_concat.webp)
![pose_scribble3](./images/000004_openpose_scribble_concat.webp)
![pose_scribble4](./images/000005_openpose_scribble_concat.webp)
![pose_scribble5](./images/000006_openpose_scribble_concat.webp)

### Openpose + Normal
![pose_normal0](./images/000019_openpose_normal_concat.webp)
![pose_normal1](./images/000020_openpose_normal_concat.webp)
![pose_normal2](./images/000021_openpose_normal_concat.webp)
![pose_normal3](./images/000022_openpose_normal_concat.webp)
![pose_normal4](./images/000023_openpose_normal_concat.webp)
![pose_normal5](./images/000024_openpose_normal_concat.webp)

### Openpose + Segment
![pose_segment0](./images/000025_openpose_sam_concat.webp)
![pose_segment1](./images/000026_openpose_sam_concat.webp)
![pose_segment2](./images/000027_openpose_sam_concat.webp)
![pose_segment3](./images/000028_openpose_sam_concat.webp)
![pose_segment4](./images/000029_openpose_sam_concat.webp)
![pose_segment5](./images/000030_openpose_sam_concat.webp)
  
## 数据集
我们收集了大量高质量的图像。图像经过严格筛选和注释，涵盖的题材广泛，包括摄影、动漫、自然、midjourney等。

## 网络架构
![images](./images/ControlNet++.png)

在ControlNet++中，我们提出了两个新模块，分别命名为Condition Transformer和Control Encoder。我们对一个旧模块进行了微调，以增强其表示能力。此外，我们提出了一种统一的训练策略，以实现单个和多个控制在一个阶段的实现。
### Control Encoder
对于每个条件，我们为其分配一个控制类型id，例如，openpose--(1, 0, 0, 0, 0, 0)，depth--(0, 1, 0, 0, 0, 0)，多个条件将像(openpose, depth) --(1, 1, 0, 0, 0, 0)。在Control Encoder中，控制类型id将转换为控制类型嵌入（使用正弦位置嵌入），然后我们使用一个线性层将控制类型嵌入投射到与时间嵌入相同的维度。控制类型特征被添加到时间嵌入中，以指示不同的控制类型，此简单设置可以帮助ControlNet区分不同的控制类型，因为时间嵌入倾向于对整个网络产生全局影响。无论是单条件还是多条件，都有一个与其对应的唯一控制类型id。  
### Condition Transformer
我们扩展了ControlNet，使用相同的网络同时支持多个控制输入。Condition Transformer用于组合不同的图像条件特征。我们的方法有两大创新，首先，不同的条件共享相同的条件编码器，这使得网络更简单和轻量级。这与T2I或UniControlNet等主流方法不同。其次，我们添加了一个transformer层来交换原始图像和条件图像的信息，而不是直接使用transformer的输出，我们使用它来预测原始条件特征的条件偏差。这有点像ResNet，我们实验性地发现这种设置可以明显提高网络的性能。  
### 修改后的Condition Encoder
ControlNet的原始条件编码器是一系列卷积层和Silu激活的堆叠。我们没有改变编码器架构，我们只是增加了卷积通道以获得一个“胖”编码器。这可以明显提高网络的性能。原因是，我们为所有图像条件共享相同的编码器，因此它要求编码器具有更高的表示能力。原始设置对于单条件可能很好，但对于10+条件则不那么好。请注意，使用原始设置也很好，只是在图像生成质量上会有所牺牲。
### 统一的训练策略
使用单个条件训练可能会受到数据多样性的限制。例如，openpose要求你使用有人的图像进行训练，mlsd要求你使用有线条的图像进行训练，因此可能会影响生成未见对象时的性能。此外，训练不同条件的难度不同，同时使所有条件收敛并达到每个单条件的最佳性能是棘手的。最后，我们将倾向于同时使用两个或多个条件，多条件训练将使不同条件的融合更加顺畅，增加网络的鲁棒性（因为单条件学习的知识有限）。我们提出了一种统一的训练阶段，以同时实现单条件优化收敛和多条件融合。

## 控制模式
ControlNet++需要向网络传递一个控制类型id。我们将10多种控制合并为6种控制类型，每种类型的含义如下：
0 -- openpose  
1 -- depth  
2 -- thick line(scribble/hed/softedge/ted-512)  
3 -- thin line(canny/mlsd/lineart/animelineart/ted-1280)  
4 -- normal  
5 -- segment  


## 安装
我们建议使用python版本 >= 3.8，你可以使用以下命令设置虚拟环境：

```shell
conda create -n controlplus python=3.8
conda activate controlplus
pip install -r requirements.txt
```

## 下载权重
你可以在 /xinsir/controlnet-union-sdxl-1.0 下载模型权重。任何新模型的发布都会放在huggingface上，你可以关注 /xinsir 以获取最新的模型信息。

## 推理脚本
我们为每个控制条件提供了一个推理脚本。请参考它获取更多细节。

存在一些预处理差异，为了获得最佳的openpose-control性能，请执行以下操作：
在controlnet_aux包中找到util.py，将draw_bodypose函数替换为以下代码
```python
def draw_bodypose(canvas: np.ndarray, keypoints: List[Keypoint]) -> np.ndarray:
    """
    在给定的画布上绘制表示身体姿势的关键点和肢体。

    参数:
        canvas (np.ndarray): 一个3D numpy数组，表示要绘制身体姿势的画布（图像）。
        keypoints (List[Keypoint]): 一个Keypoint对象列表，表示要绘制的身体关键点。

    返回:
        np.ndarray: 一个3D numpy数组，表示带有绘制身体姿势的修改后的画布。

    注意:
        该函数期望关键点的x和y坐标在0和1之间进行归一化。
    """
    H, W, C = canvas.shape

    
    if max(W, H) < 500:
        ratio = 1.0
    elif max(W, H) >= 500 and max(W, H) < 1000:
        ratio = 2.0
    elif max(W, H) >= 1000 and max(W, H) < 2000:
        ratio = 3.0
    elif max(W, H) >= 2000 and max(W, H) < 3000:
        ratio = 4.0
    elif max(W, H) >= 3000 and max(W, H) < 4000:
        ratio = 5.0
    elif max(W, H) >= 4000 and max(W, H) < 5000:
        ratio = 6.0
    else:
        ratio = 7.0

    stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue

        Y = np.array([keypoint1.x, keypoint2.x]) * float(W)
        X = np.array([keypoint1.y, keypoint2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), int(stickwidth * ratio)), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), int(4 * ratio), color, thickness=-1)

    return canvas
```
对于单条件推理，你应该给出一个提示和一个控制图像，更改python文件中相应的行。
```shell
python controlnet_union_test_openpose.py
```
对于多条件推理，你应该确保你的输入image_list与你的control_type兼容，例如，如果你想使用openpose和深度控制，image_list --> [controlnet_img_pose, controlnet_img_depth, 0, 0, 0, 0]，control_type --> [1, 1, 0, 0, 0, 0]。请参考controlnet_union_test_multi_control.py获取更多细节。
理论上，你不需要为不同条件设置条件尺度，网络设计和训练以自然融合不同条件。每个条件输入的默认设置是1.0，这与多条件训练相同。
然而，如果你想增加某个特定输入条件的影响，你可以在Condition Transformer模块中调整条件尺度。在该模块中，输入条件将与源图像特征一起加上偏差预测进行相加。
将其乘以特定的尺度将产生很大影响（但可能会导致一些未知的结果）。

```shell
python controlnet_union_test_multi_control.py
```
