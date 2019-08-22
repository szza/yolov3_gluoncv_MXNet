"""You Only Look Once Object Detection v3"""
# pylint: disable=arguments-differ
from __future__ import absolute_import
from __future__ import division

import os
import warnings
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet import autograd
from mxnet.gluon import nn
from mxnet.gluon.nn import BatchNorm
from .darknet import _conv2d, darknet53
from .yolo_target import YOLOV3TargetMerger
from ..loss import YOLOV3Loss

__all__ = ['YOLOV3',
           'get_yolov3',
           'yolo3_darknet53_voc',
           'yolo3_darknet53_coco',
           'yolo3_darknet53_custom',
           'yolo3_mobilenet1_0_coco',
           'yolo3_mobilenet1_0_voc',
           'yolo3_mobilenet1_0_custom',
           'yolo3_mobilenet0_25_coco',
           'yolo3_mobilenet0_25_voc',
           'yolo3_mobilenet0_25_custom'
           ]

def _upsample(x, stride=2):
    """Simple upsampling layer by stack pixel alongside horizontal and vertical directions.
    Parameters
    ----------
    x : mxnet.nd.NDArray or mxnet.symbol.Symbol
        The input array.
    stride : int, default is 2
        Upsampling stride
    """
    return x.repeat(axis=-1, repeats=stride).repeat(axis=-2, repeats=stride)

class YOLODetectionBlockV3(gluon.HybridBlock):
    """YOLO V3 Detection Block which does the following:
    - add a few conv layers
    - return the output
    - have a branch that do yolo detection.
    Parameters
    ----------
    channel : int
        Number of channels for 1x1 conv. 3x3 Conv will have 2*channel.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    def __init__(self, channel, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(YOLODetectionBlockV3, self).__init__(**kwargs)
        assert channel % 2 == 0, "channel {} cannot be divided by 2".format(channel)
        '''
            输入经过这个层，在route没有改变分辨率，而是在tip下采样了
        '''
        with self.name_scope():
            self.body = nn.HybridSequential(prefix='')
            for _ in range(2):
                # 1x1 reduce
                self.body.add(_conv2d(channel, 1, 0, 1,
                                      norm_layer=norm_layer, norm_kwargs=norm_kwargs))
                # 3x3 expand
                self.body.add(_conv2d(channel * 2, 3, 1, 1,
                                      norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.body.add(_conv2d(channel, 1, 0, 1,
                                  norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            self.tip = _conv2d(channel * 2, 3, 1, 1, norm_layer=norm_layer, norm_kwargs=norm_kwargs)  # 就是一个3*3的下采样卷积

    # pylint: disable=unused-argument
    def hybrid_forward(self, F, x):
        route = self.body(x)
        tip = self.tip(route)
        return route, tip

class YOLOOutputV3(gluon.HybridBlock):
    """YOLO output layer V3.
    Parameters
    ----------
    index : int
        Index of the yolo output layer, to avoid naming conflicts only.
    num_class : int
        Number of foreground objects.
    anchors : iterable
        The anchor setting. Reference: https://arxiv.org/pdf/1804.02767.pdf.
    stride : int
        Stride of feature map.
    alloc_size : tuple of int, default is (128, 128)
        For advanced users. Define `alloc_size` to generate large enough anchor
        maps, which will later saved in parameters. During inference, we support arbitrary
        input image by cropping corresponding area of the anchor map. This allow us
        to export to symbol so we can run it in c++, Scalar, etc.
    """
    # anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]] 的其中一个维度
    # stride = [8, 16, 32] # 其中一个值 
    
    def __init__(self, index, num_class, anchors, stride,
                 alloc_size=(128, 128), **kwargs):
        super(YOLOOutputV3, self).__init__(**kwargs)
        anchors = np.array(anchors).astype('float32')
        self._classes = num_class
        self._num_pred = 1 + 4 + num_class  # 1 objness + 4 box + num_class  一个box预测的内容
        self._num_anchors = anchors.size    # 3，比如 [10, 13, 16, 30, 33, 23]
        self._stride = stride               # 比如 8，即是从DarkNet-53处的stride = 8最后一层提取的信息

        '''
            下面主要是创建了几个变量：
                self.anchor  shape = [1,1,3,2]      宽和高的值
                self.offsets shape=[1,1,128,128,2]  偏移量
        '''
        with self.name_scope():
            all_pred = self._num_pred * self._num_anchors  # 一个box输出的tensor： 3 * (11 + 4+ 20) ,其中20是voc类别数目 ，因此：all_pred=75.
            self.prediction = nn.Conv2D(all_pred, kernel_size=1, padding=0, strides=1) # 用于测试输出的函数
            # anchors will be multiplied to predictions
            anchors = anchors.reshape(1, 1, -1, 2)  # [1,1,3,2]
            self.anchors = self.params.get_constant('anchor_%d'%(index), anchors)  # index：表示的是阶段

            # offsets will be added to predictions
            grid_x = np.arange(alloc_size[1])
            grid_y = np.arange(alloc_size[0])
            grid_x, grid_y = np.meshgrid(grid_x, grid_y) # 128 * 128
            # stack to (n, n, 2)  
            offsets = np.concatenate((grid_x[:, :, np.newaxis], grid_y[:, :, np.newaxis]), axis=-1)  # 128*128*2
            # expand dims to (1, 1, n, n, 2) so it's easier for broadcasting
            offsets = np.expand_dims(np.expand_dims(offsets, axis=0), axis=0) # [1,1,128,128,2]
            # 偏移量是相对于当前单元格的偏移量，经过sigmoid变换加上原始c_x,c_y,就得到中心
            self.offsets = self.params.get_constant('offset_%d'%(index), offsets) 

    def hybrid_forward(self, F, x, anchors, offsets):

        '''
        输出层的总体思路：  
            1 YOLODetectionBlockV3层的输出tip部分，经过需要经过opuput一个卷积操来得到训练/预测的信息
            2 将得到的向量pred，提取各个分量，
       
        # prediction flat to (batch, pred per pixel, height * width)
        self.prediction(x) 就是一个1*1的卷积, 仅仅是改变channels，而不改变Feature map大小，所以输出的经过reshape为：(batch, 75, h*w)
        '''
        pred = self.prediction(x).reshape((0, self._num_anchors * self._num_pred, -1))  # (batch,  75, h*w)
        '''
        pred:
            transpose to (batch, height * width, num_anchor, num_pred)
            pred--> (batch, h*w, channel) --> (batch, h*w, 3, 25),其中25= 4+1+20

        再取出各个分量：
            raw_box_centers:
                pred[:,:,:,0:2] --> shape = [batch,h*w,3,2]    原始中心结合偏移量就得了预测的中心
            raw_box_scales:
                pred[:,:,:,2:4] --> shape = [batch,h*w,3,2]    对应的宽和高
            objness:
                pred[:,:,:,4:5] --> shape = [batch,h*w,3,1]    有没有对象 非9即1
            class_pred:
                pred[:,:,:,5:]  --> shape = [batch,h*w,3,20]    20个类别预测（对于voc数据集）
        '''
        pred = pred.transpose(axes=(0, 2, 1)).reshape((0, -1, self._num_anchors, self._num_pred))
        raw_box_centers = pred.slice_axis(axis=-1, begin=0, end=2)  # tx, ty
        raw_box_scales = pred.slice_axis(axis=-1, begin=2, end=4)   # tw, th
        objness = pred.slice_axis(axis=-1, begin=4, end=5)          # obj 
        class_pred = pred.slice_axis(axis=-1, begin=5, end=None)    # 都是没有经过sigmoid处理的

        '''
        offsets 是针对整个feature map的offsets，因此需要将其变成和输出x同样的宽高，才是有效。 
            valid offsets, (1, 1, height, width, 2)
            这里的输入x，是YOLODetectionBlockV3层的输出tip.shape = [bathc, channels, h, w]
        
        offsets:
            offset[:,:,0:h,0:w,:] --> offsets.shape=[1,1,h,w,2]-->reshape: to (1, height*width, 1, 2)

            offsets:每个grid cell的偏移量，就是这个grid cell左上角坐标。因此，对于h*w的feature map，offsets就是(1,h,w,2)-->reshape 
        '''
        offsets = F.slice_like(offsets, x * 0, axes=(2, 3))  
        offsets = offsets.reshape((1, -1, 1, 2))  # offsets 应该是对整个feature map进行偏移 这个大小是原图大小/stride

        '''
        box_centers:
            offsets.shape     = [1,h*w,1,2]  
            raw.shape         = [batch,h*w,3,2]   
        --> box_centers.shape = [batch,h*w, 3, 2]   用于train/inference 返回
        
    以下部分都是用于inferecne
        box_scales:
            raw_box_scales.shape =  [batch,h*w,3,2]  
            anchor.shape         =  [1,1,3,2]  
        --> box_scales.shape     =  [batch,h*w, 3, 2] 
        confidence: 
            [batch,g*w,3,1] 
        class_score:
            class_pred.shape  = [batch, h*w, 3, 20]  
            confidence.shape  = [batch, h*w, 3, 1]  
        --> class_score.shape = [batch, h*w, 3, 20]
        bbox: 
            box.shape =  [batch, h*w, 3, 4]
        '''
       
        box_centers = F.broadcast_add(F.sigmoid(raw_box_centers), offsets) * self._stride   # bx =  sigmoid(tx) + cx
        box_scales = F.broadcast_mul(F.exp(raw_box_scales), anchors)                        # bw =  pw x exp{tw}
        confidence = F.sigmoid(objness)                                                     # sigmoid(objness) d
        class_score = F.broadcast_mul(F.sigmoid(class_pred), confidence)                    # score = sigmodi(class_pred) * sigmoid(objness)
        wh = box_scales / 2.0  
        bbox = F.concat(box_centers - wh, box_centers + wh, dim=-1)  # 得到的是最终的bbox，box_center - wh是（X，y)-wh ，

        if autograd.is_training():
            '''
                during training, we don't need to convert whole bunch of info to detection results
                因为在训练时，此时输出的东西要参与在整个框架部分的损失计算。而在预测时，只是需要输出一个bbox即可。

                这里的raw 表达的意思是指没有经过sigmoid转换
                bbox.shape    = [batch, h*W*3, 4] 
                anchor.shape  = [1,1,3,2]
                offsets.shape = [1,h*w,1,2]  

            返回的都是没有经过sigmoid处理的，因此在进行损失计算时，都是先经过simgoid函数处理才使用cross entroy loss。而raw_box_scales 使用的是L1-loss，不需要sigmoid处理
            而且raw_box_scales对应的ground-truth 也没有经过sigmoid 处理，前后一致。
                其中对应ground-truth:
                    scale_targets[b,  index, match, 0] = np.log(max(gtw, 1) / np_anchors[match, 0]) # tw
                    scale_targets[b,  index, match, 1] = np.log(max(gth, 1) / np_anchors[match, 1]) # th
            '''
            return (bbox.reshape((0, -1, 4)), raw_box_centers, raw_box_scales,
                    objness, class_pred, anchors, offsets)

        ''' 
        Inference:

        prediction per class
            输出的特征维度
                box.shape:          [batch, h*w, 3, 4]    --> [20, batch, h*w, 3, 4]   原本只是预测一个类，现在是20个
                scores.shape:       [batch, h*w, 3,20]    --> [20, batch, h*w, 3, 1]   相当于，原本是一个box预测20个类别，现在用20个box分别来预测
                ids.shape:          [20, 1, 1, 1, 1]      --> [20, batch, h*w, 3, 1]   为每个类建立一个对应的索引
                detections.shape:    concat:[20, batch, h*w, 3, 6] --> [batch, 20, h*w, 3, 6]--> [batch, 20*h*w*3, 6]

            说明：
            [batch, h*w, 3, 4]，
                在h*w大小的feature map上，一个grid cell产生3个相对应的boxes，取IoU最大的那个作为预测类别。
                一个grid cell 预测一个类，预测20个类，需要20个这样的数据结构 --->[20, batch, h*w, 3, 4]   

        '''
        bboxes = F.tile(bbox, reps=(self._classes, 1, 1, 1, 1))
        scores = F.transpose(class_score, axes=(3, 0, 1, 2)).expand_dims(axis=-1)
        ids = F.broadcast_add(scores * 0, F.arange(0, self._classes).reshape((0, 1, 1, 1, 1))) 
        detections = F.concat(ids, scores, bboxes, dim=-1)
        detections = F.reshape(detections.transpose(axes=(1, 0, 2, 3, 4)), (0, -1, 6))
        return detections

    def reset_class(self, classes, reuse_weights=None):
        """Reset class prediction.
        Parameters
        ----------
        classes : type
            Description of parameter `classes`.
        reuse_weights : dict
            A {new_integer : old_integer} mapping dict that allows the new predictor to reuse the
            previously trained weights specified by the integer index.
        Returns
        -------
        type
            Description of returned object.
        """
        self._clear_cached_op()
        # keep old records
        old_classes = self._classes
        old_pred = self.prediction
        old_num_pred = self._num_pred
        ctx = list(old_pred.params.values())[0].list_ctx()
        self._classes = len(classes)
        self._num_pred = 1 + 4 + len(classes)
        all_pred = self._num_pred * self._num_anchors
        # to avoid deferred init, number of in_channels must be defined
        in_channels = list(old_pred.params.values())[0].shape[1]
        self.prediction = nn.Conv2D(
            all_pred, kernel_size=1, padding=0, strides=1,
            in_channels=in_channels, prefix=old_pred.prefix)
        self.prediction.initialize(ctx=ctx)
        if reuse_weights:
            new_pred = self.prediction
            assert isinstance(reuse_weights, dict)
            for old_params, new_params in zip(old_pred.params.values(), new_pred.params.values()):
                old_data = old_params.data()
                new_data = new_params.data()
                for k, v in reuse_weights.items():
                    if k >= self._classes or v >= old_classes:
                        warnings.warn("reuse mapping {}/{} -> {}/{} out of range".format(
                            k, self._classes, v, old_classes))
                        continue
                    for i in range(self._num_anchors):
                        off_new = i * self._num_pred
                        off_old = i * old_num_pred
                        # copy along the first dimension
                        new_data[1 + 4 + k + off_new] = old_data[1 + 4 + v + off_old]
                        # copy non-class weights as well
                        new_data[off_new : 1 + 4 + off_new] = old_data[off_old : 1 + 4 + off_old]
                # set data to new conv layers
                new_params.set_data(new_data)

class YOLOV3(gluon.HybridBlock):
    """YOLO V3 detection network.
    Reference: https://arxiv.org/pdf/1804.02767.pdf.
    Parameters
    ----------
    stages : mxnet.gluon.HybridBlock
        Staged feature extraction blocks.
        For example, 3 stages and 3 YOLO output layers are used original paper.
    channels : iterable
        Number of conv channels for each appended stage.
        `len(channels)` should match `len(stages)`.
    num_class : int
        Number of foreground objects.
    anchors : iterable
        The anchor setting. `len(anchors)` should match `len(stages)`.
    strides : iterable
        Strides of feature map. `len(strides)` should match `len(stages)`.
    alloc_size : tuple of int, default is (128, 128)
        For advanced users. Define `alloc_size` to generate large enough anchor
        maps, which will later saved in parameters. During inference, we support arbitrary
        input image by cropping corresponding area of the anchor map. This allow us
        to export to symbol so we can run it in c++, Scalar, etc.
    nms_thresh : float, default is 0.45.
        Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
    nms_topk : int, default is 400
        Apply NMS to top k detection results, use -1 to disable so that every Detection
         result is used in NMS.
    post_nms : int, default is 100
        Only return top `post_nms` detection results, the rest is discarded. The number is
        based on COCO dataset which has maximum 100 objects per image. You can adjust this
        number if expecting more objects. You can use -1 to return all detections.
    pos_iou_thresh : float, default is 1.0
        IOU threshold for true anchors that match real objects.
        'pos_iou_thresh < 1' is not implemented.
    ignore_iou_thresh : float
        Anchors that has IOU in `range(ignore_iou_thresh, pos_iou_thresh)` don't get
        penalized of objectness score.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    """
    '''
                        stages, [512, 256, 128], anchors, [8, 16, 32], classes,
    '''
    def __init__(self, stages, channels, anchors, strides, classes, alloc_size=(128, 128),
                 nms_thresh=0.45, nms_topk=400, post_nms=100, pos_iou_thresh=1.0,
                 ignore_iou_thresh=0.7, norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
        super(YOLOV3, self).__init__(**kwargs)
        self._classes = classes
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms
        self._pos_iou_thresh = pos_iou_thresh
        self._ignore_iou_thresh = ignore_iou_thresh
        if pos_iou_thresh >= 1:
            self._target_generator = YOLOV3TargetMerger(len(classes), ignore_iou_thresh)
        else:
            raise NotImplementedError(
                "pos_iou_thresh({}) < 1.0 is not implemented!".format(pos_iou_thresh))
        self._loss = YOLOV3Loss()

        ''' 对整个YOLO模型 进行整合'''
        with self.name_scope():
            self.stages = nn.HybridSequential()
            self.transitions = nn.HybridSequential()
            self.yolo_blocks = nn.HybridSequential()
            self.yolo_outputs = nn.HybridSequential()
            # note that anchors and strides should be used in reverse order
            ''' MXNet 将Darknet 化作三个stage，以配合输出的三阶段 '''
            for i, stage, channel, anchor, stride in zip(
                    range(len(stages)), stages, channels, anchors[::-1], strides[::-1]):

                self.stages.add(stage)

                block = YOLODetectionBlockV3(
                    channel, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
                self.yolo_blocks.add(block)

                output = YOLOOutputV3(i, len(classes), anchor, stride, alloc_size=alloc_size)
                self.yolo_outputs.add(output)

                if i > 0:
                    self.transitions.add(_conv2d(channel, 1, 0, 1,
                                                 norm_layer=norm_layer, norm_kwargs=norm_kwargs))

    def hybrid_forward(self, F, x, *args):
        """YOLOV3 network hybrid forward.
        Parameters
        ----------
        F : mxnet.nd or mxnet.sym
            `F` is mxnet.sym if hybridized or mxnet.nd if not.
        x : mxnet.nd.NDArray
            Input data.
        *args : optional, mxnet.nd.NDArray
            During training, extra inputs are required:
            (gt_boxes, obj_t, centers_t, scales_t, weights_t, clas_t)
            These are generated by YOLOV3PrefetchTargetGenerator in dataloader transform function.
        Returns
        -------
        (tuple of) mxnet.nd.NDArray
            
            During inference, return detections in shape (B, N, 6)  # 6:bbox坐标有4个，一个置信度，一个类别
            with format (cid, score, xmin, ymin, xmax, ymax)
            During training, return losses only: (obj_loss, center_loss, scale_loss, cls_loss).
        """
        all_box_centers = []
        all_box_scales = []
        all_objectness = []
        all_class_pred = []
        all_anchors = []
        all_offsets = []
        all_feat_maps = []
        all_detections = []
        routes = []                 # 这是用于后续上采样
        '''
            # 都是三个阶段 相互对应
            # 在训练时，输入的 x.shape = [batch，channel, width, height]
        '''
        for stage, block, output in zip(self.stages, self.yolo_blocks, self.yolo_outputs):
            x = stage(x)
            routes.append(x)  # 存储的是每个阶段的存储值，后续在此基本上进行上采样

        # the YOLO output layers are used in reverse order, i.e., from very deep layers to shallow
        for i, block, output in zip(range(len(routes)), self.yolo_blocks, self.yolo_outputs):
            # block : YOLODetectionBlockV3 
            '''
                x  : 是用于后序上采样，
                    第一次使用的x,   是最后一个阶段stride = 32 的输出
                    第二次使用的x，  是第二个阶段stride = 16 的输出
                    最后一次使用的x，是最浅层的阶段，stride = 8的输出

                tip: 用于这一阶段的输出预测

                if i >= len(routes) - 1:
                    break
                只有两个部分。
            '''
            x, tip = block(x)  
            if autograd.is_training():
                '''
                    从tip输出连接到YOLOouyput
                     During training, return (bbox, raw_box_centers, raw_box_scales, objness,
                    class_pred, anchors, offsets)
                '''
                dets, box_centers, box_scales, objness, class_pred, anchors, offsets = output(tip) 
                '''
                    0 : 直接复制这个维度
                    -3: 在axis = 0之后的连续两个维度的乘积作为axis=1的维度
                    -1：最后一个维度自动reference
                    dets: 
                        shape = [batch, h*W*3, 4]       # 得到的是最终的bbox预测输出
                    box_centers:
                        shape = [batch, h*w,3, 2]   --> [batch, h*w*3, 2]
                    box_scales:
                        shape = [batch, h*w,3, 2]   --> [batch, h*w*3, 2]
                    objness:
                        shape = [batch, h*w,3, 1]   --> [batch, h*w*3, 1]  # 因为在损失函数中，有三部的损失，所以要留下
                    class_pred:
                        shape = [batch, h*w,3,20]  --> [batch, h*w*3, 20]
                    anchor
                        shape  = [1, 1, 3, 2]
                    offsets
                        shape  = [1, h*w,1,2] 
                '''
                all_box_centers.append(box_centers.reshape((0, -3, -1)))
                all_box_scales.append(box_scales.reshape((0, -3, -1)))
                all_objectness.append(objness.reshape((0, -3, -1)))
                all_class_pred.append(class_pred.reshape((0, -3, -1)))
                                                                        # 最先输出的                 次输出                         最后输出的
                all_anchors.append(anchors)  # for循环结束，是 anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
                all_offsets.append(offsets)
                # here we use fake featmap to reduce memory consuption, only shape[2, 3] is used
                # fake_featmap = tip[0][0] 分别是13*13, 26*26, 52*52 ，对应着stride = 32, 16, 8
                fake_featmap = F.zeros_like(tip.slice_axis(
                    axis=0, begin=0, end=1).slice_axis(axis=1, begin=0, end=1))
                all_feat_maps.append(fake_featmap)
            else:
                dets = output(tip)  
            # inference 输出是： shape = [batch, 20*h*w*3, 6]
            # trian时   输出是： shape = [batch, h*W*3, 4]     
            all_detections.append(dets) 
            '''
                if i >= len(routes) - 1:
                    break
                为分界线
                上面的部分，循环了3次，下面的部分进行循环了2次。

                前面两个阶段有进行融合，第三个阶段没有融合，直接进行输出。
            '''
            if i >= len(routes) - 1:
                break

            '''
                下面是用于和后面继续上采样相关
                transitions 就是一个1*1的卷积，改变通道数
                upsample = _upsample(x, stride=2) 再继续上采用 
                然后和浅层的特征route_now，所以这里需要使用routes[::-1][i + 1]进行融合。
            '''
            # add transition layers
            # YOLODetectionBlockV3 输出的一个分支，经过transition
            x = self.transitions[i](x)  
            # upsample feature map reverse to shallow layers
            upsample = _upsample(x, stride=2)
            route_now = routes[::-1][i + 1]   #  1 2 

            '''
                # route_now 是Darknet某个阶段的输出，如果   
                # 前阶段            stride = 8,  此时 feature map: [batch, 256, 52, 52]
                # 那么route_now 就是stride = 16, 此时 feature map: [batch， 512, 26, 26]
                # upsample 输出的信息是： featur map: [batch, 1024， 16， 16]
                # upsample是从DarkNet-53的最后一层输出进行采样的，所以第一次循环结束时，stride = 8->16, upsample：将DarkNet-53的输出从8*8->16*16，实现大小的一致
                # 然后子通道维进行concatenate。
                # x = [batch, 1024 + 512, 16, 16]
                # F.slice_like(upsample, route_now * 0, axes=(2, 3)) 估计是为了保证二者一定一致，因为可能有时候细小的偏差

            '''
            x = F.concat(F.slice_like(upsample, route_now * 0, axes=(2, 3)), route_now, dim=1)  #

        '''
            for循环已经结束，下面开始处理for循环得到的结果
        '''
        if autograd.is_training():
            # during training, the network behaves differently since we don't need detection results
            # inferecne才需要检测结果
            
            if autograd.is_recording():
                '''
                下面的完成的是，把三个阶段采集到的信息融合在一起。从不同feature map采集的信息融合在一个grid cell对应的tensor里面。
                    # generate losses and return them directly
                        出于书写方便： 10647 = 13 * 13 * 3 +  26 * 26 * 3 + 52 * 52 * 3 记为：3*(h*W*3)，下面的3*(h*W*3) 都是表示三个通道的融合

                        # box_preds : shape = [batch, 3*(h*W*3), 4]
                        # all_preds : shape = [ 
                            [batch, 3*(h*W*3), 1] , 
                            [batch, 3*(h*W*3), 2] , 
                            [batch, 3*(h*W*3), 2] , 
                            [batch, 3*(h*W*3), 20] 
                        ]

                        # all_targets: shape = [
                            [batch, 3*(h*W*3), 1] ,  obj_t 
                            [batch, 3*(h*W*3), 2] ,  center_targets
                            [batch, 3*(h*W*3), 2] ,  scale_targets
                            [batch, 3*(h*W*3), 2] ,  weights
                            [batch, 3*(h*W*3), 20],  class_targets
                            [batch, 3*(h*W*3), 20],  class_mask
                        ]
                '''
                box_preds = F.concat(*all_detections, dim=1)
                all_preds = [F.concat(*p, dim=1) for p in [all_objectness, all_box_centers, all_box_scales, all_class_pred]]
                all_targets = self._target_generator(box_preds, *args) # 这里获得的是根据传入的box_preds对应的各个ground -truth
                return self._loss(*(all_preds + all_targets))

            # return raw predictions, this is only used in DataLoader transform function.
            return (F.concat(*all_detections, dim=1), 
                    all_anchors, all_offsets, all_feat_maps,  # 只是用到这三个数据
                    F.concat(*all_box_centers, dim=1), 
                    F.concat(*all_box_scales, dim=1),
                    F.concat(*all_objectness, dim=1), 
                    F.concat(*all_class_pred, dim=1))

        '''
            如果是直接在推理模式下，也不需要loss再进行后向传播
        '''
        # concat all detection results from different stages
        # 不同阶段的检测输出
        # inference 输出是： shape = [batch, 20*h*w*9, 6]
        result = F.concat(*all_detections, dim=1)  
        # apply nms per class
        if self.nms_thresh > 0 and self.nms_thresh < 1:
            result = F.contrib.box_nms(
                result, overlap_thresh=self.nms_thresh, valid_thresh=0.01,
                topk=self.nms_topk, id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms) # result[0:100]
        ids = result.slice_axis(axis=-1, begin=0, end=1)        # result[..., 0]
        scores = result.slice_axis(axis=-1, begin=1, end=2)     # result[..., 1]
        bboxes = result.slice_axis(axis=-1, begin=2, end=None)  # result[..., 2:]
        return ids, scores, bboxes
   
    @property
    def num_class(self):
        """Number of (non-background) categories.
        Returns
        -------
        int
            Number of (non-background) categories.
        """
        return self._num_class

    @property
    def classes(self):
        """Return names of (non-background) categories.
        Returns
        -------
        iterable of str
            Names of (non-background) categories.
        """
        return self._classes
   
    def set_nms(self, nms_thresh=0.45, nms_topk=400, post_nms=100):
        """Set non-maximum suppression parameters.
        Parameters
        ----------
        nms_thresh : float, default is 0.45.
            Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
        nms_topk : int, default is 400
            Apply NMS to top k detection results, use -1 to disable so that every Detection
             result is used in NMS.
        post_nms : int, default is 100
            Only return top `post_nms` detection results, the rest is discarded. The number is
            based on COCO dataset which has maximum 100 objects per image. You can adjust this
            number if expecting more objects. You can use -1 to return all detections.
        Returns
        -------
        None
        """
        self._clear_cached_op()
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

    def reset_class(self, classes, reuse_weights=None):
        """Reset class categories and class predictors.
        Parameters
        ----------
        classes : iterable of str
            The new categories. ['apple', 'orange'] for example.
        reuse_weights : dict
            A {new_integer : old_integer} or mapping dict or {new_name : old_name} mapping dict,
            or a list of [name0, name1,...] if class names don't change.
            This allows the new predictor to reuse the
            previously trained weights specified.

        Example
        -------
        >>> net = gluoncv.model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
        >>> # use direct name to name mapping to reuse weights
        >>> net.reset_class(classes=['person'], reuse_weights={'person':'person'})
        >>> # or use interger mapping, person is the 14th category in VOC
        >>> net.reset_class(classes=['person'], reuse_weights={0:14})
        >>> # you can even mix them
        >>> net.reset_class(classes=['person'], reuse_weights={'person':14})
        >>> # or use a list of string if class name don't change
        >>> net.reset_class(classes=['person'], reuse_weights=['person'])

        """
        self._clear_cached_op()
        old_classes = self._classes
        self._classes = classes
        if self._pos_iou_thresh >= 1:
            self._target_generator = YOLOV3TargetMerger(len(classes), self._ignore_iou_thresh)
        if isinstance(reuse_weights, (dict, list)):
            if isinstance(reuse_weights, dict):
                # trying to replace str with indices
                for k, v in reuse_weights.items():
                    if isinstance(v, str):
                        try:
                            v = old_classes.index(v)  # raise ValueError if not found
                        except ValueError:
                            raise ValueError(
                                "{} not found in old class names {}".format(v, old_classes))
                        reuse_weights[k] = v
                    if isinstance(k, str):
                        try:
                            new_idx = self._classes.index(k)  # raise ValueError if not found
                        except ValueError:
                            raise ValueError(
                                "{} not found in new class names {}".format(k, self._classes))
                        reuse_weights.pop(k)
                        reuse_weights[new_idx] = v
            else:
                new_map = {}
                for x in reuse_weights:
                    try:
                        new_idx = self._classes.index(x)
                        old_idx = old_classes.index(x)
                        new_map[new_idx] = old_idx
                    except ValueError:
                        warnings.warn("{} not found in old: {} or new class names: {}".format(
                            x, old_classes, self._classes))
                reuse_weights = new_map

        for outputs in self.yolo_outputs:
            outputs.reset_class(classes, reuse_weights=reuse_weights)

def get_yolov3(name, stages, filters, anchors, strides, classes,
               dataset, pretrained=False, ctx=mx.cpu(),
               root=os.path.join('~', '.mxnet', 'models'), **kwargs):
    """Get YOLOV3 models.
    Parameters
    ----------
    name : str or None
        Model name, if `None` is used, you must specify `features` to be a `HybridBlock`.
    stages : iterable of str or `HybridBlock`
        List of network internal output names, in order to specify which layers are
        used for predicting bbox values.
        If `name` is `None`, `features` must be a `HybridBlock` which generate multiple
        outputs for prediction.
    filters : iterable of float or None
        List of convolution layer channels which is going to be appended to the base
        network feature extractor. If `name` is `None`, this is ignored.
    sizes : iterable fo float
        Sizes of anchor boxes, this should be a list of floats, in incremental order.
        The length of `sizes` must be len(layers) + 1. For example, a two stage SSD
        model can have ``sizes = [30, 60, 90]``, and it converts to `[30, 60]` and
        `[60, 90]` for the two stages, respectively. For more details, please refer
        to original paper.
    ratios : iterable of list
        Aspect ratios of anchors in each output layer. Its length must be equals
        to the number of SSD output layers.
    steps : list of int
        Step size of anchor boxes in each output layer.
    classes : iterable of str
        Names of categories.
    dataset : str
        Name of dataset. This is used to identify model name because models trained on
        different datasets are going to be very different.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained_base : bool or str, optional, default is True
        Load pretrained base network, the extra layers are randomized. Note that
        if pretrained is `True`, this has no effect.
    ctx : mxnet.Context
        Context such as mx.cpu(), mx.gpu(0).
    root : str
        Model weights storing path.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    HybridBlock
        A YOLOV3 detection network.
    """
    '''
    定义： 
    def get_yolov3(name, stages, filters, anchors, strides, classes,
               dataset, pretrained=False, ctx=mx.cpu(),
               root=os.path.join('~', '.mxnet', 'models'), **kwargs):

    调用： s
    get_yolov3(
        'darknet53', stages, [512, 256, 128], anchors, [8, 16, 32], classes, 'voc',
        pretrained=pretrained, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
    '''

    net = YOLOV3(stages, filters, anchors, strides, classes=classes, **kwargs)
    if pretrained:
        from ..model_store import get_model_file
        full_name = '_'.join(('yolo3', name, dataset))
        net.load_parameters(get_model_file(full_name, tag=pretrained, root=root), ctx=ctx)
    return net

def yolo3_darknet53_voc(pretrained_base=True, pretrained=False,
                        norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    """YOLO3 multi-scale with darknet53 base network on VOC dataset.
    Parameters
    ----------
    pretrained_base : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.
    """
    from ...data import VOCDetection
    pretrained_base = False if pretrained else pretrained_base
    base_net = darknet53(
        pretrained=pretrained_base, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
    '''
        将Darknet 按照如下阶段进行划分成三个阶段。
        其中，Darknet-53的feature就是卷积层

        anchors 是三个scale对应的锚点
        strides 是三个scale对应的补偿
    '''
    stages = [base_net.features[:15], base_net.features[15:24], base_net.features[24:]]
    
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    strides = [8, 16, 32]
    classes = VOCDetection.CLASSES
    return get_yolov3(
        'darknet53', stages, [512, 256, 128], anchors, strides, classes, 'voc',
        pretrained=pretrained, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)

def yolo3_darknet53_coco(pretrained_base=True, pretrained=False,
                         norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    """YOLO3 multi-scale with darknet53 base network on COCO dataset.
    Parameters
    ----------
    pretrained_base : boolean
        Whether fetch and load pretrained weights for base network.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.
    """
    from ...data import COCODetection
    pretrained_base = False if pretrained else pretrained_base
    base_net = darknet53(
        pretrained=pretrained_base, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
    stages = [base_net.features[:15], base_net.features[15:24], base_net.features[24:]]
    anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
    strides = [8, 16, 32]
    classes = COCODetection.CLASSES
    return get_yolov3(
        'darknet53', stages, [512, 256, 128], anchors, strides, classes, 'coco',
        pretrained=pretrained, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)

def yolo3_darknet53_custom(classes, transfer=None, pretrained_base=True, pretrained=False,
                           norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    """YOLO3 multi-scale with darknet53 base network on custom dataset.
    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from yolo networks trained on other
        datasets.
    pretrained_base : boolean
        Whether fetch and load pretrained weights for base network.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.
    """
    if pretrained:
        warnings.warn("Custom models don't provide `pretrained` weights, ignored.")
    if transfer is None:
        base_net = darknet53(
            pretrained=pretrained_base, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
        stages = [base_net.features[:15], base_net.features[15:24], base_net.features[24:]]
        anchors = [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]]
        strides = [8, 16, 32]
        net = get_yolov3(
            'darknet53', stages, [512, 256, 128], anchors, strides, classes, '',
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
    else:
        from ...model_zoo import get_model
        net = get_model('yolo3_darknet53_' + str(transfer), pretrained=True, **kwargs)
        reuse_classes = [x for x in classes if x in net.classes]
        net.reset_class(classes, reuse_weights=reuse_classes)
    return net

def yolo3_mobilenet1_0_voc(pretrained_base=True, pretrained=False,
                           norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    """YOLO3 multi-scale with mobilenet base network on VOC dataset.
    Parameters
    ----------
    pretrained_base : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.
    """
    from ...data import VOCDetection

    pretrained_base = False if pretrained else pretrained_base
    base_net = get_mobilenet(
        multiplier=1,
        pretrained=pretrained_base,
        norm_layer=norm_layer, norm_kwargs=norm_kwargs,
        **kwargs)
    stages = [base_net.features[:33],
              base_net.features[33:69],
              base_net.features[69:-2]]

    anchors = [[10, 13, 16, 30, 33, 23],
               [30, 61, 62, 45, 59, 119],
               [116, 90, 156, 198, 373, 326]]
    strides = [8, 16, 32]
    classes = VOCDetection.CLASSES
    return get_yolov3(
        'mobilenet1.0', stages, [512, 256, 128], anchors, strides, classes, 'voc',
        pretrained=pretrained, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)

def yolo3_mobilenet1_0_custom(classes, transfer=None, pretrained_base=True, pretrained=False,
                              norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    """YOLO3 multi-scale with mobilenet base network on custom dataset.
    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from yolo networks trained on other
        datasets.
    pretrained_base : boolean
        Whether fetch and load pretrained weights for base network.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.
    """
    if pretrained:
        warnings.warn("Custom models don't provide `pretrained` weights, ignored.")
    if transfer is None:
        base_net = get_mobilenet(multiplier=1,
                                 pretrained=pretrained_base,
                                 norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                 **kwargs)
        stages = [base_net.features[:33],
                  base_net.features[33:69],
                  base_net.features[69:-2]]
        anchors = [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]]
        strides = [8, 16, 32]
        net = get_yolov3(
            'mobilenet1.0', stages, [512, 256, 128], anchors, strides, classes, '',
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
    else:
        from ...model_zoo import get_model
        net = get_model(
            'yolo3_mobilenet1.0_' +
            str(transfer),
            pretrained=True,
            **kwargs)
        reuse_classes = [x for x in classes if x in net.classes]
        net.reset_class(classes, reuse_weights=reuse_classes)
    return net

def yolo3_mobilenet1_0_coco(pretrained_base=True, pretrained=False, norm_layer=BatchNorm,
                            norm_kwargs=None, **kwargs):
    """YOLO3 multi-scale with mobilenet base network on COCO dataset.
    Parameters
    ----------
    pretrained_base : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.
    """
    from ...data import COCODetection

    pretrained_base = False if pretrained else pretrained_base
    base_net = get_mobilenet(
        multiplier=1,
        pretrained=pretrained_base,
        norm_layer=norm_layer, norm_kwargs=norm_kwargs,
        **kwargs)
    stages = [base_net.features[:33],
              base_net.features[33:69],
              base_net.features[69:-2]]

    anchors = [[10, 13, 16, 30, 33, 23],
               [30, 61, 62, 45, 59, 119],
               [116, 90, 156, 198, 373, 326]]
    strides = [8, 16, 32]
    classes = COCODetection.CLASSES
    return get_yolov3(
        'mobilenet1.0', stages, [512, 256, 128], anchors, strides, classes, 'coco',
        pretrained=pretrained, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)

def yolo3_mobilenet0_25_voc(pretrained_base=True, pretrained=False,
                            norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    """YOLO3 multi-scale with mobilenet0.25 base network on VOC dataset.
    Parameters
    ----------
    pretrained_base : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.
    """
    from ...data import VOCDetection

    pretrained_base = False if pretrained else pretrained_base
    base_net = get_mobilenet(
        multiplier=0.25,
        pretrained=pretrained_base,
        norm_layer=norm_layer, norm_kwargs=norm_kwargs,
        **kwargs)
    stages = [base_net.features[:33],
              base_net.features[33:69],
              base_net.features[69:-2]]

    anchors = [[10, 13, 16, 30, 33, 23],
               [30, 61, 62, 45, 59, 119],
               [116, 90, 156, 198, 373, 326]]
    strides = [8, 16, 32]
    classes = VOCDetection.CLASSES
    return get_yolov3(
        'mobilenet0.25', stages, [256, 128, 128], anchors, strides, classes, 'voc',
        pretrained=pretrained, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)

def yolo3_mobilenet0_25_custom(classes, transfer=None, pretrained_base=True, pretrained=False,
                               norm_layer=BatchNorm, norm_kwargs=None, **kwargs):
    """YOLO3 multi-scale with mobilenet0.25 base network on custom dataset.
    Parameters
    ----------
    classes : iterable of str
        Names of custom foreground classes. `len(classes)` is the number of foreground classes.
    transfer : str or None
        If not `None`, will try to reuse pre-trained weights from yolo networks trained on other
        datasets.
    pretrained_base : boolean
        Whether fetch and load pretrained weights for base network.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.
    """
    if pretrained:
        warnings.warn("Custom models don't provide `pretrained` weights, ignored.")
    if transfer is None:
        base_net = get_mobilenet(multiplier=0.25,
                                 pretrained=pretrained_base,
                                 norm_layer=norm_layer, norm_kwargs=norm_kwargs,
                                 **kwargs)
        stages = [base_net.features[:33],
                  base_net.features[33:69],
                  base_net.features[69:-2]]
        anchors = [
            [10, 13, 16, 30, 33, 23],
            [30, 61, 62, 45, 59, 119],
            [116, 90, 156, 198, 373, 326]]
        strides = [8, 16, 32]
        net = get_yolov3(
            'mobilenet0.25', stages, [256, 128, 128], anchors, strides, classes, '',
            norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
    else:
        from ...model_zoo import get_model
        net = get_model(
            'yolo3_mobilenet0.25_' +
            str(transfer),
            pretrained=True,
            **kwargs)
        reuse_classes = [x for x in classes if x in net.classes]
        net.reset_class(classes, reuse_weights=reuse_classes)
    return net

def yolo3_mobilenet0_25_coco(pretrained_base=True, pretrained=False, norm_layer=BatchNorm,
                             norm_kwargs=None, **kwargs):
    """YOLO3 multi-scale with mobilenet0.25 base network on COCO dataset.
    Parameters
    ----------
    pretrained_base : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    pretrained : bool or str
        Boolean value controls whether to load the default pretrained weights for model.
        String value represents the hashtag for a certain version of pretrained weights.
    norm_layer : object
        Normalization layer used (default: :class:`mxnet.gluon.nn.BatchNorm`)
        Can be :class:`mxnet.gluon.nn.BatchNorm` or :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    norm_kwargs : dict
        Additional `norm_layer` arguments, for example `num_devices=4`
        for :class:`mxnet.gluon.contrib.nn.SyncBatchNorm`.
    Returns
    -------
    mxnet.gluon.HybridBlock
        Fully hybrid yolo3 network.
    """
    from ...data import COCODetection

    pretrained_base = False if pretrained else pretrained_base
    base_net = get_mobilenet(
        multiplier=0.25,
        pretrained=pretrained_base,
        norm_layer=norm_layer, norm_kwargs=norm_kwargs,
        **kwargs)
    stages = [base_net.features[:33],
              base_net.features[33:69],
              base_net.features[69:-2]]

    anchors = [[10, 13, 16, 30, 33, 23],
               [30, 61, 62, 45, 59, 119],
               [116, 90, 156, 198, 373, 326]]
    strides = [8, 16, 32]
    classes = COCODetection.CLASSES
    return get_yolov3(
        'mobilenet1.0', stages, [256, 128, 128], anchors, strides, classes, 'coco',
        pretrained=pretrained, norm_layer=norm_layer, norm_kwargs=norm_kwargs, **kwargs)
