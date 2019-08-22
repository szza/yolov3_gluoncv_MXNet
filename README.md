# yolov3_MXNet_gluoncv 
MXNet的一个开源项目gluoncv(链接：https://github.com/dmlc/gluon-cv/ ) 里的yolov3代码，写了一份中文注解

需要安装：MXNet,gluoncv 
 
阅读顺序：
  建议Darknet.py --> yolov3.py --> train_yolo3.py --> yolo_target.py

'''
 '''
            objectness.squeeze(axis=-1) 
                shape = [1, 3549, 9] 
            class_targets              
                shape = [gt_ids.shape[0], 3549, 9, self._num_class] 
                默认值全是-1，即忽略
                
            '''
            class_targets = nd.one_hot(objectness.squeeze(axis=-1), depth=self._num_class)
            class_targets[:] = -1  # prefill -1 for ignores
           
            '''
            # for each ground-truth, find the best matching anchor within the particular grid
            # for instance, center of object 1 reside in grid (3, 4) in (16, 16) feature map
            # then only the anchor in (3, 4) is going to be matched
                即，对于每个ground-truth寻找与之最匹配的anchor box，要在ground-truth所在的grid cell产生的box里寻找

            
            shift_gt_boxes 还是一个四角坐标，[1, M, 4]
 
            anchor_boxes shape = [1, 9,4] 前面两个数是表示是box的中心(0, 0)，后面两个数是priors的宽和高
            shift_anchor_boxes 化为四角坐标： [1, 9, 4]
            

            ious shape = [1,9, M]，M是具体某个gt-bbox里面的objness数量

            gtx shape:[1, M, 1]
            gty shape:[1, M, 1]
            gtw shape:[1, M, 1]
            gth shape:[1, M, 1]
            
            ''' 
            gtx, gty, gtw, gth = self.bbox2center(gt_boxes)  
            shift_gt_boxes = nd.concat(-0.5 * gtw, -0.5 * gth, 0.5 * gtw, 0.5 * gth, dim=-1)  # zero center  
            
            anchor_boxes = nd.concat(0 * all_anchors, all_anchors, dim=-1)  # zero center anchors
            shift_anchor_boxes = self.bbox2corner(anchor_boxes) # 又转换为四角坐标

            ious = nd.contrib.box_iou(shift_anchor_boxes, shift_gt_boxes).transpose((1, 0, 2))  # (1, 9, M)

'''

