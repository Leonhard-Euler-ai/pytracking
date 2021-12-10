from pytracking.tracker.base import BaseTracker
import torch
import torch.nn.functional as F
import math
import time
from pytracking import dcf, TensorList
from pytracking.features.preprocessing import numpy_to_torch
from pytracking.utils.plotting import show_tensor, plot_graph
from pytracking.features.preprocessing import sample_patch_multiscale, sample_patch_transformed
from pytracking.features import augmentation
import ltr.data.bounding_box_utils as bbutils
from ltr.models.target_classifier.initializer import FilterInitializerZero
from ltr.models.layers import activation


class DiMP(BaseTracker):

    multiobj_mode = 'parallel'

    def initialize_features(self):
        if not getattr(self, 'features_initialized', False):
            self.params.net.initialize()
        self.features_initialized = True

    def initialize(self, image, info: dict) -> dict:
        # Initialize some stuff
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Initialize network  包括使用超参生成dimpnet网络，加载网络参数
        self.initialize_features()

        # The DiMP network
        self.net = self.params.net

        # Time initialization
        tic = time.time()

        # Convert image
        im = numpy_to_torch(image)

        # Get target position and size
        state = info['init_bbox']
        self.pos = torch.Tensor([state[1] + (state[3] - 1)/2, state[0] + (state[2] - 1)/2])
        self.target_sz = torch.Tensor([state[3], state[2]])

        # Get object id
        self.object_id = info.get('object_ids', [None])[0]
        self.id_str = '' if self.object_id is None else ' {}'.format(self.object_id)

        # Set sizes
        self.image_sz = torch.Tensor([im.shape[2], im.shape[3]])
        sz = self.params.image_sample_size
        sz = torch.Tensor([sz, sz] if isinstance(sz, int) else sz)
        if self.params.get('use_image_aspect_ratio', False):
            sz = self.image_sz * sz.prod().sqrt() / self.image_sz.prod().sqrt()  # 相当于长宽分别乘以一个浮点数，长宽比保持不变
            stride = self.params.get('feature_stride', 32)
            sz = torch.round(sz / stride) * stride  # 保证长宽为整数
        self.img_sample_sz = sz
        self.img_support_sz = self.img_sample_sz  # 这个参数在track的获取样本中心位置时和iounet中用到

        # Set search area
        search_area = torch.prod(self.target_sz * self.params.search_area_scale).item()
        self.target_scale =  math.sqrt(search_area) / self.img_sample_sz.prod().sqrt()  # 目标搜索区域的大小除以图片样本大小得到的比例

        # Target size in base scale
        self.base_target_sz = self.target_sz / self.target_scale

        # Setup scale factors 比例因子
        if not self.params.has('scale_factors'):
            self.params.scale_factors = torch.ones(1)  # 超参中没有设置，因此只有一个比例因子为1，在track中输入一个图像im，得道的im_patch个数也是1
        elif isinstance(self.params.scale_factors, (list, tuple)):
            self.params.scale_factors = torch.Tensor(self.params.scale_factors)

        # Setup scale bounds
        self.min_scale_factor = torch.max(10 / self.base_target_sz)  # 最小比例因子，相当于目标的长宽的最小值不能小于10
        self.max_scale_factor = torch.min(self.image_sz / self.base_target_sz)  # 最大比例因子，相当于目标的长宽的最大值不能大于超过图像的边缘

        # Extract and transform sample
        init_backbone_feat = self.generate_init_samples(im)

        # Initialize classifier  主要是获得分类器的属性，使用样本初始化内存空间，获取过滤器f
        self.init_classifier(init_backbone_feat)

        # Initialize IoUNet  主要是在reference分支上求得modulation
        if self.params.get('use_iou_net', True):
            self.init_iou_net(init_backbone_feat)

        out = {'time': time.time() - tic}
        return out


    def track(self, image, info: dict = None) -> dict:
        self.debug_info = {}

        self.frame_num += 1
        self.debug_info['frame_num'] = self.frame_num

        # Convert image
        im = numpy_to_torch(image)

        # ------- LOCALIZATION ------- #

        # Extract backbone features  坐标是左上和右下角的
        backbone_feat, sample_coords, im_patches = self.extract_backbone_features(im, self.get_centered_sample_pos(),
                                                                      self.target_scale * self.params.scale_factors,
                                                                      self.img_sample_sz)
        # Extract classification features
        test_x = self.get_classification_features(backbone_feat)

        # Location of sample  sample_pos是样本中心坐标，采样尺度是左上和右下框定的采样区域大小相对于[288，288]的倍数
        sample_pos, sample_scales = self.get_sample_location(sample_coords)

        # Compute classification scores
        scores_raw = self.classify_target(test_x)

        # Localize the target
        translation_vec, scale_ind, s, flag = self.localize_target(scores_raw, sample_pos, sample_scales)
        new_pos = sample_pos[scale_ind,:] + translation_vec  # 样本中心是根据左上和右下坐标求得的，tran_vec也是原图尺度上的，获得新的样本中心点

        # Update position and scale
        if flag != 'not_found':
            if self.params.get('use_iou_net', True):  # 默认使用iou_net
                update_scale_flag = self.params.get('update_scale_when_uncertain', True) or flag != 'uncertain'  # 更新scale标志，为True
                if self.params.get('use_classifier', True):  # 默认使用use_classifier
                    self.update_state(new_pos)
                # 使用ATOM_iou_net refine bbox,并更新参数
                self.refine_target_box(backbone_feat, sample_pos[scale_ind,:], sample_scales[scale_ind], scale_ind, update_scale_flag)
            elif self.params.get('use_classifier', True):
                self.update_state(new_pos, sample_scales[scale_ind])


        # ------- UPDATE ------- #

        update_flag = flag not in ['not_found', 'uncertain']  # 当被跟踪的目标状态是normal或hard_negative时才影响分类器的更新
        hard_negative = (flag == 'hard_negative')  # 当为hn是lr会更高一些，dimp50.py里是0.02正常是0.01
        learning_rate = self.params.get('hard_negative_learning_rate', None) if hard_negative else None

        # 更新分类器
        if update_flag and self.params.get('update_classifier', False):
            # Get train sample 将当前样本加入训练样本中
            train_x = test_x[scale_ind:scale_ind+1, ...]

            # Create target_box and label for spatial sample
            target_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos[scale_ind,:], sample_scales[scale_ind])

            # Update the classifier model
            self.update_classifier(train_x, target_box, learning_rate, s[scale_ind,...])

        # Set the pos of the tracker to iounet pos
        if self.params.get('use_iou_net', True) and flag != 'not_found' and hasattr(self, 'pos_iounet'):
            self.pos = self.pos_iounet.clone()

        score_map = s[scale_ind, ...]   # 置信度分数map
        max_score = torch.max(score_map).item()  # 最大置信度分数

        # Visualize and set debug info 可视化
        # 重新设置搜索区域，由左上角坐标和大小组成 yxhw
        self.search_area_box = torch.cat((sample_coords[scale_ind,[1,0]], sample_coords[scale_ind,[3,2]] - sample_coords[scale_ind,[1,0]] - 1))
        self.debug_info['flag' + self.id_str] = flag
        self.debug_info['max_score' + self.id_str] = max_score
        if self.visdom is not None:
            self.visdom.register(score_map, 'heatmap', 2, 'Score Map' + self.id_str)
            self.visdom.register(self.debug_info, 'info_dict', 1, 'Status')
        elif self.params.debug >= 2:
            show_tensor(score_map, 5, title='Max score = {:.2f}'.format(max_score))

        # Compute output bounding box  new_state format: xywh
        new_state = torch.cat((self.pos[[1,0]] - (self.target_sz[[1,0]]-1)/2, self.target_sz[[1,0]]))

        if self.params.get('output_not_found_box', False) and flag == 'not_found':  # dimp50里没这个属性,目标丢失时也会输出对应的bbox，而不是[-1,-1,-1,-1]
            output_state = [-1, -1, -1, -1]
        else:
            output_state = new_state.tolist()

        out = {'target_bbox': output_state}
        return out


    def get_sample_location(self, sample_coord):
        """Get the location of the extracted sample."""
        sample_coord = sample_coord.float()  # 左上和右下坐标
        sample_pos = 0.5*(sample_coord[:,:2] + sample_coord[:,2:] - 1)
        sample_scales = ((sample_coord[:,2:] - sample_coord[:,:2]) / self.img_sample_sz).prod(dim=1).sqrt() # 根据坐标采样得到的im_patch被插值过，因此坐标框定的区域大小可能不等于self.img_sample_sz
        return sample_pos, sample_scales

    def get_centered_sample_pos(self):
        """Get the center position for the new sample. Make sure the target is correctly centered. 这个中心是什么意思 是目标也是样本的中心，目标就在样本中心？ 这个函数产生样本中心的操作好像是对后面根据预测的score定位有关"""
        return self.pos + ((self.feature_sz + self.kernel_size) % 2) * self.target_scale * \
               self.img_support_sz / (2*self.feature_sz)

    def classify_target(self, sample_x: TensorList):
        """Classify target by applying the DiMP filter."""
        with torch.no_grad():
            scores = self.net.classifier.classify(self.target_filter, sample_x)
        return scores

    def localize_target(self, scores, sample_pos, sample_scales):
        """Run the target localization."""
        # score的维度：Dimensions (images_in_sequence, sequences, yH, yW)  去掉sequences维度
        scores = scores.squeeze(1)

        preprocess_method = self.params.get('score_preprocess', 'none')
        if preprocess_method == 'none':
            pass
        elif preprocess_method == 'exp':
            scores = scores.exp()
        elif preprocess_method == 'softmax':
            reg_val = getattr(self.net.classifier.filter_optimizer, 'softmax_reg', None)
            scores_view = scores.view(scores.shape[0], -1)
            scores_softmax = activation.softmax_reg(scores_view, dim=-1, reg=reg_val)
            scores = scores_softmax.view(scores.shape)
        else:
            raise Exception('Unknown score_preprocess in params.')

        score_filter_ksz = self.params.get('score_filter_ksz', 1)  # 置信分数过滤器大小
        if score_filter_ksz > 1:
            assert score_filter_ksz % 2 == 1
            kernel = scores.new_ones(1,1,score_filter_ksz,score_filter_ksz)
            scores = F.conv2d(scores.view(-1,1,*scores.shape[-2:]), kernel, padding=score_filter_ksz//2).view(scores.shape)

        if self.params.get('advanced_localization', False):
            return self.localize_advanced(scores, sample_pos, sample_scales)

        # Get maximum
        score_sz = torch.Tensor(list(scores.shape[-2:]))
        score_center = (score_sz - 1)/2
        max_score, max_disp = dcf.max2d(scores)
        _, scale_ind = torch.max(max_score, dim=0)
        max_disp = max_disp[scale_ind,...].float().cpu().view(-1)
        target_disp = max_disp - score_center

        # Compute translation vector and scale change factor
        output_sz = score_sz - (self.kernel_size + 1) % 2
        translation_vec = target_disp * (self.img_support_sz / output_sz) * sample_scales[scale_ind]

        return translation_vec, scale_ind, scores, None


    def localize_advanced(self, scores, sample_pos, sample_scales):
        """Run the target advanced localization (as in ATOM)."""

        sz = scores.shape[-2:]
        score_sz = torch.Tensor(list(sz))
        output_sz = score_sz - (self.kernel_size + 1) % 2
        score_center = (score_sz - 1)/2

        # scores_hn是scores_hard_negative
        scores_hn = scores
        if self.output_window is not None and self.params.get('perform_hn_without_windowing', False):
            scores_hn = scores.clone()
            scores *= self.output_window

        max_score1, max_disp1 = dcf.max2d(scores)  # 在多尺度下max_score1可以是二维的  max_disp1是最大分数在2D图上的行列坐标
        _, scale_ind = torch.max(max_score1, dim=0)  # 找出多尺度下最大分数所在的尺度坐标
        sample_scale = sample_scales[scale_ind]
        max_score1 = max_score1[scale_ind]
        max_disp1 = max_disp1[scale_ind,...].float().cpu().view(-1)  # score map最大值对应的行列坐标
        target_disp1 = max_disp1 - score_center  # 分数最大的点距离score map中心点的距离

        # self.img_support_sz/output_sz是sample_size相对于scoremap size的比例
        # sample_scale是采样尺度=原图左上和右下框定的区域/样本输出大小
        # 前两项相乘是在样本尺度上的距离，再乘以采样尺度（原图/样本大小），是在原图尺度上的距离
        translation_vec1 = target_disp1 * (self.img_support_sz / output_sz) * sample_scale

        if max_score1.item() < self.params.target_not_found_threshold:  # 0.25
            return translation_vec1, scale_ind, scores_hn, 'not_found'
        if max_score1.item() < self.params.get('uncertain_threshold', -float('inf')):  # dimp50.py没有设置
            return translation_vec1, scale_ind, scores_hn, 'uncertain'
        if max_score1.item() < self.params.get('hard_sample_threshold', -float('inf')):  # dimp50.py没有设置
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        # Mask out target neighborhood  这里target_neigh_sz是在output_sz尺度下的大小，且以最大的score点为中心
        target_neigh_sz = self.params.target_neighborhood_scale * (self.target_sz / sample_scale) * (output_sz / self.img_support_sz)

        tneigh_top = max(round(max_disp1[0].item() - target_neigh_sz[0].item() / 2), 0)
        tneigh_bottom = min(round(max_disp1[0].item() + target_neigh_sz[0].item() / 2 + 1), sz[0])
        tneigh_left = max(round(max_disp1[1].item() - target_neigh_sz[1].item() / 2), 0)
        tneigh_right = min(round(max_disp1[1].item() + target_neigh_sz[1].item() / 2 + 1), sz[1])
        scores_masked = scores_hn[scale_ind:scale_ind + 1, ...].clone()  # scores_hn是scores的克隆，不是同一个对象
        scores_masked[...,tneigh_top:tneigh_bottom,tneigh_left:tneigh_right] = 0  # 目标遮罩内的分数都为0

        # Find new maximum
        max_score2, max_disp2 = dcf.max2d(scores_masked)
        max_disp2 = max_disp2.float().cpu().view(-1)
        target_disp2 = max_disp2 - score_center
        translation_vec2 = target_disp2 * (self.img_support_sz / output_sz) * sample_scale
        # 原目标在原图尺度上的偏移
        prev_target_vec = (self.pos - sample_pos[scale_ind,:]) / ((self.img_support_sz / output_sz) * sample_scale)

        # Handle the different cases
        if max_score2 > self.params.distractor_threshold * max_score1:   # 说明出现干扰目标
            disp_norm1 = torch.sqrt(torch.sum((target_disp1-prev_target_vec)**2))  # 目标1相对之前目标位置的偏移距离的平方
            disp_norm2 = torch.sqrt(torch.sum((target_disp2-prev_target_vec)**2))  # 目标2相对之前目标位置的偏移距离的平方
            disp_threshold = self.params.dispalcement_scale * math.sqrt(sz[0] * sz[1]) / 2  # 目标相对于之前目标中心的偏移阈值
            # 说明2是难例样本，1是目标，因为目标一般位于中间，而不是边缘，此刻返回1的各种值
            if disp_norm2 > disp_threshold and disp_norm1 < disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'hard_negative'
            # 说明1是难例样本，2是目标，此刻返回2的各种值
            if disp_norm2 < disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec2, scale_ind, scores_hn, 'hard_negative'
            # 此刻1和2都距离中心点较远，出现不确定的情况，返回1的各种值
            if disp_norm2 > disp_threshold and disp_norm1 > disp_threshold:
                return translation_vec1, scale_ind, scores_hn, 'uncertain'

            # If also the distractor is close, return with highest score
            # 如果1和2都距离中心点很近，也是不确定的情况，返回分数更高的1
            return translation_vec1, scale_ind, scores_hn, 'uncertain'

        # 不是干扰目标，但是大于hard_negative的阈值，但出现难例情况，返回分数高的1
        if max_score2 > self.params.hard_negative_threshold * max_score1 and max_score2 > self.params.target_not_found_threshold:
            return translation_vec1, scale_ind, scores_hn, 'hard_negative'

        return translation_vec1, scale_ind, scores_hn, 'normal'

    def extract_backbone_features(self, im: torch.Tensor, pos: torch.Tensor, scales, sz: torch.Tensor):
        im_patches, patch_coords = sample_patch_multiscale(im, pos, scales, sz,
                                                           mode=self.params.get('border_mode', 'replicate'),
                                                           max_scale_change=self.params.get('patch_max_scale_change', None))
        with torch.no_grad():
            backbone_feat = self.net.extract_backbone(im_patches)
        return backbone_feat, patch_coords, im_patches

    def get_classification_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.extract_classification_feat(backbone_feat)

    def get_iou_backbone_features(self, backbone_feat):
        return self.net.get_backbone_bbreg_feat(backbone_feat)

    def get_iou_features(self, backbone_feat):
        with torch.no_grad():
            return self.net.bb_regressor.get_iou_feat(self.get_iou_backbone_features(backbone_feat))

    def get_iou_modulation(self, iou_backbone_feat, target_boxes):
        with torch.no_grad():
            return self.net.bb_regressor.get_modulation(iou_backbone_feat, target_boxes)


    def generate_init_samples(self, im: torch.Tensor) -> TensorList:
        """Perform data augmentation to generate initial training samples."""

        mode = self.params.get('border_mode', 'replicate')
        if mode == 'inside':
            # Get new sample size if forced inside the image
            im_sz = torch.Tensor([im.shape[2], im.shape[3]])
            sample_sz = self.target_scale * self.img_sample_sz
            shrink_factor = (sample_sz.float() / im_sz)
            if mode == 'inside':
                shrink_factor = shrink_factor.max()
            elif mode == 'inside_major':
                shrink_factor = shrink_factor.min()
            shrink_factor.clamp_(min=1, max=self.params.get('patch_max_scale_change', None))
            sample_sz = (sample_sz.float() / shrink_factor)
            self.init_sample_scale = (sample_sz / self.img_sample_sz).prod().sqrt()
            tl = self.pos - (sample_sz - 1) / 2
            br = self.pos + sample_sz / 2 + 1
            global_shift = - ((-tl).clamp(0) - (br - im_sz).clamp(0)) / self.init_sample_scale
        else:
            self.init_sample_scale = self.target_scale  # 这个样本比例是为了得到patches时即将init_sample_scale乘以sample_sz再去采样
            global_shift = torch.zeros(2)

        self.init_sample_pos = self.pos.round()

        # Compute augmentation size
        aug_expansion_factor = self.params.get('augmentation_expansion_factor', None)
        aug_expansion_sz = self.img_sample_sz.clone()
        aug_output_sz = None  # 对每种扩充方法指定输出大小
        if aug_expansion_factor is not None and aug_expansion_factor != 1:
            aug_expansion_sz = (self.img_sample_sz * aug_expansion_factor).long()
            aug_expansion_sz += (aug_expansion_sz - self.img_sample_sz.long()) % 2
            aug_expansion_sz = aug_expansion_sz.float()
            aug_output_sz = self.img_sample_sz.long().tolist()

        # Random shift for each sample
        get_rand_shift = lambda: None
        random_shift_factor = self.params.get('random_shift_factor', 0)
        if random_shift_factor > 0:
            get_rand_shift = lambda: ((torch.rand(2) - 0.5) * self.img_sample_sz * random_shift_factor + global_shift).long().tolist()  # 没有入参

        # Always put identity transformation first, since it is the unaugmented sample that is always used
        self.transforms = [augmentation.Identity(aug_output_sz, global_shift.long().tolist())]  # 恒等变换

        augs = self.params.augmentation if self.params.get('use_augmentation', True) else {}

        # Add all augmentations
        if 'shift' in augs:
            self.transforms.extend([augmentation.Translation(shift, aug_output_sz, global_shift.long().tolist()) for shift in augs['shift']])
        if 'relativeshift' in augs:  # [(0.6, 0.6), (-0.6, 0.6), (0.6, -0.6), (-0.6,-0.6)] 共4个
            get_absolute = lambda shift: (torch.Tensor(shift) * self.img_sample_sz/2).long().tolist()  # 将相对偏移转换为绝对偏移
            self.transforms.extend([augmentation.Translation(get_absolute(shift), aug_output_sz, global_shift.long().tolist()) for shift in augs['relativeshift']])
        if 'fliplr' in augs and augs['fliplr']:  # 1个
            self.transforms.append(augmentation.FlipHorizontal(aug_output_sz, get_rand_shift()))
        if 'blur' in augs:  # [(3,1), (1, 3), (2, 2)]  3个
            self.transforms.extend([augmentation.Blur(sigma, aug_output_sz, get_rand_shift()) for sigma in augs['blur']])
        if 'scale' in augs:
            self.transforms.extend([augmentation.Scale(scale_factor, aug_output_sz, get_rand_shift()) for scale_factor in augs['scale']])
        if 'rotate' in augs:  # [10, -10, 45, -45] 4个
            self.transforms.extend([augmentation.Rotate(angle, aug_output_sz, get_rand_shift()) for angle in augs['rotate']])

        # Extract augmented image patches  开始采样出来的patches大小是aug_expansion_sz 经过transforms才变为aug_output_sz
        im_patches = sample_patch_transformed(im, self.init_sample_pos, self.init_sample_scale, aug_expansion_sz, self.transforms)

        # Extract initial backbone features
        # with torch.no_grad()中的数据不需要计算梯度，也不需要进行反向传播
        with torch.no_grad():
            init_backbone_feat = self.net.extract_backbone(im_patches)

        return init_backbone_feat

    def init_target_boxes(self):
        """Get the target bounding boxes for the initial augmented samples."""
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        init_target_boxes = TensorList()  # xywh
        for T in self.transforms:  # T.shift:hw
            init_target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))  # 偏移
        init_target_boxes = torch.cat(init_target_boxes.view(1, 4), 0).to(self.params.device)
        self.target_boxes = init_target_boxes.new_zeros(self.params.sample_memory_size, 4)  # 先初始化sample_memory_size个全0的target_boxes
        self.target_boxes[:init_target_boxes.shape[0],:] = init_target_boxes # 赋值
        return init_target_boxes

    def init_memory(self, train_x: TensorList):
        # Initialize first-frame spatial training samples  # 初始化第一帧的空间训练样本，train_x是提取的分类特征
        self.num_init_samples = train_x.size(0)  # 初始化样本数量，为15
        init_sample_weights = TensorList([x.new_ones(1) / x.shape[0] for x in train_x])  # 初始化样本权重为1/15

        # Sample counters and weights for spatial
        self.num_stored_samples = self.num_init_samples.copy()  # 存储的样本数量
        self.previous_replace_ind = [None] * len(self.num_stored_samples)
        self.sample_weights = TensorList([x.new_zeros(self.params.sample_memory_size) for x in train_x])  # 初始化全0的50个样本权重
        for sw, init_sw, num in zip(self.sample_weights, init_sample_weights, self.num_init_samples):
            sw[:num] = init_sw  # 将15个初始化的样本权重赋值

        # Initialize memory
        self.training_samples = TensorList(
            [x.new_zeros(self.params.sample_memory_size, x.shape[1], x.shape[2], x.shape[3]) for x in train_x])

        for ts, x in zip(self.training_samples, train_x):
            ts[:x.shape[0],...] = x


    def update_memory(self, sample_x: TensorList, target_box, learning_rate = None):
        # Update weights and get replace ind
        replace_ind = self.update_sample_weights(self.sample_weights, self.previous_replace_ind, self.num_stored_samples, self.num_init_samples, learning_rate)
        self.previous_replace_ind = replace_ind

        # Update sample and label memory
        for train_samp, x, ind in zip(self.training_samples, sample_x, replace_ind):
            train_samp[ind:ind+1,...] = x

        # Update bb memory
        self.target_boxes[replace_ind[0],:] = target_box

        self.num_stored_samples += 1


    def update_sample_weights(self, sample_weights, previous_replace_ind, num_stored_samples, num_init_samples, learning_rate = None):
        # Update weights and get index to replace
        replace_ind = []
        for sw, prev_ind, num_samp, num_init in zip(sample_weights, previous_replace_ind, num_stored_samples, num_init_samples):
            lr = learning_rate
            if lr is None:
                lr = self.params.learning_rate

            init_samp_weight = self.params.get('init_samples_minimum_weight', None)
            if init_samp_weight == 0:
                init_samp_weight = None
            s_ind = 0 if init_samp_weight is None else num_init  # s_ind是开始替换的索引 当样本权重没有指定大小或者为0,s_ind为0，从头替换样本

            if num_samp == 0 or lr == 1:  # 没有样本或者学习率为1时，会清除memory
                sw[:] = 0
                sw[0] = 1
                r_ind = 0
            else:
                # Get index to replace   r_ind是替代的索引，下面的if else 是为了确定r_ind
                if num_samp < sw.shape[0]:
                    r_ind = num_samp
                else:
                    _, r_ind = torch.min(sw[s_ind:], 0)  # 从init_samples后找到权重最小的样本索引
                    r_ind = r_ind.item() + s_ind

                # Update weights
                if prev_ind is None:
                    sw /= 1 - lr  # 所有样本权重增加
                    sw[r_ind] = lr
                else:
                    sw[r_ind] = sw[prev_ind] / (1 - lr)  # 当前要替换的样本权重，比上一次大，跟踪正常情况下跟踪到的权重越来也高

            sw /= sw.sum()
            if init_samp_weight is not None and sw[:num_init].sum() < init_samp_weight:
                sw /= init_samp_weight + sw[num_init:].sum()
                sw[:num_init] = init_samp_weight / num_init

            replace_ind.append(r_ind)

        return replace_ind

    def update_state(self, new_pos, new_scale = None):
        # Update scale
        if new_scale is not None:
            self.target_scale = new_scale.clamp(self.min_scale_factor, self.max_scale_factor)
            self.target_sz = self.base_target_sz * self.target_scale

        # Update pos 没看懂这部分操作
        inside_ratio = self.params.get('target_inside_ratio', 0.2)
        inside_offset = (inside_ratio - 0.5) * self.target_sz
        self.pos = torch.max(torch.min(new_pos, self.image_sz - inside_offset), inside_offset)


    def get_iounet_box(self, pos, sz, sample_pos, sample_scale):
        """All inputs in original image coordinates.
        Generates a box in the cropped image sample reference frame, in the format used by the IoUNet.  format是xywh 入参pos是yx"""
        box_center = (pos - sample_pos) / sample_scale + (self.img_sample_sz - 1) / 2  # 将bb的中心点坐标从原图片的坐标系转换到采样后图片的坐标系下
        box_sz = sz / sample_scale  # 原图中bb大小在im_patches中对应的大小
        target_ul = box_center - (box_sz - 1) / 2    # 左上点坐标
        return torch.cat([target_ul.flip((0,)), box_sz.flip((0,))])


    def init_iou_net(self, backbone_feat):
        # Setup IoU net and objective
        for p in self.net.bb_regressor.parameters():
            p.requires_grad = False  # iou_net使用的是训练好的权重，不需要计算梯度

        # Get target boxes for the different augmentations
        self.classifier_target_box = self.get_iounet_box(self.pos, self.target_sz, self.init_sample_pos, self.init_sample_scale)
        target_boxes = TensorList()  # iounet使用的target_boxes，根据下面的iounet_augmentation的不同，可能不同于dimp里self.target_boxes
        if self.params.iounet_augmentation:
            for T in self.transforms:
                if not isinstance(T, (augmentation.Identity, augmentation.Translation, augmentation.FlipHorizontal, augmentation.FlipVertical, augmentation.Blur)):
                    break
                target_boxes.append(self.classifier_target_box + torch.Tensor([T.shift[1], T.shift[0], 0, 0]))
        else:
            target_boxes.append(self.classifier_target_box + torch.Tensor([self.transforms[0].shift[1], self.transforms[0].shift[0], 0, 0]))
        target_boxes = torch.cat(target_boxes.view(1,4), 0).to(self.params.device)

        # Get iou features
        iou_backbone_feat = self.get_iou_backbone_features(backbone_feat)  # 是一个列表，第一个是layer2,第二个是layer3

        # Remove other augmentations such as rotation  x是使用增强的样本提取的特征，这里需要移除增强样本的特征
        iou_backbone_feat = TensorList([x[:target_boxes.shape[0],...] for x in iou_backbone_feat])

        # Get modulation vector
        self.iou_modulation = self.get_iou_modulation(iou_backbone_feat, target_boxes)
        if torch.is_tensor(self.iou_modulation[0]):
            self.iou_modulation = TensorList([x.detach().mean(0) for x in self.iou_modulation]) # 对两个modulation在第0维上求平均


    def init_classifier(self, init_backbone_feat):
        # Get classification features
        x = self.get_classification_features(init_backbone_feat)

        # Overwrite some parameters in the classifier. (These are not generally changed)
        self._overwrite_classifier_params(feature_dim=x.shape[-3])

        # Add the dropout augmentation here, since it requires extraction of the classification features
        if 'dropout' in self.params.augmentation and self.params.get('use_augmentation', True):
            num, prob = self.params.augmentation['dropout']  # dropout augmentation的个数和神经网络节点dropout的概率
            self.transforms.extend(self.transforms[:1]*num)  # transforms中增加num个恒等变换
            x = torch.cat([x, F.dropout2d(x[0:1,...].expand(num,-1,-1,-1), p=prob, training=True)])  # 将num个x[0]dropout后连接到x中

        # Set feature size and other related sizes
        self.feature_sz = torch.Tensor(list(x.shape[-2:]))  # 用于classifier的特征的大小
        ksz = self.net.classifier.filter_size        # classifier滤波器大小
        self.kernel_size = torch.Tensor([ksz, ksz] if isinstance(ksz, (int, float)) else ksz)
        self.output_sz = self.feature_sz + (self.kernel_size + 1)%2  # 输出特征大小

        # Construct output window
        self.output_window = None
        if self.params.get('window_output', False):
            if self.params.get('use_clipped_window', False):
                self.output_window = dcf.hann2d_clipped(self.output_sz.long(), (self.output_sz*self.params.effective_search_area / self.params.search_area_scale).long(), centered=True).to(self.params.device)
            else:
                self.output_window = dcf.hann2d(self.output_sz.long(), centered=True).to(self.params.device)
            self.output_window = self.output_window.squeeze(0)

        # Get target boxes for the different augmentations
        target_boxes = self.init_target_boxes()

        # Set number of iterations
        plot_loss = self.params.debug > 0
        num_iter = self.params.get('net_opt_iter', None)

        # Get target filter by running the discriminative model prediction module
        with torch.no_grad():
            self.target_filter, _, losses = self.net.classifier.get_filter(x, target_boxes, num_iter=num_iter,
                                                                           compute_losses=plot_loss)

        # Init memory  这里的memory指代的应该是self.train_samples大小为50，同时每个样本的权重存储在self.sample_weights中，用于分类器的优化模块和memory中的样本替换
        if self.params.get('update_classifier', True):
            self.init_memory(TensorList([x]))

        if plot_loss:
            if isinstance(losses, dict):
                losses = losses['train']
            self.losses = torch.cat(losses)
            if self.visdom is not None:
                self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
            elif self.params.debug >= 3:
                plot_graph(self.losses, 10, title='Training Loss' + self.id_str)

    def _overwrite_classifier_params(self, feature_dim):
        # Overwrite some parameters in the classifier. (These are not generally changed)
        pred_module = getattr(self.net.classifier.filter_optimizer, 'score_predictor', self.net.classifier.filter_optimizer)
        if self.params.get('label_threshold', None) is not None:
            self.net.classifier.filter_optimizer.label_threshold = self.params.label_threshold
        if self.params.get('label_shrink', None) is not None:
            self.net.classifier.filter_optimizer.label_shrink = self.params.label_shrink
        if self.params.get('softmax_reg', None) is not None:
            self.net.classifier.filter_optimizer.softmax_reg = self.params.softmax_reg
        if self.params.get('filter_reg', None) is not None:
            pred_module.filter_reg[0] = self.params.filter_reg
            pred_module.min_filter_reg = self.params.filter_reg
        if self.params.get('filter_init_zero', False):
            self.net.classifier.filter_initializer = FilterInitializerZero(self.net.classifier.filter_size, feature_dim)


    def update_classifier(self, train_x, target_box, learning_rate=None, scores=None):
        # Set flags and learning rate
        hard_negative_flag = learning_rate is not None
        if learning_rate is None:
            learning_rate = self.params.learning_rate

        # Update the tracker memory
        if hard_negative_flag or self.frame_num % self.params.get('train_sample_interval', 1) == 0:  # params没这个属性，值为1,每次都更新memory
            self.update_memory(TensorList([train_x]), target_box, learning_rate)

        # Decide the number of iterations to run
        num_iter = 0  # 根据三种情况确定分类器迭代优化的次数,hard_neg, low_score, normal
        low_score_th = self.params.get('low_score_opt_threshold', None)   # dimp50参数里没设置
        if hard_negative_flag:
            num_iter = self.params.get('net_opt_hn_iter', None)
        elif low_score_th is not None and low_score_th > scores.max().item():
            num_iter = self.params.get('net_opt_low_iter', None)  # dimp50参数里没设置
        elif (self.frame_num - 1) % self.params.train_skipping == 0:  # 除去初始帧，跟踪正常时20帧优化一次分类器
            num_iter = self.params.get('net_opt_update_iter', None)

        plot_loss = self.params.debug > 0

        if num_iter > 0:
            # Get inputs for the DiMP filter optimizer module  会使用memory里存储的所有样本优化分类器
            samples = self.training_samples[0][:self.num_stored_samples[0],...]
            target_boxes = self.target_boxes[:self.num_stored_samples[0],:].clone()
            sample_weights = self.sample_weights[0][:self.num_stored_samples[0]]

            # Run the filter optimizer module
            with torch.no_grad():
                self.target_filter, _, losses = self.net.classifier.filter_optimizer(self.target_filter,
                                                                                     num_iter=num_iter, feat=samples,
                                                                                     bb=target_boxes,
                                                                                     sample_weight=sample_weights,
                                                                                     compute_losses=plot_loss)

            if plot_loss:
                if isinstance(losses, dict):
                    losses = losses['train']
                self.losses = torch.cat((self.losses, torch.cat(losses)))
                if self.visdom is not None:
                    self.visdom.register((self.losses, torch.arange(self.losses.numel())), 'lineplot', 3, 'Training Loss' + self.id_str)
                elif self.params.debug >= 3:
                    plot_graph(self.losses, 10, title='Training Loss' + self.id_str)

    def refine_target_box(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
        """Run the ATOM IoUNet to refine the target bounding box."""

        if hasattr(self.net.bb_regressor, 'predict_bb'):  # 没有这个属性
            return self.direct_box_regression(backbone_feat, sample_pos, sample_scale, scale_ind, update_scale)

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale  提取用于iounet的特征
        iou_features = self.get_iou_features(backbone_feat)
        iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])  # 每个feat中第0维的个数代表im_patche个数，与scales个数对应

        # Generate random initial boxes  生成9个额外的随机bbox
        init_boxes = init_box.view(1,4).clone()
        if self.params.num_init_random_boxes > 0:
            square_box_sz = init_box[2:].prod().sqrt()
            rand_factor = square_box_sz * torch.cat([self.params.box_jitter_pos * torch.ones(2), self.params.box_jitter_sz * torch.ones(2)])  # 相当于中心位置和大小的偏移量因子

            minimal_edge_size = init_box[2:].min()/3  # 最小的边长
            rand_bb = (torch.rand(self.params.num_init_random_boxes, 4) - 0.5) * rand_factor  # 中心位置和大小的偏移量
            new_sz = (init_box[2:] + rand_bb[:,2:]).clamp(minimal_edge_size)
            new_center = (init_box[:2] + init_box[2:]/2) + rand_bb[:,:2]
            init_boxes = torch.cat([new_center - new_sz/2, new_sz], 1)
            init_boxes = torch.cat([init_box.view(1,4), init_boxes])

        # Optimize the boxes  对init_boxes进行梯度下降优化
        output_boxes, output_iou = self.optimize_boxes(iou_features, init_boxes)

        # Remove weird boxes  移除长宽比不正常的boxes
        output_boxes[:, 2:].clamp_(1)  # boxes的高和宽最小为1
        aspect_ratio = output_boxes[:,2] / output_boxes[:,3]  # 长宽比 output_box格式是xywh  w是长，h是宽
        keep_ind = (aspect_ratio < self.params.maximal_aspect_ratio) * (aspect_ratio > 1/self.params.maximal_aspect_ratio)  # 图片的长宽比要小于6同时大于1/6
        output_boxes = output_boxes[keep_ind,:]
        output_iou = output_iou[keep_ind]

        # If no box found
        if output_boxes.shape[0] == 0:
            return

        # Predict box
        k = self.params.get('iounet_k', 5)
        topk = min(k, output_boxes.shape[0])
        _, inds = torch.topk(output_iou, topk)  # 降序排列后的前k个大小的元素值及原索引
        predicted_box = output_boxes[inds, :].mean(0)
        predicted_iou = output_iou.view(-1, 1)[inds, :].mean(0)

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos  # new_pos是在原图尺度下目标的中心坐标
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())

        self.pos_iounet = new_pos.clone()

        if self.params.get('use_iounet_pos_for_learning', True):  # params里没有这个属性，默认更新
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale

        # self.visualize_iou_pred(iou_features, predicted_box)


    def optimize_boxes(self, iou_features, init_boxes):
        box_refinement_space = self.params.get('box_refinement_space', 'default')
        if box_refinement_space == 'default':
            return self.optimize_boxes_default(iou_features, init_boxes)
        if box_refinement_space == 'relative':
            return self.optimize_boxes_relative(iou_features, init_boxes)
        raise ValueError('Unknown box_refinement_space {}'.format(box_refinement_space))


    def optimize_boxes_default(self, iou_features, init_boxes):
        """Optimize iounet boxes with the default parametrization"""
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]], device=self.params.device).view(1,1,4)
        # The predicted IoU of each box is maximized using 5 gradient ascent iterations with a step length of 1.
        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init = output_boxes.clone().detach()  # 返回的tensor和原tensor在梯度上或者数据上没有任何关系，即bb_init和output_boxes没有联系
            bb_init.requires_grad = True
            # 输出每个box的预测iou
            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient = torch.ones_like(outputs))  # 反向传播，可以计算出bb_init的梯度

            # Update proposal
            output_boxes = bb_init + step_length * bb_init.grad * bb_init[:, :, 2:].repeat(1, 1, 2) #这里将原bb_init的wh复制一份得到whwh,对于左上角坐标为什么不是bb_init.grad[0:2] * bb_init[0:2],而是bb_init.grad[0:2] * wh
            output_boxes.detach_()  # 从计算图中分离出来，与bb_init没有任何关系了

            step_length *= self.params.box_refinement_step_decay

        return output_boxes.view(-1,4).cpu(), outputs.detach().view(-1).cpu()


    def optimize_boxes_relative(self, iou_features, init_boxes):
        """Optimize iounet boxes with the relative parametrization ised in PrDiMP"""
        output_boxes = init_boxes.view(1, -1, 4).to(self.params.device)
        step_length = self.params.box_refinement_step_length
        if isinstance(step_length, (tuple, list)):
            step_length = torch.Tensor([step_length[0], step_length[0], step_length[1], step_length[1]]).to(self.params.device).view(1,1,4)

        sz_norm = output_boxes[:,:1,2:].clone()
        output_boxes_rel = bbutils.rect_to_rel(output_boxes, sz_norm)
        for i_ in range(self.params.box_refinement_iter):
            # forward pass
            bb_init_rel = output_boxes_rel.clone().detach()
            bb_init_rel.requires_grad = True

            bb_init = bbutils.rel_to_rect(bb_init_rel, sz_norm)
            outputs = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, bb_init)

            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]

            outputs.backward(gradient = torch.ones_like(outputs))

            # Update proposal
            output_boxes_rel = bb_init_rel + step_length * bb_init_rel.grad
            output_boxes_rel.detach_()

            step_length *= self.params.box_refinement_step_decay

        #     for s in outputs.view(-1):
        #         print('{:.2f}  '.format(s.item()), end='')
        #     print('')
        # print('')

        output_boxes = bbutils.rel_to_rect(output_boxes_rel, sz_norm)

        return output_boxes.view(-1,4).cpu(), outputs.detach().view(-1).cpu()

    def direct_box_regression(self, backbone_feat, sample_pos, sample_scale, scale_ind, update_scale = True):
        """Implementation of direct bounding box regression.  这个没用到 bb_regressor里没有predict_bb函数"""

        # Initial box for refinement
        init_box = self.get_iounet_box(self.pos, self.target_sz, sample_pos, sample_scale)

        # Extract features from the relevant scale
        iou_features = self.get_iou_features(backbone_feat)  # backone_feat经过两个连续的卷积层后的feat
        iou_features = TensorList([x[scale_ind:scale_ind+1,...] for x in iou_features])

        # Generate random initial boxes
        init_boxes = init_box.view(1, 1, 4).clone().to(self.params.device)

        # Optimize the boxes
        output_boxes = self.net.bb_regressor.predict_bb(self.iou_modulation, iou_features, init_boxes).view(-1,4).cpu()

        # Remove weird boxes
        output_boxes[:, 2:].clamp_(1)

        predicted_box = output_boxes[0, :]

        # Get new position and size
        new_pos = predicted_box[:2] + predicted_box[2:] / 2
        new_pos = (new_pos.flip((0,)) - (self.img_sample_sz - 1) / 2) * sample_scale + sample_pos
        new_target_sz = predicted_box[2:].flip((0,)) * sample_scale
        new_scale_bbr = torch.sqrt(new_target_sz.prod() / self.base_target_sz.prod())
        new_scale = new_scale_bbr

        self.pos_iounet = new_pos.clone()

        if self.params.get('use_iounet_pos_for_learning', True):
            self.pos = new_pos.clone()

        self.target_sz = new_target_sz

        if update_scale:
            self.target_scale = new_scale


    def visualize_iou_pred(self, iou_features, center_box):
        center_box = center_box.view(1,1,4)
        sz_norm = center_box[...,2:].clone()
        center_box_rel = bbutils.rect_to_rel(center_box, sz_norm)

        pos_dist = 1.0
        sz_dist = math.log(3.0)
        pos_step = 0.01
        sz_step = 0.01

        pos_scale = torch.arange(-pos_dist, pos_dist+pos_step, step=pos_step)
        sz_scale = torch.arange(-sz_dist, sz_dist+sz_step, step=sz_step)

        bbx = torch.zeros(1, pos_scale.numel(), 4)
        bbx[0,:,0] = pos_scale.clone()
        bby = torch.zeros(pos_scale.numel(), 1, 4)
        bby[:,0,1] = pos_scale.clone()
        bbw = torch.zeros(1, sz_scale.numel(), 4)
        bbw[0,:,2] = sz_scale.clone()
        bbh = torch.zeros(sz_scale.numel(), 1, 4)
        bbh[:,0,3] = sz_scale.clone()

        pos_boxes = bbutils.rel_to_rect((center_box_rel + bbx) + bby, sz_norm).view(1,-1,4).to(self.params.device)
        sz_boxes = bbutils.rel_to_rect((center_box_rel + bbw) + bbh, sz_norm).view(1,-1,4).to(self.params.device)

        pos_scores = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, pos_boxes).exp()
        sz_scores = self.net.bb_regressor.predict_iou(self.iou_modulation, iou_features, sz_boxes).exp()

        show_tensor(pos_scores.view(pos_scale.numel(),-1), title='Position scores', fig_num=21)
        show_tensor(sz_scores.view(sz_scale.numel(),-1), title='Size scores', fig_num=22)


    def visdom_draw_tracking(self, image, box, segmentation=None):
        if hasattr(self, 'search_area_box'):
            self.visdom.register((image, box, self.search_area_box), 'Tracking', 1, 'Tracking')
        else:
            self.visdom.register((image, box), 'Tracking', 1, 'Tracking')