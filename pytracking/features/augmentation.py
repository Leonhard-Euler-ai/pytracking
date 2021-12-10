import numpy as np
import math
import torch
import torch.nn.functional as F
import cv2 as cv
import random
from pytracking.features.preprocessing import numpy_to_torch, torch_to_numpy


class Transform:
    """Base data augmentation transform class."""

    def __init__(self, output_sz = None, shift = None):
        self.output_sz = output_sz
        self.shift = (0,0) if shift is None else shift

    def __call__(self, image, is_mask=False):
        raise NotImplementedError

    def crop_to_output(self, image):
        if isinstance(image, torch.Tensor):
            imsz = image.shape[2:]
            if self.output_sz is None:
                pad_h = 0
                pad_w = 0
            else:
                pad_h = (self.output_sz[0] - imsz[0]) / 2
                pad_w = (self.output_sz[1] - imsz[1]) / 2

            # 取整：floor把数字变小，ceil把数字变大
            # 计算四周填充的次数，使填充后的图片大小为output_sz  shift[0]是高度方向上的偏移，shift[1]是宽度方向上的偏移
            pad_left = math.floor(pad_w) + self.shift[1]
            pad_right = math.ceil(pad_w) - self.shift[1]
            pad_top = math.floor(pad_h) + self.shift[0]
            pad_bottom = math.ceil(pad_h) - self.shift[0]
            # 得到output_sz大小的图像
            return F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), 'replicate')
        else:
            raise NotImplementedError

class Identity(Transform):
    """Identity transformation. 恒等变换"""
    def __call__(self, image, is_mask=False):
        return self.crop_to_output(image)

class FlipHorizontal(Transform):
    """Flip along horizontal axis. 水平翻转"""
    def __call__(self, image, is_mask=False):
        if isinstance(image, torch.Tensor):
            return self.crop_to_output(image.flip((3,)))  # image的维度是(batch,channels,h,w),水平翻转在第3维度上
        else:
            return np.fliplr(image)  # 传入的image是numpy处理的ndarray类型的，使用numpy翻转

class FlipVertical(Transform):
    """Flip along vertical axis."""
    def __call__(self, image: torch.Tensor, is_mask=False):
        if isinstance(image, torch.Tensor):
            return self.crop_to_output(image.flip((2,)))
        else:
            return np.flipud(image)

class Translation(Transform):
    """Translate. 平移"""
    def __init__(self, translation, output_sz = None, shift = None):
        super().__init__(output_sz, shift)
        # 由于父类采用的是复制模式的填充，因此通过translation控制填充来实现平移  translation[0]表示上下平移量，translation[1]表示左右平移量，因为有输出大小的限制，并不是数值是几就平移几
        self.shift = (self.shift[0] + translation[0], self.shift[1] + translation[1])

    def __call__(self, image, is_mask=False):
        if isinstance(image, torch.Tensor):
            return self.crop_to_output(image)
        else:
            raise NotImplementedError

class Scale(Transform):
    """Scale. 缩放"""
    def __init__(self, scale_factor, output_sz = None, shift = None):
        super().__init__(output_sz, shift)
        self.scale_factor = scale_factor

    def __call__(self, image, is_mask=False):
        if isinstance(image, torch.Tensor):
            # Calculate new size. Ensure that it is even so that crop/pad becomes easier  确保原图长宽是偶数
            h_orig, w_orig = image.shape[2:]

            if h_orig != w_orig:  # 原图长宽不相等的缩放代码没实现
                raise NotImplementedError
            # 新的长宽保证也是偶数
            h_new = round(h_orig /self.scale_factor)
            h_new += (h_new - h_orig) % 2
            w_new = round(w_orig /self.scale_factor)
            w_new += (w_new - w_orig) % 2
            # 先采用二次线性插值将原图插值到缩放后的大小
            image_resized = F.interpolate(image, [h_new, w_new], mode='bilinear')

            return self.crop_to_output(image_resized)
        else:
            raise NotImplementedError


class Affine(Transform):
    """Affine transformation.仿射变换"""
    def __init__(self, transform_matrix, output_sz = None, shift = None):
        super().__init__(output_sz, shift)
        self.transform_matrix = transform_matrix

    def __call__(self, image, is_mask=False):
        if isinstance(image, torch.Tensor):
            return self.crop_to_output(numpy_to_torch(self(torch_to_numpy(image))))  # 先将Tensor形式的image传入可调用对象self,会调用__call__()方法，得到numpy格式的转换结果，再转为torch，crop大小
        else:
            return cv.warpAffine(image, self.transform_matrix, image.shape[1::-1], borderMode=cv.BORDER_REPLICATE)


class Rotate(Transform):
    """Rotate with given angle. 给定一个角度的旋转，单位是度"""
    def __init__(self, angle, output_sz = None, shift = None):
        super().__init__(output_sz, shift)
        self.angle = math.pi * angle/180

    def __call__(self, image, is_mask=False):
        if isinstance(image, torch.Tensor):
            return self.crop_to_output(numpy_to_torch(self(torch_to_numpy(image))))
        else:
            # c代表中心点坐标
            c = (np.expand_dims(np.array(image.shape[:2]),1)-1)/2
            R = np.array([[math.cos(self.angle), math.sin(self.angle)],
                          [-math.sin(self.angle), math.cos(self.angle)]])
            # H为旋转矩阵
            H =np.concatenate([R, c - R @ c], 1)
            return cv.warpAffine(image, H, image.shape[1::-1], borderMode=cv.BORDER_REPLICATE)


class Blur(Transform):
    """Blur with given sigma (can be axis dependent)."""
    def __init__(self, sigma, output_sz = None, shift = None):
        super().__init__(output_sz, shift)
        if isinstance(sigma, (float, int)):
            sigma = (sigma, sigma)
        self.sigma = sigma
        self.filter_size = [math.ceil(2*s) for s in self.sigma]
        x_coord = [torch.arange(-sz, sz+1, dtype=torch.float32) for sz in self.filter_size]
        self.filter = [torch.exp(-(x**2)/(2*s**2)) for x, s in zip(x_coord, self.sigma)]
        self.filter[0] = self.filter[0].view(1,1,-1,1) / self.filter[0].sum()   # 垂直方向上高斯滤波
        self.filter[1] = self.filter[1].view(1,1,1,-1) / self.filter[1].sum()   # 水平方向上高斯滤波

    def __call__(self, image, is_mask=False):
        if isinstance(image, torch.Tensor):
            sz = image.shape[2:]
            im1 = F.conv2d(image.view(-1,1,sz[0],sz[1]), self.filter[0], padding=(self.filter_size[0],0))
            return self.crop_to_output(F.conv2d(im1, self.filter[1], padding=(0,self.filter_size[1])).view(1,-1,sz[0],sz[1]))
        else:
            raise NotImplementedError


class RandomAffine(Transform):
    """Affine transformation. 随机仿射变换（包括旋转，裁减，缩放等）"""
    def __init__(self, p_flip=0.0, max_rotation=0.0, max_shear=0.0, max_scale=0.0, max_ar_factor=0.0,
                 border_mode='constant', output_sz = None, shift = None):
        super().__init__(output_sz, shift)
        self.p_flip = p_flip
        self.max_rotation = max_rotation
        self.max_shear = max_shear
        self.max_scale = max_scale
        self.max_ar_factor = max_ar_factor

        self.pad_amount = 0
        if border_mode == 'constant':
            self.border_flag = cv.BORDER_CONSTANT
        elif border_mode == 'replicate':
            self.border_flag == cv.BORDER_REPLICATE
        else:
            raise Exception

        self.roll_values = self.roll()

    def roll(self):
        do_flip = random.random() < self.p_flip
        theta = random.uniform(-self.max_rotation, self.max_rotation)

        shear_x = random.uniform(-self.max_shear, self.max_shear)
        shear_y = random.uniform(-self.max_shear, self.max_shear)

        ar_factor = np.exp(random.uniform(-self.max_ar_factor, self.max_ar_factor))
        scale_factor = np.exp(random.uniform(-self.max_scale, self.max_scale))

        return do_flip, theta, (shear_x, shear_y), (scale_factor, scale_factor * ar_factor)

    def _construct_t_mat(self, image_shape, do_flip, theta, shear_values, scale_factors):
        im_h, im_w = image_shape
        t_mat = np.identity(3)

        if do_flip:
            if do_flip:
                t_mat[0, 0] = -1.0
                t_mat[0, 2] = im_w

        t_rot = cv.getRotationMatrix2D((im_w * 0.5, im_h * 0.5), theta, 1.0)
        t_rot = np.concatenate((t_rot, np.array([0.0, 0.0, 1.0]).reshape(1, 3)))

        t_shear = np.array([[1.0, shear_values[0], -shear_values[0] * 0.5 * im_w],
                            [shear_values[1], 1.0, -shear_values[1] * 0.5 * im_h],
                            [0.0, 0.0, 1.0]])

        t_scale = np.array([[scale_factors[0], 0.0, (1.0 - scale_factors[0]) * 0.5 * im_w],
                            [0.0, scale_factors[1], (1.0 - scale_factors[1]) * 0.5 * im_h],
                            [0.0, 0.0, 1.0]])

        t_mat = t_scale @ t_rot @ t_shear @ t_mat

        t_mat[0, 2] += self.pad_amount
        t_mat[1, 2] += self.pad_amount

        t_mat = t_mat[:2, :]

        return t_mat

    def __call__(self, image, is_mask=False):
        input_tensor = torch.is_tensor(image)
        if input_tensor:
            image = torch_to_numpy(image)

        do_flip, theta, shear_values, scale_factors = self.roll_values
        t_mat = self._construct_t_mat(image.shape[:2], do_flip, theta, shear_values, scale_factors)
        output_sz = (image.shape[1] + 2*self.pad_amount, image.shape[0] + 2*self.pad_amount)

        if not is_mask:
            image_t = cv.warpAffine(image, t_mat, output_sz, flags=cv.INTER_LINEAR,
                                    borderMode=self.border_flag)
        else:
            image_t = cv.warpAffine(image, t_mat, output_sz, flags=cv.INTER_NEAREST,
                                    borderMode=self.border_flag)
            image_t = image_t.reshape(image.shape)

        if input_tensor:
            image_t = numpy_to_torch(image_t)

        return self.crop_to_output(image_t)
