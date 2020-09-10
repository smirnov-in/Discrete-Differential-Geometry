import math
import matplotlib.image as img
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import interpolation


class Image:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def read(file):
        return Image(img.imread(file))

    def show(self):
        plt.imshow(self.data, interpolation='nearest')
        plt.axis('off')
        plt.show()

    def __mul__(self, other):
        return Image(self.data * other.as_array())

    def zoom(self, zoom):
        self.data = interpolation.zoom(self.data, zoom + (1,), order=0)

    def resize(self, size):
        self.zoom((size[0] / self.data.shape[0], size[1] / self.data.shape[1]))


class Color:
    def __init__(self, color, opacity=1.):
        self.color = np.array(color)
        self.opacity = opacity

    @staticmethod
    def white(brightness=1., opacity=1.):
        return Color(brightness * np.array([1., 1., 1.]), opacity)

    @staticmethod
    def black(brightness=1., opacity=1.):
        return Color(brightness * np.array([0., 0., 0.]), opacity)

    @staticmethod
    def red(brightness=1., opacity=1.):
        return Color(brightness * np.array([1., 0., 0.]), opacity)

    @staticmethod
    def green(brightness=1., opacity=1.):
        return Color(brightness * np.array([0., 1., 0.]), opacity)

    @staticmethod
    def blue(brightness=1., opacity=1.):
        return Color(brightness * np.array([0., 0., 1.]), opacity)

    @staticmethod
    def transparent():
        return Color(np.array([0., 0., 0.]), 0)

    @property
    def transmittance(self):
        return 1 - self.opacity

    @transmittance.setter
    def transmittance(self, transmittance):
        self.opacity = 1 - transmittance

    def as_array(self):
        return np.concatenate((self.color, [self.opacity]))

    @staticmethod
    def from_array(array):
        return Color(array[-1] * array[:-1], array[-1])

    def __repr__(self):
        return str((self.color, self.opacity))

    def __rmul__(self, other):
        return Color(other * self.color, other * self.opacity)

    def __add__(self, other):
        return Color(self.color + other.color, self.opacity + other.opacity)

    def over(self, other):
        return self + self.transmittance * other

    def merge(self, other):
        assert self.opacity != 1 or other.opacity != 1
        
	if self.opacity == 0 and other.opacity == 0:
            return self + other
        elif self.opacity == 1:
            return self
        elif other.opacity == 1:
            return other
        elif self.opacity == 0:
            return (other.opacity / math.log(other.transmittance)) * self + other
        elif other.opacity == 0:
            return self + (self.opacity / math.log(self.transmittance)) * other
        else:
            return (1 - self.transmittance * other.transmittance) / \
                   (math.log(self.transmittance) + math.log(other.transmittance)) * \
                   (math.log(self.transmittance) / self.opacity * self +
                    math.log(other.transmittance) / other.opacity * other)

    def pow(self, power):
        if self.opacity == 0:
            return power * self
        else:
            return (1 - self.transmittance ** power) / self.opacity * self


class Pixel:
    def __init__(self, color, front, back):
        self.color = color
        self.front = front
        self.back = back

    def __mul__(self, other):
        if self.back <= other.front or other.back <= self.front:
            return None
        else:
            return Pixel(self.color.merge(other.color), max(self.front, other.front), min(self.back, other.back))

    def __lshift__(self, other):
        if other is None:
            return self
        elif other.front <= self.front or self.front == self.back:
            return None
        else:
            return Pixel(self.color.pow((min(self.back, other.front) - self.front) / (self.back - self.front)), self.front, min(self.back, other.front))

    def __rshift__(self, other):
        if other is None:
            return self
        elif self.back <= other.back or self.front == self.back:
            return None
        else:
            return Pixel(self.color.pow((self.back - max(self.front, other.back)) / (self.back - self.front)), max(self.front, other.back), self.back)

    def __repr__(self):
        return str((self.color, self.front, self.back))


def merge_pixels(a, b):
    if len(a) == 0:
        return b
    
    if len(b) == 0:
        return a

    result = list()
    i, j = 0, 0
    a_current, b_current = a[0], b[0]
    
    while True:
        tmp = a_current << b_current or b_current << a_current
        
	if tmp is not None:
            a_current = a_current >> tmp
            b_current = b_current >> tmp
            
	    result.append(tmp)

        if (a_current and b_current) is not None:
            a_next = a_current >> b_current
            b_next = b_current >> a_current
            
	    a_current = a_current << a_next
            b_current = b_current << b_next

            if (a_current and b_current) is not None:
                tmp = a_current * b_current
                
		if tmp is not None:
                    result.append(tmp)

            a_current, b_current = a_next, b_next

        if a_current is None:
            i += 1
            
	    if i == len(a):
                break
            
	    a_current = a[i]

        if b_current is None:
            j += 1
            
	    if j == len(b):
                break

	    b_current = b[j]

    if a_current is not None:
        result.append(a_current)
    
    if b_current is not None:
        result.append(b_current)
    
    result += a[i + 1:] + b[j + 1:]

    return result


class DeepImage:
    def __init__(self, image, front=None, back=None):
        if type(image) is tuple:
            self.data = np.full(image, None, dtype=object)
            
	    for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                    self.data[i, j] = list()
        else:
            self.data = np.full(image.data.shape[:-1], None, dtype=object)
            for i in range(self.data.shape[0]):
                for j in range(self.data.shape[1]):
                    self.data[i, j] = [Pixel(Color.from_array(image.data[i, j]), front[i, j], back[i, j])] \
                        if image.data[i, j] is not None else list()

    def __mul__(self, other):
        result = DeepImage(self.data.shape)
        
	for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                result.data[i, j] = merge_pixels(self.data[i, j], other.data[i, j])
        
	return result

    def render(self, background=Color.white()):
        result = np.zeros(self.data.shape + (4,), dtype=float)
        
	for i in range(self.data.shape[0]):
            for j in range(self.data.shape[1]):
                tmp = background
                
		for k in range(len(self.data[i, j]), 0, -1):
                    tmp = self.data[i, j][k - 1].color.over(tmp)
                
		result[i, j] = tmp.as_array()
        
	return Image(result)