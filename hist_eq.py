from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# загружаем исходное изображение
image = Image.open(r'/content/deti.jpg')

# разделение изображения на 3 компоненты
r, g, b = image.split() 
r = np.array(r)
g = np.array(g)
b = np.array(b)

# переводим в YCbCr
Y = np.round(0.299*r + 0.587*g + 0.114*b)
Y = np.clip(Y, 0, 255)
Cb = np.round(-0.1687*r - 0.3313*g + 0.5*b + 128)
Cb = np.clip(Cb, 0, 255)
Cr = np.round(0.5*r - 0.4187*g - 0.0813*b + 128)
Cr = np.clip(Cr, 0, 255)

# сохраняем только яркостную компоненту
new_Y = Image.fromarray(Y.astype('uint8'))
YCbCr = (new_Y, new_Y, new_Y)
out_img = Image.merge('RGB', YCbCr)
out_img.save("Y.jpg")

# создает гистограмму изображения
def create_hist(Y):
  Y_hist = np.zeros(256)
  for i in range(len(Y)):
    for j in range(len(Y[0])):
      Y_hist[int(Y[i][j])] += 1
  return Y_hist

# считает вероятностное распределение
def cdf(x, hist):
  cdf_x = 0
  for i in range(x):
    cdf_x += hist[i]
  return int(cdf_x)

def create_LU_table(cdf_x, num_of_pixs):
  lookup_table = np.zeros((256, 2))
  for x in range(len(lookup_table)):
    lookup_table[x][0] = x
    lookup_table[x][1] = np.clip(np.round(255*cdf_x[x]/num_of_pixs), 0, 255)
  return lookup_table

Y_hist = create_hist(Y)
x_range = range(0, 256)
plt.plot(x_range, Y_hist, 'r')
plt.show()

num_of_pixs = len(Y)*len(Y[0])
norm_hist = np.divide(Y_hist, num_of_pixs)
plt.plot(x_range, norm_hist, 'r')
plt.show()

cdf_x = []
for x in x_range:
  cdf_x.append(cdf(x, Y_hist))

plt.plot(x_range, cdf_x, 'r')

# выравнивание гистограммы всего изображения
lookup_table = create_LU_table(cdf_x, num_of_pixs)
s_x = np.array(Y)
for i in range(len(s_x)):
  for j in range(len(s_x[0])):
    s_x[i][j] = lookup_table[int(s_x[i][j])][1]

# адаптивное выравнивание гистограммы
s_x_block = np.array(Y)
block_size_x = 96
block_size_y = 62
for i in range(0, len(s_x_block), block_size_x):
  for j in range(0, len(s_x_block[0]), block_size_y):
    block = s_x_block[i:i+block_size_x, j:j+block_size_y]
    block_hist = create_hist(block)
    cdf_block = []
    for x in x_range:
      cdf_block.append(cdf(x, block_hist))
    LU_block = create_LU_table(cdf_block, (block_size_x)*(block_size_y))
    for x in range(len(block)):
      for y in range(len(block[0])):
        s_x_block[i + x][j + y] = LU_block[int(block[x][y])][1]

# строим гистограммы получившихся изображений
f_hist = np.zeros(256)
f_hist_block = np.zeros(256)
for i in range(len(s_x)):
  for j in range(len(s_x[0])):
    f_hist[int(s_x[i][j])] += 1
    f_hist_block[int(s_x_block[i][j])] += 1

plt.plot(x_range, f_hist, 'r', x_range, f_hist_block, 'b--')
plt.legend(['Normal', 'Block'])
plt.show()

# функции распределений для получившихся изображений
cdf_f = []
cdf_f_block = []
for x in x_range:
  cdf_f.append(cdf(x, f_hist))
  cdf_f_block.append(cdf(x, f_hist_block))

plt.plot(x_range, cdf_f, 'r', x_range, cdf_f_block, 'b--')
plt.legend(['Normal', 'Block'])

# сохраняем результаты
new_Y = Image.fromarray(s_x.astype('uint8'))
new_Cb = Image.fromarray(Cb.astype('uint8'))
new_Cr = Image.fromarray(Cr.astype('uint8'))
YCbCr = (new_Y, new_Cb, new_Cr)
out_img = Image.merge('YCbCr', YCbCr).convert('RGB')
out_img.save("new_photo.jpg")

YCbCr = (new_Y, new_Y, new_Y)
out_img = Image.merge('RGB', YCbCr)
out_img.save("new_Y.jpg")

new_Y = Image.fromarray(s_x_block.astype('uint8'))
YCbCr = (new_Y, new_Cb, new_Cr)
out_img = Image.merge('YCbCr', YCbCr).convert('RGB')
out_img.save("new_photo_block.jpg")