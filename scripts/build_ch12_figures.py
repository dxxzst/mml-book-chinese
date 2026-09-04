import pymupdf
import os
from PIL import Image, ImageDraw, ImageFont

doc = pymupdf.open(r'E:\需求记录\文档资料\Books\mml-book_printed.pdf')
out_dir = 'docs/images'
os.makedirs(out_dir, exist_ok=True)

def render_page_rect(page_num, rect, zoom=4.0):
    page = doc[page_num - 1]
    mat = pymupdf.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, clip=rect)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img

def get_font(size):
    for f in ['msyh.ttc', 'simsun.ttc', 'simhei.ttf', 'arial.ttf']:
        try:
            return ImageFont.truetype(f, size)
        except:
            pass
    return ImageFont.load_default()

# -------------------------------------------------------------
# Figure 12.1 (P377)
# -------------------------------------------------------------
im12_1 = render_page_rect(377, pymupdf.Rect(165, 120, 320, 265))
im12_1.save(os.path.join(out_dir, 'Figure12.1.png'))
print("Figure 12.1 saved:", im12_1.size)

# -------------------------------------------------------------
# Figure 12.2 (P379)
# y0=120 to eliminate running header
# -------------------------------------------------------------
im12_2 = render_page_rect(379, pymupdf.Rect(75, 120, 410, 252))
draw12_2 = ImageDraw.Draw(im12_2)

# (a) Subcaption: y: 436
draw12_2.rectangle([80, 430, 600, 485], fill=(255, 255, 255))
draw12_2.text((120, 436), '(a) 三维空间中的分隔超平面', fill=(0, 0, 0), font=get_font(24))

# (b) Subcaption:
draw12_2.rectangle([700, 430, 1320, 520], fill=(255, 255, 255))
draw12_2.text((740, 436), '(b) 将 (a) 中的设定投影到平面上', fill=(0, 0, 0), font=get_font(24))

# Positive / Negative labels in (b)
draw12_2.rectangle([1140, 245, 1270, 290], fill=(255, 255, 255))
draw12_2.text((1150, 250), '正例', fill=(0, 0, 0), font=get_font(22))

draw12_2.rectangle([880, 355, 1025, 395], fill=(255, 255, 255))
draw12_2.text((900, 358), '负例', fill=(0, 0, 0), font=get_font(22))

im12_2.save(os.path.join(out_dir, 'Figure12.2.png'))
print("Figure 12.2 saved:", im12_2.size)

# -------------------------------------------------------------
# Figure 12.3 (P380)
# -------------------------------------------------------------
im12_3 = render_page_rect(380, pymupdf.Rect(235, 120, 395, 265))
im12_3.save(os.path.join(out_dir, 'Figure12.3.png'))
print("Figure 12.3 saved:", im12_3.size)

# -------------------------------------------------------------
# Figure 12.4 (P381) - tighten x1 to 330
# -------------------------------------------------------------
im12_4 = render_page_rect(381, pymupdf.Rect(170, 120, 330, 248))
im12_4.save(os.path.join(out_dir, 'Figure12.4.png'))
print("Figure 12.4 saved:", im12_4.size)

# -------------------------------------------------------------
# Figure 12.5 (P382)
# -------------------------------------------------------------
im12_5 = render_page_rect(382, pymupdf.Rect(205, 120, 415, 270))
im12_5.save(os.path.join(out_dir, 'Figure12.5.png'))
print("Figure 12.5 saved:", im12_5.size)

# -------------------------------------------------------------
# Figure 12.6 (P385) - y0=120 to eliminate running header, y1=268 to eliminate body text sliver
# -------------------------------------------------------------
im12_6 = render_page_rect(385, pymupdf.Rect(80, 120, 410, 268))
draw12_6 = ImageDraw.Draw(im12_6)

# (a) Subcaption
draw12_6.rectangle([30, 540, 620, 615], fill=(255, 255, 255))
draw12_6.text((60, 550), '(a) 具有大间隔的线性可分数据', fill=(0, 0, 0), font=get_font(24))

# (b) Subcaption
draw12_6.rectangle([700, 540, 1280, 615], fill=(255, 255, 255))
draw12_6.text((750, 550), '(b) 线性不可分数据', fill=(0, 0, 0), font=get_font(24))

im12_6.save(os.path.join(out_dir, 'Figure12.6.png'))
print("Figure 12.6 saved:", im12_6.size)

# -------------------------------------------------------------
# Figure 12.7 (P386)
# -------------------------------------------------------------
im12_7 = render_page_rect(386, pymupdf.Rect(205, 120, 415, 270))
im12_7.save(os.path.join(out_dir, 'Figure12.7.png'))
print("Figure 12.7 saved:", im12_7.size)

# -------------------------------------------------------------
# Figure 12.8 (P388)
# Border of plot is at x=682 in crop coordinates (rx0=220, rx1=415)
# Mask rectangle must NOT exceed x=675
# -------------------------------------------------------------
im12_8 = render_page_rect(388, pymupdf.Rect(220, 125, 415, 248))
draw12_8 = ImageDraw.Draw(im12_8)

# 'Zero-one loss' at box=(452, 62, 648, 99)
draw12_8.rectangle([448, 58, 672, 102], fill=(255, 255, 255))
draw12_8.text((450, 62), '0-1 损失', fill=(0, 0, 0), font=get_font(22))

# 'Hinge loss' at box=(452, 115, 605, 152)
draw12_8.rectangle([448, 112, 672, 156], fill=(255, 255, 255))
draw12_8.text((450, 116), '合页损失（Hinge 损失）', fill=(0, 0, 0), font=get_font(20))

im12_8.save(os.path.join(out_dir, 'Figure12.8.png'))
print("Figure 12.8 saved:", im12_8.size)

# -------------------------------------------------------------
# Figure 12.9 (P392)
# -------------------------------------------------------------
im12_9 = render_page_rect(392, pymupdf.Rect(145, 120, 480, 325))
draw12_9 = ImageDraw.Draw(im12_9)

# Subcaption (a)
draw12_9.rectangle([180, 625, 460, 680], fill=(255, 255, 255))
draw12_9.text((220, 635), '(a) 凸包', fill=(0, 0, 0), font=get_font(24))

# Subcaption (b)
draw12_9.rectangle([650, 625, 1340, 810], fill=(255, 255, 255))
draw12_9.text((680, 635), '(b) 正例（蓝色）与负例（红色）各自的凸包', fill=(0, 0, 0), font=get_font(24))

im12_9.save(os.path.join(out_dir, 'Figure12.9.png'))
print("Figure 12.9 saved:", im12_9.size)

# -------------------------------------------------------------
# Figure 12.10 (P395) - 4 subplots!
# Crop: rx0=70, ry0=118, rx1=405, ry1=476
# -------------------------------------------------------------
im12_10 = render_page_rect(395, pymupdf.Rect(70, 118, 405, 476))
draw12_10 = ImageDraw.Draw(im12_10)

# 1. First feature labels (bottom of each subplot)
# Subplot (a) x-axis: pix=(270, 582, 440, 616)
draw12_10.rectangle([250, 580, 450, 620], fill=(255, 255, 255))
draw12_10.text((270, 585), '第一个特征', fill=(0, 0, 0), font=get_font(22))

# Subplot (b) x-axis: pix=(945, 582, 1115, 616)
draw12_10.rectangle([925, 580, 1130, 620], fill=(255, 255, 255))
draw12_10.text((945, 585), '第一个特征', fill=(0, 0, 0), font=get_font(22))

# Subplot (c) x-axis: pix=(270, 1312, 440, 1346)
draw12_10.rectangle([250, 1310, 450, 1350], fill=(255, 255, 255))
draw12_10.text((270, 1315), '第一个特征', fill=(0, 0, 0), font=get_font(22))

# Subplot (d) x-axis: pix=(945, 1312, 1115, 1346)
draw12_10.rectangle([925, 1310, 1130, 1350], fill=(255, 255, 255))
draw12_10.text((945, 1315), '第一个特征', fill=(0, 0, 0), font=get_font(22))

# 2. Second feature labels (rotated vertical)
# Subplot (a) y-axis: pix=(28, 197, 63, 398)
draw12_10.rectangle([15, 190, 60, 405], fill=(255, 255, 255))
v_img_a = Image.new('RGB', (200, 35), (255, 255, 255))
ImageDraw.Draw(v_img_a).text((10, 2), '第二个特征', fill=(0, 0, 0), font=get_font(22))
v_rot_a = v_img_a.rotate(90, expand=True)
im12_10.paste(v_rot_a, (18, 200))

# Subplot (b) y-axis: gutter is 660..780, text is at 704..738
draw12_10.rectangle([685, 190, 735, 405], fill=(255, 255, 255))
v_img_b = Image.new('RGB', (200, 35), (255, 255, 255))
ImageDraw.Draw(v_img_b).text((10, 2), '第二个特征', fill=(0, 0, 0), font=get_font(22))
v_rot_b = v_img_b.rotate(90, expand=True)
im12_10.paste(v_rot_b, (695, 200))

# Subplot (c) y-axis: pix=(28, 926, 63, 1127)
draw12_10.rectangle([15, 920, 60, 1135], fill=(255, 255, 255))
v_img_c = Image.new('RGB', (200, 35), (255, 255, 255))
ImageDraw.Draw(v_img_c).text((10, 2), '第二个特征', fill=(0, 0, 0), font=get_font(22))
v_rot_c = v_img_c.rotate(90, expand=True)
im12_10.paste(v_rot_c, (18, 930))

# Subplot (d) y-axis: pix=(704, 926, 738, 1127)
draw12_10.rectangle([685, 920, 735, 1135], fill=(255, 255, 255))
v_img_d = Image.new('RGB', (200, 35), (255, 255, 255))
ImageDraw.Draw(v_img_d).text((10, 2), '第二个特征', fill=(0, 0, 0), font=get_font(22))
v_rot_d = v_img_d.rotate(90, expand=True)
im12_10.paste(v_rot_d, (695, 930))

# 3. Subcaptions (a), (b), (c), (d)
# (a)
draw12_10.rectangle([120, 640, 550, 690], fill=(255, 255, 255))
draw12_10.text((150, 648), '(a) 线性核 SVM', fill=(0, 0, 0), font=get_font(24))

# (b)
draw12_10.rectangle([800, 640, 1220, 690], fill=(255, 255, 255))
draw12_10.text((820, 648), '(b) RBF 核（高斯核）SVM', fill=(0, 0, 0), font=get_font(24))

# (c)
draw12_10.rectangle([20, 1370, 650, 1420], fill=(255, 255, 255))
draw12_10.text((50, 1378), '(c) 2 次多项式核 SVM', fill=(0, 0, 0), font=get_font(24))

# (d)
draw12_10.rectangle([700, 1370, 1330, 1420], fill=(255, 255, 255))
draw12_10.text((730, 1378), '(d) 3 次多项式核 SVM', fill=(0, 0, 0), font=get_font(24))

im12_10.save(os.path.join(out_dir, 'Figure12.10.png'))
print("Figure 12.10 saved:", im12_10.size)
