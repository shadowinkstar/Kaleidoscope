transform fullscreen_cover:
    xysize (config.screen_width, config.screen_height)
    fit "cover"           # 铺满屏幕，超出部分被裁
    xalign 0.5
    yalign 0.5

# -------------- 预计算几组常用尺寸 --------------
init python:
    W, H = config.screen_width, config.screen_height
    CHAR_W, CHAR_H = int(W * 0.45), int(H * 0.90)

# 角色通用缩放框
transform charbox:
    xysize (CHAR_W, CHAR_H)
    fit "contain"
    anchor (0.5, 1.0)
    yalign 1.0

# 三个站位 —— 用“transform-expression”引入 charbox，再额外设 xalign
transform char_left:
    charbox          # ← 关键：单独一行即可继承
    xalign 0.20

transform char_center:
    charbox
    xalign 0.50

transform char_right:
    charbox
    xalign 0.80

