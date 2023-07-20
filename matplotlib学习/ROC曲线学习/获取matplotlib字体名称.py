import matplotlib.font_manager as ftm

fm = ftm.fontManager
for font in fm.ttflist:
    if 'Hei' in font.name:
        print(font.name)
