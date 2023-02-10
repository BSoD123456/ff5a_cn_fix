#! python3
# coding: utf-8

try:
    from cnocr import CnOcr
except:
    print('''
please install Pillow with
pip3 install cnocr
or
pip install cnocr
or
py -3 -m pip install cnocr
or
python -m pip install cnocr
''')
    raise

class ff5a_ocr_parser:

    def __init__(self, txt_parser):
        self.tpsr = txt_parser

    def parse(self):
        self.ocr = CnOcr(det_model_name='naive_det')

    def draw_text(self, tidxs, pad = 3):
        blks = []
        for tidx in tidxs:
            blks.append(self.tpsr.draw_text(tidx, pad_col = pad))
        blk = self.tpsr.draw.draw_horiz(*blks, pad = pad)
        return self.tpsr.draw.make_img(blk)

if __name__ == '__main__':
    
    from ff5a_parser import c_ff5a_parser
    
    def main():
        psr = c_ff5a_parser('ff5acn.gba')
        psr.load_rom()
        psr.parse()
        ocr = ff5a_ocr_parser(psr.txt_parser['cn'])
        return ocr
    ocr = main()
    
