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
        txt = []
        for tidx in tidxs:
            t = self.tpsr.get_text(tidx)
            for c in t:
                if not self.tpsr.is_ctrl(c):
                    txt.append(c)
        blk = self.tpsr.draw_chars(txt, pad_col = pad)
        return self.tpsr.draw.make_img(blk), txt

    def ocr_text(self, tidxs):
        im, stxt = self.draw_text(tidxs)
        rinfo = self.ocr.ocr(im)
        rtxt = ''.join(i['text'] for i in rinfo)
        return rtxt, stxt, im

if __name__ == '__main__':
    
    from ff5a_parser import c_ff5a_parser
    
    def main():
        psr = c_ff5a_parser('ff5acn.gba')
        psr.load_rom()
        psr.parse()
        ocr = ff5a_ocr_parser(psr.txt_parser['cn'])
        ocr.parse()
        return ocr
    ocr = main()
    rtxt, stxt, im = ocr.ocr_text(range(1801, 1821, 2))
    
