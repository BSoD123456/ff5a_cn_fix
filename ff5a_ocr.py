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

def report(*args):
    r = ' '.join(args)
    print(r)
    return r

class c_map_guesser:

    MAX_LA_WARN = 3
    MAX_LA_SKIP = 6

    def __init__(self):
        self.det = {}
        self.det_r = {}
        self.nondet = {}
        self.nondet_r = {}
        self.cnflct = {}

    def _guess_match(self, c1, i1, c2, i2):
        pass

    def feed(self, s1, s2):
        l1 = len(s1)
        l2 = len(s2)
        i1 = 0
        i2 = 0
        while i1 < l1 and i2 < l2:
            c1 = s1[i1]
            c2 = s2[i2]
            if c1 in self.det:
                # known c1
                c1r = self.det[c1]
                if c1r == c2:
                    # matched, bypass
                    i1 += 1
                    i2 += 1
                    continue
                # find matched char in s2 next
                _i2dlt = 0
                for _i2 in range(i2 + 1, min(l2, i2 + self.MAX_LA_SKIP)):
                    _c2 = s2[_i2]
                    if _c2 == c1r:
                        _i2dlt = _i2 - i2
                        break
                if _i2dlt > 0:
                    if _i2dlt >= self.MAX_LA_WARN:
                        report('warning',
                            f's1 lookahead too mush {i2}+{_i2dlt}/{self.MAX_LA_WARN}~{self.MAX_LA_SKIP}')
                    i1 += 1
                    i2 += _i2dlt + 1
                    continue
                # no matched char found, skip c1
                i1 += 1
                continue
            # unknown c1
            if c2 in self.det_r:
                # known c2
                c2r = self.det_r[c2]
                assert c2r != c1
                # find matched char in s1 next
                _i1dlt = 0
                for _i1 in range(i1 + 1, min(l1, i1 + self.MAX_LA_SKIP)):
                    _c1 = s1[_i1]
                    if _c1 == c2r:
                        _i1dlt = _i1 - i1
                        break
                if _i1dlt > 0:
                    if _i1dlt >= self.MAX_LA_WARN:
                        report('warning',
                            f's2 lookahead too mush {i1}+{_i1dlt}/{self.MAX_LA_WARN}~{self.MAX_LA_SKIP}')
                    i2 += 1
                    i1 += _i1dlt + 1
                    continue
                # no matched char found, skip c2
                i2 += 1
                continue
            # both c1 c2 unknown
            self._guess_match(c1, i1, c2, i2)
            i1 += 1
            i2 += 2

class c_ff5a_ocr_parser:

    def __init__(self, txt_parser):
        self.tpsr = txt_parser

    def parse(self):
        self.ocr = CnOcr(det_model_name='naive_det')

    def draw_chars(self, chars, pad = 3):
        blk = self.tpsr.draw_chars(chars, pad_col = pad)
        return self.tpsr.draw.make_img(blk)

    def ocr_chars(self, chars, ret_img = False):
        im = self.draw_chars(chars)
        rinfo = self.ocr.ocr(im)
        rchars = ''.join(i['text'] for i in rinfo)
        if ret_img:
            return rchars, im
        else:
            return rchars

    def pick_text(self, tidxs, txt = None):
        if txt is None:
            txt = []
        for tidx in tidxs:
            t = self.tpsr.get_text(tidx)
            for c in t:
                if not self.tpsr.is_ctrl(c):
                    txt.append(c)
        return txt

    def pick_cn_texts(self, st, cnt, txt = None):
        pass

if __name__ == '__main__':
    
    from ff5a_parser import c_ff5a_parser
    
    def main():
        psr = c_ff5a_parser('ff5acn.gba')
        psr.load_rom()
        psr.parse()
        ocr = c_ff5a_ocr_parser(psr.txt_parser['cn'])
        ocr.parse()
        return ocr
    ocr = main()
    rtxt, stxt, im = ocr.ocr_text(range(1801, 1821, 2))
    
